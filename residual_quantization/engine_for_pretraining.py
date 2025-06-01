# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

from cgitb import enable
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from einops import rearrange
from contextlib import nullcontext


def nt_xent_loss(z1, z2, temperature=0.2):
    """
    z1, z2 : (B, D) already L2-normalised.
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -float("inf"))

    targets = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return F.cross_entropy(sim, targets)


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    # mask = np.hstack([
    #     np.zeros(len_keep),
    #     np.ones(L - len_keep),
    # ])
    # np.random.shuffle(mask)

    return mask.to(torch.bool)

# --------------------------------------------------------
# engine_for_pretraining.py
# full, drop-in replacement for train_one_epoch
# --------------------------------------------------------

def train_one_epoch(model: torch.nn.Module, vqnsp: torch.nn.Module,
                    data_loader_list: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    max_norm: float = 0, log_writer=None, lr_scheduler=None,
                    start_steps: int = 0, lr_schedule_values=None,
                    wd_schedule_values=None, ch_names_list=None, args=None):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr',      utils.SmoothedValue(1, '{value:.6f}'))
    metric_logger.add_meter('min_lr',  utils.SmoothedValue(1, '{value:.6f}'))
    header, print_freq = f"Epoch: [{epoch}]", 10

    ce = nn.CrossEntropyLoss()
    step_loader = 0


    n_q = getattr(vqnsp.quantize, 'num_quantizers', 1)           # Q

    for loader, ch_names in zip(data_loader_list, ch_names_list):
        if len(loader) == 0:
            continue

        in_chans = utils.get_input_chans(ch_names)

        for step, (eeg, _) in enumerate(
                metric_logger.log_every(loader, print_freq * args.gradient_accumulation_steps, header)):

            it = start_steps + step + step_loader
            for g in optimizer.param_groups:          # step-wise LR / WD
                if lr_schedule_values is not None:
                    g['lr'] = lr_schedule_values[it] * g['lr_scale']
                if wd_schedule_values is not None and g['weight_decay'] > 0:
                    g['weight_decay'] = wd_schedule_values[it]

            # ------------- 1.   prepare inputs --------------------------------
            eeg = rearrange(eeg.float().to(device, non_blocking=True) / 100,
                            'B N (A T) -> B N A T', T=200)
            B = eeg.size(0)

            # mask *patch* tokens (shape [(B,  N·A),   ]) – identical for every code-book head
            bool_mask_patch = random_masking(eeg.flatten(1, 2), mask_ratio=0.5).to(device)

            # ------------- 2.   get ground-truth ids from tokenizer ------------
            with torch.no_grad(), torch.cuda.amp.autocast():
                code_ids = vqnsp.get_codebook_indices(eeg, in_chans)      # [B,   L_total]

            L_total = code_ids.size(1)
            assert L_total % n_q == 0, "tokenizer length not divisible by #quantizers"
            L = L_total // n_q                                            # tokens / quantizer

            # reshape  →  [B,  Q,  L]
            code_ids = code_ids.view(B, n_q, L)

            # expand the *patch* mask to all codebooks:  [B, Q, L]
            mask_full = bool_mask_patch.unsqueeze(1).expand(-1, n_q, -1)

            tgt_masked   = code_ids[ mask_full]   # 1-D (♯masked·Q)
            tgt_unmasked = code_ids[~mask_full]   # 1-D (♯un-masked·Q)

            # ------------- 3.   forward ---------------------------------------
            my_ctx = model.no_sync if args.distributed and \
                     (step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
            with my_ctx(), torch.cuda.amp.autocast():

                # model is asked to predict only the *patch* positions
                # it normally returns:
                #   x_rec      – logits for masked tokens
                #   x_rec_sym  – logits for the symmetric (un-masked) tokens
                x_rec, x_rec_sym = model(eeg, in_chans, bool_masked_pos=bool_mask_patch)

                # make sure shapes are [♯masked, Q, V]
                if x_rec.dim()  == 2:  # [♯masked·Q, V]  →  [♯masked, Q, V]
                    V = x_rec.size(1)
                    x_rec     = x_rec    .view(-1, n_q, V)
                    x_rec_sym = x_rec_sym.view(-1, n_q, V)

                # flatten head-dim for CE
                x_rec      = x_rec    .flatten(0, 1)                      # [♯masked·Q, V]
                x_rec_sym  = x_rec_sym.flatten(0, 1)                      # [♯un-masked·Q, V]

                loss_m     = ce(x_rec,     tgt_masked)
                loss_um    = ce(x_rec_sym, tgt_unmasked)
                loss       = loss_m + loss_um

            # ------------- 4.   bookkeeping / optimisation --------------------
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                print(f"Loss is {loss_val}, stopping (rank {utils.get_rank()})", force=True)
                sys.exit(1)

            loss /= args.gradient_accumulation_steps
            is_2nd = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(),
                                    create_graph=is_2nd,
                                    update_grad=(step + 1) % args.gradient_accumulation_steps == 0)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            # ---- metrics -----------------------------------------------------
            with torch.no_grad():
                acc_m  = (x_rec    .max(-1)[1] == tgt_masked  ).float().mean().item()
                acc_um = (x_rec_sym.max(-1)[1] == tgt_unmasked).float().mean().item()

            metric_logger.update(loss=loss_val,
                                 mlm_acc=acc_m,
                                 mlm_acc_sym=acc_um,
                                 loss_scale=loss_scaler.state_dict()["scale"],
                                 grad_norm=grad_norm)

            max_lr = max(g['lr'] for g in optimizer.param_groups)
            min_lr = min(g['lr'] for g in optimizer.param_groups)
            wd_val = next((g['weight_decay'] for g in optimizer.param_groups
                           if g['weight_decay'] > 0), None)

            metric_logger.update(lr=max_lr, min_lr=min_lr, weight_decay=wd_val)

            if log_writer is not None:
                log_writer.update(loss=loss_val,              head="loss")
                log_writer.update(mlm_acc=acc_m,              head="loss")
                log_writer.update(mlm_acc_sym=acc_um,         head="loss")
                log_writer.update(lr=max_lr, min_lr=min_lr,   head="opt")
                log_writer.update(grad_norm=grad_norm,        head="opt")
                log_writer.update(loss_scale=metric_logger.meters['loss_scale'].value, head="opt")
                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(it)

        step_loader += step

    # ---------------- aggregate across ranks & return ------------------------
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: m.global_avg for k, m in metric_logger.meters.items()}
