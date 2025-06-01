import torch
from torch import nn

from norm_ema_quantizer import NormEMAVectorQuantizer


class ResidualQuantizer(nn.Module):
    def __init__(self, num_quantizers, embedding_dim, n_embed, decay=0.99, beta=1.0):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.quantizers = nn.ModuleList([
            NormEMAVectorQuantizer(
                n_embed=n_embed,
                embedding_dim=embedding_dim,
                decay=decay,
                beta=beta,
                kmeans_init=True
            )
            for _ in range(num_quantizers)
        ])
    def forward(self, x):
        residual = x
        quantized_outputs = []
        losses = []
        indices = []

        for quantizer in self.quantizers:
            quantized, loss, index = quantizer(residual)
            residual = residual - quantized.detach() + quantized  # preserve gradient flow
            quantized_outputs.append(quantized)
            losses.append(loss)
            indices.append(index)

        quantized_sum = sum(quantized_outputs)
        total_loss = sum(losses)
        return quantized_sum, total_loss, torch.stack(indices, dim=1)
