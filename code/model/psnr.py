# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def calculate_psnr(img1, img2, normalize=False):
    if normalize:
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2

    mse = torch.mean((img1 - img2) ** 2)
    psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).to(mse))

    return psnr


class PSNR(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()

        self.normalize = normalize

    def forward(self, img1, img2):
        """

        Args:
            img1:
            img2:

        Returns:

        """
        psnr = calculate_psnr(img1, img2, self.normalize)

        return psnr
