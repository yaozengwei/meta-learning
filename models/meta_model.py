#!/usr/bin/env python3
# Copyright         2023  Xiaomi Corp.        (authors: Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import combinations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaModel(nn.Module):
    def __init__(self, in_channels: int = 6, hidden_channels: int = 32) -> None:
        # in_channels = 6 since we take all 2-element combinations in 4 elements
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        self.proj = nn.Linear(hidden_channels, 1, bias=False)
        # I found that initializing the weights to zero cause instability at the start,
        # so I just don't add the meta_loss to the main_loss before the meta_model learned something.
        # self.proj.weight.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)

        out = out.mean(dim=(2, 3))  # (B, C_)
        out = self.proj(out) ** 2  # (B, 1)
        out = out.mean()

        return out


class Model(nn.Module):
    def __init__(
        self, main_model: nn.Module, meta_model: nn.Module, mask_ratio: float = 0.5
    ) -> None:
        super().__init__()
        self.main_model = main_model
        self.meta_model = meta_model

        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        B = x.shape[0] // 2
        embeddings = self.main_model.forward_embedding(x)

        # used to compute the main loss
        main_out = self.main_model.head(embeddings)  # (B * 2, num_classes)

        _, C_, H_, W_ = embeddings.shape
        embeddings = embeddings.view(B // 2, 4, C_, H_, W_)
        embeddings = embeddings.unsqueeze(1).expand(B // 2, 6, 4, C_, H_, W_)

        indexes = torch.tensor(list(combinations(range(4), 2)), device=x.device)
        indexes = indexes.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5)
        indexes = indexes.expand(B // 2, 6, 2, C_, H_, W_)

        # (B // 2, 6, 2, C_, H_, W_)
        embeddings_pairs = torch.gather(embeddings, dim=2, index=indexes)

        inner_prod = (
            embeddings_pairs[:, :, 0, :, :, :] * embeddings_pairs[:, :, 1, :, :, :]
        ).sum(dim=2)  # (B // 2, 6, H_, W_)

        aux_loss = self.meta_model(inner_prod)

        return main_out, aux_loss
