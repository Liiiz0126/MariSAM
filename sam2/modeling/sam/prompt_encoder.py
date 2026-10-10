# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.position_encoding import PositionEmbeddingRandom

from sam2.modeling.sam2_utils import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            image_embedding_size: Tuple[int, int],
            input_image_size: Tuple[int, int],
            mask_in_chans: int,
            activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )

        # [Critical] This block processes dense masks
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

        # [NEW] Adapter for AutoPrompt Dense Features
        # If your AutoPrompt outputs 256-dim features directly, we might need a
        # simple adapter to ensure it matches what MaskDecoder expects.
        # However, usually MaskDecoder expects [B, 256, 64, 64].
        # If AutoPrompt output is already this shape, we can pass it through identity
        # or a lightweight projection if you want to refine it.
        # Here we assume AutoPrompt output matches `mask_in_chans` or `embed_dim`.

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
            self,
            points: torch.Tensor,
            labels: torch.Tensor,
            pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
            self,
            points: Optional[Tuple[torch.Tensor, torch.Tensor]],
            boxes: Optional[torch.Tensor],
            masks: Optional[torch.Tensor],
            sparse_auto_prompts: Optional[torch.Tensor],  # Renamed for clarity
            dense_auto_prompts: Optional[torch.Tensor],  # Renamed for clarity
    ) -> int:
        if points is not None:
            return points[0].shape[0]
        elif sparse_auto_prompts is not None:
            return sparse_auto_prompts.shape[0]
        elif dense_auto_prompts is not None:
            return dense_auto_prompts.shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
            self,
            points: Optional[Tuple[torch.Tensor, torch.Tensor]],
            boxes: Optional[torch.Tensor],
            masks: Optional[torch.Tensor],

            # [NEW] Inputs from AutoPrompt
            sparse_auto_prompts: Optional[torch.Tensor],  # [B, N, C]
            dense_auto_prompts: Optional[torch.Tensor],  # [B, 256, H, W]

            use_auto_points: bool = True,
            use_auto_dense: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modified forward to accept pre-computed embeddings from AutoPrompt.
        """
        bs = self._get_batch_size(points, boxes, masks, sparse_auto_prompts, dense_auto_prompts)

        # 1. Sparse Embeddings (Points/Boxes)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )

        # A. Add AutoPrompt Points
        if sparse_auto_prompts is not None and use_auto_points:
            # Assumes sparse_auto_prompts is already [B, N, 256] embedding
            sparse_embeddings = torch.cat([sparse_embeddings, sparse_auto_prompts], dim=1)

        # B. Add Manual Points (if any)
        if points is not None:
            coords, labels = points
            # Only pad if we have NO other prompts (boxes or auto points)
            do_pad = (boxes is None and (sparse_auto_prompts is None or not use_auto_points))
            point_embeddings = self._embed_points(coords, labels, pad=do_pad)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        # C. Add Boxes
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        # 2. Dense Embeddings (Masks)
        # Priority: Manual Mask > AutoPrompt Dense > No Mask Embed

        if masks is not None:
            # Case 1: Manual binary mask input (needs downscaling)
            dense_embeddings = self._embed_masks(masks)

        elif dense_auto_prompts is not None and use_auto_dense:
            # Case 2: AutoPrompt Feature Map (Already [B, 256, H, W])
            # We assume AutoPrompt output is already the correct dimension.
            # If AutoPrompt outputs 128-dim, you might want to project it here.
            dense_embeddings = dense_auto_prompts

        else:
            # Case 3: No dense input, use learnable "no_mask_embed"
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings
