"""Differentiable image-only wrapper around the current local SAM2 modules."""
import importlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import resize
from omegaconf import OmegaConf

from .autoprompt import AutoPrompt
from .lora import LoRALayer
from .spectral_adapter import UniversalSpectralStyleAdapter


def _instantiate(config):
    if isinstance(config, list):
        return [_instantiate(value) for value in config]
    if not isinstance(config, dict):
        return config
    values = {key: _instantiate(value) for key, value in config.items() if not key.startswith('_')}
    if '_target_' not in config:
        return values
    module, name = config['_target_'].rsplit('.', 1)
    return getattr(importlib.import_module(module), name)(**values)


def build_marisam(sam_config, sam_checkpoint=None, device='cpu', num_classes=2,
                  num_points=3, lora_rank=4, lora_alpha=8, adapter_channels=64,
                  transformer_dim=128, class_tokens=4, indices=(),
                  prompt_radius=6.0, prompt_threshold=0.2, prompt_temperature=1.5,
                  decoder_threshold=0.5, prompt_sampling='ids', prompt_mode='both'):
    cfg = OmegaConf.load(sam_config)
    OmegaConf.resolve(cfg)
    sam = _instantiate(OmegaConf.to_container(cfg.model, resolve=True))
    # The original builder enables this single-mask stability fallback.
    sam.sam_mask_decoder.dynamic_multimask_via_stability = True
    sam.sam_mask_decoder.dynamic_multimask_stability_delta = 0.05
    sam.sam_mask_decoder.dynamic_multimask_stability_thresh = 0.98
    if sam_checkpoint:
        checkpoint = torch.load(Path(sam_checkpoint), map_location='cpu', weights_only=True)
        sam.load_state_dict(checkpoint['model'], strict=True)
    LoRALayer.apply_lora_to_encoder(sam, r=lora_rank, alpha=lora_alpha)
    for parameter in sam.parameters():
        parameter.requires_grad = False
    for parameter in sam.sam_mask_decoder.parameters():
        parameter.requires_grad = True
    for name, parameter in sam.image_encoder.named_parameters():
        if 'lora' in name:
            parameter.requires_grad = True
    prompt = AutoPrompt(num_classes=num_classes, transformer_dim=transformer_dim,
                        embed_dim=256, num_class_tokens=class_tokens, num_points=num_points,
                        sam_feature_dim=256, high_res_dim=32, prompt_sampling=prompt_sampling,
                        prompt_mode=prompt_mode, nms_radius=prompt_radius,
                        min_threshold=prompt_threshold, sampling_temperature=prompt_temperature)
    adapter = UniversalSpectralStyleAdapter(in_bands=4, out_channels=3,
                                            mid_channels=adapter_channels,
                                            use_indices=bool(indices), indices_list=list(indices))
    return MariSAM(sam, prompt, adapter, decoder_threshold).to(device)


class MariSAM(nn.Module):
    """Input: normalized B, G, R, NIR tensors. Output: background/foreground logits."""
    def __init__(self, sam, autoprompt, imageadapter, decoder_threshold=0.5):
        super().__init__()
        self.sam_model = sam
        self.autoprompt = autoprompt
        self.imageadapter = imageadapter
        self.decoder_threshold = decoder_threshold

    def forward(self, image, target=None, return_aux=False):
        if image.ndim != 4 or image.shape[1] != 4:
            raise ValueError('Input must have shape [batch, 4, height, width] in B,G,R,NIR order.')
        if image.shape[-2] != image.shape[-1]:
            raise ValueError('Use square patches; the original AutoPrompt assumes square feature maps.')
        batch, _, height, width = image.shape
        sam_image = resize(self.imageadapter(image), [self.sam_model.image_size] * 2)
        backbone = self.sam_model.forward_image(sam_image)
        _, vision, _, sizes = self.sam_model._prepare_backbone_features(backbone)
        if self.sam_model.directly_add_no_mem_embed:
            vision[-1] = vision[-1] + self.sam_model.no_mem_embed
        features = [feature.permute(1, 2, 0).contiguous().view(batch, -1, *size)
                    for feature, size in zip(vision, sizes)]
        embedding, high_res = features[-1], features[:-1]
        prompt_encoder = self.sam_model.sam_prompt_encoder
        pe = prompt_encoder.get_dense_pe()
        point_types = torch.cat([prompt_encoder.not_a_point_embed.weight,
                                 prompt_encoder.point_embeddings[1].weight,
                                 prompt_encoder.point_embeddings[0].weight]).detach()
        sparse, dense, _, _, _, orth, auxiliary = self.autoprompt(
            image=embedding, image_pe=pe, high_res_features=high_res[0], points_type=point_types)
        foreground = []
        for cls, aux in enumerate(auxiliary):
            maximum = torch.sigmoid(aux.detach()).max()
            confident = bool(maximum > self.decoder_threshold) if self.training else bool(maximum >= self.decoder_threshold)
            present = target is not None and bool((target == cls + 1).any())
            if present or confident:
                masks, _, _, _ = self.sam_model.sam_mask_decoder(
                    image_embeddings=embedding, image_pe=pe,
                    sparse_prompt_embeddings=sparse[cls], dense_prompt_embeddings=dense[cls],
                    multimask_output=False, repeat_image=False, high_res_features=high_res)
                masks = F.interpolate(masks, (height, width), mode='bilinear', align_corners=False)
            else:
                masks = embedding.new_full((batch, 1, height, width), -100.0)
            foreground.append(masks)
        foreground = torch.cat(foreground, dim=1)
        logits = torch.cat([-foreground.amax(dim=1, keepdim=True), foreground], dim=1)
        return (logits, auxiliary, orth) if return_aux else logits

    def load_legacy_weights(self, checkpoint):
        """Load a mapping containing the original three state-dict keys."""
        self.sam_model.load_state_dict(checkpoint['sam_model'], strict=True)
        self.autoprompt.load_state_dict(checkpoint['autoprompt'], strict=True)
        self.imageadapter.load_state_dict(checkpoint['imageadapter'], strict=True)
