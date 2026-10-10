# Third-party notices

The `sam2/modeling` directory and SAM2 configuration files are copied from the local SAM2 dependency used by the supplied project. Upstream copyright: Meta Platforms, Inc. and affiliates. The local copy contains project-specific modifications and is not represented as an unmodified upstream release.

`sam2/utils/misc.py` retains the upstream SDPA helper only. `marisam/modified_transformer.py` is an existing project adaptation of SAM transformer code. SAM-derived code is covered by the upstream Apache License 2.0; a copy is provided in `licenses/SAM2-APACHE-2.0.txt`. The root LICENSE repeats that license for the paths referenced by upstream source headers; it does not assign a license to original MariSAM modules.

Upstream projects: https://github.com/facebookresearch/sam2 and https://github.com/facebookresearch/segment-anything . Pretrained checkpoints are obtained separately and remain subject to their own distribution terms. No PlanetScope imagery is bundled.
