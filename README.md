# MariSAM

Core code for multispectral offshore mariculture segmentation using residual spectral adaptation, class-aware automatic prompting, Iterative Dispersed Sampling, and a SAM2 backbone with LoRA.

This source release follows the modules called by the current training entry point. It retains the local SAM2 implementation and extracts the active AutoPrompt and UniversalSpectralStyleAdapter definitions. Historical variants, baseline models, plotting scripts, manuscripts, data, pretrained weights, and local experiment outputs are excluded.

## Install

Use Python 3.10 or later. Install matching PyTorch and torchvision builds for your CPU/CUDA environment first, then install the other dependencies:

```bash
python -m pip install -r requirements.txt
python -m marisam.train --help
python -m marisam.predict --help
```

Run commands from the repository root. The bundled `sam2` package is required; do not replace it with an unrelated upstream installation. No Hydra initialization or CUDA extension compilation is needed for this image-only path. Obtain the corresponding SAM2 Hiera checkpoint separately from [Meta's official SAM2 repository](https://github.com/facebookresearch/sam2). Checkpoints and PlanetScope imagery are not redistributed here.

## Data contract

Prepare separate train/validation CSV manifests with `image,label` columns. Paths are relative to `--data-root`. For example:

```csv
image,label
images/sample_001.tif,labels/sample_001.tif
```

Each image contains four bands in **Blue, Green, Red, NIR** order. `--band-order` specifies the corresponding 1-based input raster bands. Raster labels use **0 = background, 1 = raft, 2 = cage**. Image and label must share CRS, transform, and square dimensions. Square patch size may differ from 256, but the SAM2 config controls the internal encoder resolution (1024 in the supplied configs).

Split geographic areas before tiling. Keep every tile inside its assigned area and do not reuse imagery across splits. This package does not silently create random splits. Provide normalization means and standard deviations estimated from the training subset only, in B,G,R,NIR order. The current adapter computes spectral indices from the normalized input, matching the supplied implementation; changing that preprocessing changes the experiment.

Vector polygons are not directly consumed by this training loader. Rasterize labels onto the selected authorized image grid and crop aligned image/label patches. Pixels outside the inspected annotation coverage must not automatically be treated as background.

## Train

All data locations and the learning rate must be passed explicitly. The following command uses example placeholders; replace the four normalization values with your training statistics. The learning rate is illustrative, not a verified manuscript configuration.

```bash
python -m marisam.train \
  --sam-config configs/sam2_hiera_s.yaml \
  --sam-checkpoint checkpoints/sam2_hiera_small.pt \
  --data-root data \
  --train-manifest data/train.csv --val-manifest data/val.csv \
  --output-dir outputs/run1 \
  --mean 0 0 0 0 --std 1 1 1 1 \
  --lr 0.0005 --epochs 200 --batch-size 8 --num-points 5 \
  --indices r1 r2 r3 ndwi --lora-rank 4 --lora-alpha 8 \
  --device cuda --amp
```

PowerShell users can put the command on one line or use PowerShell's line-continuation syntax. The current source entry point defaults to three points and disabled indices; the example explicitly selects five points and four indices. Omit `--indices` or use it with no following values to disable indices. CLI options cover learning-rate groups, weight decay, warmup, seed, prompt settings, LoRA, loss weights, and optional flips/right-angle rotations. Learning-rate arguments are user-supplied hyperparameters; this is not automatic hyperparameter search.

The core loss retains cross-entropy, present-class Dice, auxiliary focal supervision, and class-token orthogonality. Class gating is evaluated across the batch, matching the current implementation. Validation uses batch size one. AMP is optional and only enabled for CUDA. Best/last checkpoints and JSONL metrics are saved under `--output-dir`; no charts are generated.

Fourier style augmentation and other legacy augmentation routines are outside this minimal release. This refactored entry point is not a claim that the manuscript's complete training pipeline or reported scores have been numerically reproduced. Specify the full experimental configuration when using it for reproduction.

## Predict

```bash
python -m marisam.predict --checkpoint outputs/run1/best.pt \
  --image data/images/sample_001.tif --output outputs/prediction.tif --device cuda
```

Prediction writes a georeferenced class-ID GeoTIFF. Use a trained MariSAM checkpoint, not the SAM2-only pretrained checkpoint. New checkpoints store architecture and normalization. The model method `load_legacy_weights(mapping)` supports the original `sam_model`, `autoprompt`, and `imageadapter` state-dict keys; reconstruct the matching architecture and preprocessing yourself for those legacy weights. Serialized checkpoints containing arbitrary Python objects are not automatically unpickled.

## Metrics

JSONL scores are fractions in [0,1]. `miou` averages valid class IoUs including background; `facility_iou` merges all foreground classes into facility presence. Evaluate each holdout region separately and take an unweighted mean of its metrics to reproduce a region-mean protocol. This metric naming is explicit; confirm that it matches the manuscript's evaluation code before interpreting older results.

## Labels-only dataset

An OMF vector annotation release can be used with separately licensed PlanetScope scenes. See [the dataset guide](docs/DATASET.md) for scene matching, coverage footprints, and optional published evaluation splits. Users may choose other patch sizes; the article's benchmark split still needs to be preserved for comparable evaluation. No imagery or annotation files are included in this code repository.

## Attribution and license

SAM2-derived files retain the upstream Apache 2.0 terms; see [third-party notices](THIRD_PARTY_NOTICES.md). No new open-source license is assigned to the authors' original MariSAM code in this preparation. A public release should state its chosen license after the authors decide the reuse terms.
