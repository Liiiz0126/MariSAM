# Release validation

Validated in an isolated Python 3.12 environment using PyTorch 2.5.1 CPU, torchvision 0.20.1, OmegaConf 2.3.0, NumPy 1.26.4, and Rasterio 1.4.3.

Checks completed:

- All Python sources parse and both command-line help interfaces work.
- The original SAM2 Hiera-small checkpoint loads with strict key matching.
- IDS selects separated peaks and returns finite coordinates for empty heatmaps.
- Full model forward and backward computation produces finite outputs and nonzero gradients for USSA, AutoPrompt, LoRA, and the mask decoder.
- A synthetic GeoTIFF pair passes the data loader, actual one-epoch training command, optimizer/scheduler, best/last checkpoint writing, and actual prediction command with georeferencing preserved.
- Inference also passes at the supplied default internal encoder resolution of 1024 with a 256-by-256 four-band input.
- Source hashes confirm the original project files were not modified.

The synthetic training check uses a reduced internal resolution for speed. These are functional checks; no dataset training, published-score reproduction, or GPU/AMP benchmark is claimed. The local transformer wrapper passes the actual feature-map dimensions instead of relying on its former hard-coded 64-by-64 reshape; the standard 1024-resolution path keeps the same shapes. An initialization print containing an emoji was removed for Windows console compatibility.
