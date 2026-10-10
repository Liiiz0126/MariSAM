"""Predict a georeferenced class-ID raster from one square multispectral patch."""
import argparse
from pathlib import Path
import tempfile


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, help='Checkpoint produced by marisam.train')
    p.add_argument('--image', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args(argv)
    import torch
    import rasterio
    from .data import read_image, normalize
    from .model import build_marisam
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    image, profile = read_image(args.image, saved['band_order'])
    with tempfile.TemporaryDirectory() as directory:
        cfg = Path(directory) / 'sam.yaml'
        cfg.write_text(saved['sam_config_text'], encoding='utf-8')
        model = build_marisam(cfg, device=args.device, **saved['model_options'])
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    tensor = normalize(image, saved['mean'], saved['std']).unsqueeze(0).to(args.device)
    with torch.no_grad():
        prediction = model(tensor).argmax(1)[0].cpu().numpy().astype('uint8')
    profile.update(driver='GTiff', count=1, dtype='uint8', nodata=None, compress='deflate')
    profile.pop('photometric', None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, 'w', **profile) as destination:
        destination.write(prediction, 1)
    print(f'Saved class IDs: {output.name}')


if __name__ == '__main__':
    main()
