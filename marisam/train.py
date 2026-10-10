"""Train the core model using externally specified data and hyperparameters."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sam-config', required=True)
    p.add_argument('--sam-checkpoint', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--train-manifest', required=True)
    p.add_argument('--val-manifest', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--mean', nargs=4, type=float, required=True)
    p.add_argument('--std', nargs=4, type=float, required=True)
    p.add_argument('--band-order', nargs=4, type=int, default=[1, 2, 3, 4])
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=512)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--sam-lr-factor', type=float, default=0.1)
    p.add_argument('--prompt-weight-decay', type=float, default=1e-3)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--warmup-factor', type=float, default=1e-3)
    p.add_argument('--lr-power', type=float, default=0.9)
    p.add_argument('--num-classes', type=int, default=2)
    p.add_argument('--num-points', type=int, default=3)
    p.add_argument('--lora-rank', type=int, default=4)
    p.add_argument('--lora-alpha', type=float, default=8)
    p.add_argument('--adapter-channels', type=int, default=64)
    p.add_argument('--transformer-dim', type=int, default=128)
    p.add_argument('--class-tokens', type=int, default=4)
    p.add_argument('--indices', nargs='*', default=[])
    p.add_argument('--prompt-radius', type=float, default=6)
    p.add_argument('--prompt-threshold', type=float, default=0.2)
    p.add_argument('--prompt-temperature', type=float, default=1.5)
    p.add_argument('--decoder-threshold', type=float, default=0.5)
    p.add_argument('--prompt-sampling', choices=['ids', 'topk', 'random', 'grid'], default='ids')
    p.add_argument('--prompt-mode', choices=['both', 'dense_only', 'sparse_only', 'none'], default='both')
    p.add_argument('--ce-weight', type=float, default=1)
    p.add_argument('--dice-weight', type=float, default=1)
    p.add_argument('--focal-weight', type=float, default=0.5)
    p.add_argument('--orth-weight', type=float, default=0.1)
    p.add_argument('--focal-alpha', type=float, default=0.25)
    p.add_argument('--focal-gamma', type=float, default=2)
    p.add_argument('--empty-class-weight', type=float, default=0.5)
    p.add_argument('--clip-grad', type=float, default=1)
    p.add_argument('--flip-probability', type=float, default=0)
    p.add_argument('--rotation-probability', type=float, default=0)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--resume', default=None)
    return p


def model_options(args):
    return {name: getattr(args, name) for name in [
        'num_classes', 'num_points', 'lora_rank', 'lora_alpha', 'adapter_channels',
        'transformer_dim', 'class_tokens', 'indices', 'prompt_radius', 'prompt_threshold',
        'prompt_temperature', 'decoder_threshold', 'prompt_sampling', 'prompt_mode']}


def main(argv=None):
    args = parser().parse_args(argv)
    if args.lr <= 0 or args.epochs <= 0 or not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError('Use positive lr/epochs and 0 <= warmup_epochs < epochs.')
    if min(args.batch_size, args.num_classes, args.num_points, args.lora_rank) < 1:
        raise ValueError('Batch size, classes, points and LoRA rank must be positive.')
    if any(not 0 <= p <= 1 for p in [args.flip_probability, args.rotation_probability]):
        raise ValueError('Augmentation probabilities must be between 0 and 1.')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .data import PatchDataset
    from .losses import segmentation_loss
    from .metrics import confusion_matrix, summarize
    from .model import build_marisam
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; explicitly use --device cpu if intended.')
    common = dict(root=args.data_root, mean=args.mean, std=args.std,
                  band_order=args.band_order, num_classes=args.num_classes)
    training = PatchDataset(args.train_manifest, **common,
                            flip_probability=args.flip_probability,
                            rotation_probability=args.rotation_probability)
    validation = PatchDataset(args.val_manifest, **common)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(validation, batch_size=1, num_workers=args.workers)
    options = model_options(args)
    model = build_marisam(args.sam_config, args.sam_checkpoint, device, **options)
    sam_params = [p for p in model.sam_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD([
        {'params': sam_params, 'lr': args.lr * args.sam_lr_factor},
        {'params': model.autoprompt.parameters(), 'lr': args.lr, 'weight_decay': args.prompt_weight_decay},
        {'params': model.imageadapter.parameters(), 'lr': args.lr}],
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    steps = len(train_loader)
    def rate(step):
        warmup = args.warmup_epochs * steps
        if warmup and step <= warmup:
            fraction = step / warmup
            return args.warmup_factor * (1 - fraction) + fraction
        progress = (step - warmup) / ((args.epochs - args.warmup_epochs) * steps)
        return max(0.0, 1 - progress) ** args.lr_power
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    start, best = 0, float('-inf')
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        if saved['model_options'] != options or saved['mean'] != args.mean or saved['std'] != args.std:
            raise ValueError('Resume model/normalization settings differ from the saved run.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        start, best = saved['epoch'] + 1, saved['best_miou']
    loss_options = {name: getattr(args, name) for name in [
        'ce_weight', 'dice_weight', 'focal_weight', 'orth_weight',
        'focal_alpha', 'focal_gamma', 'empty_class_weight']}
    for epoch in range(start, args.epochs):
        model.train()
        total = 0.0
        for images, target in train_loader:
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast(device_type='cuda', enabled=True) if amp else nullcontext()
            with context:
                logits, auxiliary, orth = model(images, target=target, return_aux=True)
                loss, _ = segmentation_loss(logits, auxiliary, orth, target, **loss_options)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss; check normalization and hyperparameters.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.autoprompt.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss.detach())
        model.eval()
        matrix = torch.zeros(args.num_classes + 1, args.num_classes + 1, dtype=torch.int64, device=device)
        with torch.no_grad():
            for images, target in val_loader:
                images, target = images.to(device), target.to(device)
                prediction = model(images).argmax(1)
                matrix += confusion_matrix(prediction, target, args.num_classes + 1)
        scores = summarize(matrix.cpu())
        improved = scores['miou'] is not None and scores['miou'] > best
        if improved:
            best = scores['miou']
        # Store architecture + normalization, not machine-specific input paths.
        saved = {'model': model.state_dict(), 'model_options': options,
                 'sam_config_text': Path(args.sam_config).read_text(encoding='utf-8'),
                 'mean': args.mean, 'std': args.std, 'band_order': args.band_order,
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'scaler': scaler.state_dict(), 'epoch': epoch, 'best_miou': best}
        torch.save(saved, output / 'last.pt')
        if improved:
            torch.save(saved, output / 'best.pt')
        record = {'epoch': epoch, 'train_loss': total / steps, **scores}
        print(json.dumps(record, allow_nan=False))
        with (output / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
