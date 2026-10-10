import torch
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss


def segmentation_loss(logits, auxiliary, orth, target, ce_weight=1.0, dice_weight=1.0,
                      focal_weight=0.5, orth_weight=0.1, focal_alpha=0.25,
                      focal_gamma=2.0, empty_class_weight=0.5):
    """Preserve the current CE + present-class Dice + auxiliary focal + orthogonal loss."""
    if target.ndim == 4:
        target = target.squeeze(1)
    ce = F.cross_entropy(logits, target.long())
    probs = logits.softmax(dim=1)
    dice, focal = logits.new_tensor(0.0), logits.new_tensor(0.0)
    active = 0
    for cls, aux in enumerate(auxiliary, start=1):
        binary = (target == cls).float().unsqueeze(1)
        small = F.interpolate(binary, aux.shape[-2:], mode='nearest')
        present = bool(binary.any())
        weight = 1.0 if present else empty_class_weight
        focal = focal + weight * sigmoid_focal_loss(aux, small, alpha=focal_alpha,
                                                   gamma=focal_gamma, reduction='mean')
        if present:
            pred = probs[:, cls:cls + 1]
            intersection = (pred * binary).sum(dim=(2, 3))
            denominator = pred.sum(dim=(2, 3)) + binary.sum(dim=(2, 3))
            dice = dice + (1 - (2 * intersection + 1.0) / (denominator + 1.0)).mean()
            active += 1
    if active:
        dice = dice / active
    total = ce_weight * ce + dice_weight * dice + focal_weight * focal + orth_weight * orth
    return total, {'ce': ce.detach(), 'dice': dice.detach(), 'focal': focal.detach(), 'orth': orth.detach()}
