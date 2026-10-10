import torch


def confusion_matrix(prediction, target, classes):
    valid = (target >= 0) & (target < classes)
    encoded = classes * target[valid].long() + prediction[valid].long()
    return torch.bincount(encoded, minlength=classes ** 2).reshape(classes, classes)


def summarize(matrix):
    matrix = matrix.double()
    true_positive = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - true_positive
    iou = true_positive / union
    f1 = 2 * true_positive / (matrix.sum(0) + matrix.sum(1))
    # Binary facility metric merges all foreground classes; raft/cage confusion
    # counts as correct facility presence, while the class IoUs still penalize it.
    facility_tp = matrix[1:, 1:].sum()
    facility_fp, facility_fn = matrix[0, 1:].sum(), matrix[1:, 0].sum()
    def scalar(value):
        return float(value) if torch.isfinite(value) else None
    return {'class_iou': [scalar(v) for v in iou], 'class_f1': [scalar(v) for v in f1],
            'miou': scalar(iou.nanmean()),
            'facility_iou': scalar(facility_tp / (facility_tp + facility_fp + facility_fn)),
            'facility_f1': scalar(2 * facility_tp / (2 * facility_tp + facility_fp + facility_fn))}
