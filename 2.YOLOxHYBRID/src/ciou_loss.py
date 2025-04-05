import torch
import math

## -----------------------------------------------------------------
## CIOU loss function was refernced from 
## https://github.com/miaoshuyu/object-detection-usages/blob/master/src/ciou_loss.py 
## -----------------------------------------------------------------

def compute_ciou_loss(pred_box, target_box):
    pred_x, pred_y, pred_w, pred_h = pred_box
    target_x, target_y, target_w, target_h = [torch.tensor(v, device=pred_x.device, dtype=torch.float32) for v in target_box]

    pred_x1 = pred_x - pred_w / 2
    pred_y1 = pred_y - pred_h / 2
    pred_x2 = pred_x + pred_w / 2
    pred_y2 = pred_y + pred_h / 2

    target_x1 = target_x - target_w / 2
    target_y1 = target_y - target_h / 2
    target_x2 = target_x + target_w / 2
    target_y2 = target_y + target_h / 2

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area + 1e-6

    iou = inter_area / union_area
    center_dist = (pred_x - target_x)**2 + (pred_y - target_y)**2
    enc_x1 = torch.min(pred_x1, target_x1)
    enc_y1 = torch.min(pred_y1, target_y1)
    enc_x2 = torch.max(pred_x2, target_x2)
    enc_y2 = torch.max(pred_y2, target_y2)
    enc_diag = (enc_x2 - enc_x1)**2 + (enc_y2 - enc_y1)**2 + 1e-6

    v = (4 / math.pi ** 2) * (torch.atan(target_w / (target_h + 1e-6)) - torch.atan(pred_w / (pred_h + 1e-6))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)

    ciou = iou - center_dist / enc_diag - alpha * v
    return 1 - ciou
