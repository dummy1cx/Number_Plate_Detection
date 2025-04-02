import torch
import torch.nn.functional as F
from ciou_loss import compute_ciou_loss
from focal_loss import focal_loss

def compute_yolo_loss(pred_outputs, targets, model):
    device = pred_outputs[0].device
    batch_size = len(targets)

    anchors = {k: v.to(device) for k, v in model.anchors.items()}
    anchor_masks = model.anchor_masks
    num_classes = model.num_classes
    img_size = model.image_size

    obj_target = {}
    cls_target = {}
    scale_mapping = ['small', 'medium', 'large']
    grid_sizes = {}

    for i, scale in enumerate(scale_mapping):
        _, _, H, W = pred_outputs[i].shape
        obj_target[scale] = torch.zeros(batch_size, len(anchor_masks[scale]), H, W, device=device)
        cls_target[scale] = torch.zeros(batch_size, len(anchor_masks[scale]), H, W, num_classes, device=device)
        grid_sizes[scale] = (H, W)

    ignore_threshold = 0.5
    box_pairs = []

    for b, truth in enumerate(targets):
        if truth is None or truth.numel() == 0:
            continue
        truth = truth.to(device)
        gt_classes = truth[:, 0].long()
        gt_x = truth[:, 1] * img_size
        gt_y = truth[:, 2] * img_size
        gt_w = truth[:, 3] * img_size
        gt_h = truth[:, 4] * img_size

        for t in range(truth.shape[0]):
            gx, gy, gw, gh = gt_x[t].item(), gt_y[t].item(), gt_w[t].item(), gt_h[t].item()
            cls = gt_classes[t].item()
            all_anchors = torch.cat([anchors[k] for k in anchors], dim=0)
            anchor_w = all_anchors[:, 0]
            anchor_h = all_anchors[:, 1]
            inter_w = torch.min(anchor_w, torch.tensor(gw, device=device))
            inter_h = torch.min(anchor_h, torch.tensor(gh, device=device))
            inter_area = inter_w * inter_h
            union_area = anchor_w * anchor_h + gw * gh - inter_area
            ious = inter_area / union_area
            best_anchor = int(torch.argmax(ious))

            if best_anchor in anchor_masks['small']:
                scale = 'small'
            elif best_anchor in anchor_masks['medium']:
                scale = 'medium'
            else:
                scale = 'large'

            scale_anchor_indices = anchor_masks[scale]
            anchor_idx_on_scale = scale_anchor_indices.index(best_anchor)
            grid_h, grid_w = grid_sizes[scale]
            stride = img_size / grid_w
            gi = min(int(gx / stride), grid_w - 1)
            gj = min(int(gy / stride), grid_h - 1)

            obj_target[scale][b, anchor_idx_on_scale, gj, gi] = 1.0
            cls_target[scale][b, anchor_idx_on_scale, gj, gi, cls] = 1.0

            for anc_idx in range(all_anchors.size(0)):
                if anc_idx == best_anchor:
                    continue
                if ious[anc_idx] > ignore_threshold:
                    if anc_idx in anchor_masks['small']:
                        alt_scale = 'small'
                    elif anc_idx in anchor_masks['medium']:
                        alt_scale = 'medium'
                    else:
                        alt_scale = 'large'
                    grid_h_alt, grid_w_alt = grid_sizes[alt_scale]
                    stride_alt = img_size / grid_w_alt
                    gi_alt = min(int(gx / stride_alt), grid_w_alt - 1)
                    gj_alt = min(int(gy / stride_alt), grid_h_alt - 1)
                    alt_anchor_idx_on_scale = anchor_masks[alt_scale].index(anc_idx)
                    if obj_target[alt_scale][b, alt_anchor_idx_on_scale, gj_alt, gi_alt] == 0:
                        obj_target[alt_scale][b, alt_anchor_idx_on_scale, gj_alt, gi_alt] = -1.0

            box_pairs.append([
                b, scale, anchor_idx_on_scale, gj, gi,
                [gx / img_size, gy / img_size, gw / img_size, gh / img_size]
            ])

    total_obj_loss = 0.0
    total_cls_loss = 0.0
    total_box_loss = 0.0

    for i, scale in enumerate(scale_mapping):
        pred = pred_outputs[i]
        B, _, H, W = pred.shape
        anchor_count = len(anchor_masks[scale])
        pred = pred.view(B, anchor_count, 5 + num_classes, H, W)
        pred_box = pred[:, :, 0:4, :, :]
        pred_obj = pred[:, :, 4, :, :]
        pred_cls = pred[:, :, 5:, :, :]

        target_obj = obj_target[scale]
        mask = target_obj != -1
        pred_obj_masked = pred_obj[mask]
        target_obj_masked = target_obj[mask]
        if pred_obj_masked.numel() > 0:
            total_obj_loss += F.binary_cross_entropy_with_logits(pred_obj_masked, target_obj_masked, reduction='sum')

        pos_mask = (target_obj == 1)
        if pos_mask.any():
            pred_cls_pos = pred_cls.permute(0,1,3,4,2)[pos_mask]
            tgt_cls_pos = cls_target[scale][pos_mask]
            total_cls_loss += focal_loss(pred_cls_pos, tgt_cls_pos, alpha=0.25, gamma=2.0)

    for b, scale, anchor_idx, gj, gi, gt_box in box_pairs:
        pred = pred_outputs[scale_mapping.index(scale)]
        pred = pred.view(batch_size, len(anchor_masks[scale]), 5 + num_classes, *pred.shape[2:])
        tx, ty, tw, th = pred[b, anchor_idx, 0:4, gj, gi]
        stride = img_size / pred.shape[-1]
        anchor_tensor = anchors[scale][anchor_idx]

        pred_cx = (gi + torch.sigmoid(tx)) / pred.shape[-1]
        pred_cy = (gj + torch.sigmoid(ty)) / pred.shape[-2]
        pred_w = torch.exp(tw) * anchor_tensor[0] / img_size
        pred_h = torch.exp(th) * anchor_tensor[1] / img_size

        pred_box = [pred_cx, pred_cy, pred_w, pred_h]
        total_box_loss += compute_ciou_loss(pred_box, gt_box)

    total_loss = total_obj_loss + total_cls_loss + total_box_loss
    return total_loss / batch_size, total_obj_loss / batch_size, total_cls_loss / batch_size, total_box_loss / batch_size
