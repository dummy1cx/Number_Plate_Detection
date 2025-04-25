import torch
from torchvision.ops import nms

def decode_yolo_output(output, anchors, stride, conf_thresh=0.5, iou_thresh=0.05):
   ##--------------------------------
   ## Decide yolo output for single scale
   ## with iou _threshold = 0.5
   ##--------------------------------
    B, C, H, W = output.shape
    num_anchors = len(anchors)
    num_classes = C // num_anchors - 5

    output = output.view(B, num_anchors, 5 + num_classes, H, W)
    output = output.permute(0, 1, 3, 4, 2).contiguous()  # [B, A, H, W, 5 + C]

    anchors = torch.tensor(anchors).to(output.device).float().view(num_anchors, 1, 1, 2)
    boxes = []

    for b in range(B):
        preds = output[b]  # [A, H, W, 5 + C]
        conf = preds[..., 4].sigmoid()
        cls_scores = preds[..., 5:].sigmoid()

        # Filter boxes by objectness threshold
        mask = conf > conf_thresh
        if mask.sum() == 0:
            boxes.append([])
            continue

        # Coordinates
        x = preds[..., 0].sigmoid()
        y = preds[..., 1].sigmoid()
        w = preds[..., 2].exp()
        h = preds[..., 3].exp()

        # Grid
        grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
        grid_x = grid_x.to(output.device)
        grid_y = grid_y.to(output.device)

        x = (x + grid_x.unsqueeze(0)) * stride
        y = (y + grid_y.unsqueeze(0)) * stride
        w = w * anchors[..., 0]
        h = h * anchors[..., 1]

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        box_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)  # [A, H, W, 4]

        # Flatten all predictions
        box_xyxy = box_xyxy.view(-1, 4)
        conf = conf.view(-1)
        scores = conf  # (optionally multiply with class prob if using multi-class)

        keep = conf > conf_thresh
        box_xyxy = box_xyxy[keep]
        scores = scores[keep]

        # Apply NMS
        if box_xyxy.shape[0] > 0:
            nms_idx = nms(box_xyxy, scores, iou_thresh)
            boxes.append(box_xyxy[nms_idx].cpu())
        else:
            boxes.append([])

    return boxes

