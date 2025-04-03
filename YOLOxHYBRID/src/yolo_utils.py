# utils.py
import torch
import cv2
from torchvision import transforms
from torchvision.ops import nms

def get_transform(image_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

def preprocess_image(image_path, image_size, device):
    transform = get_transform(image_size)
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)
    return img_tensor, img_bgr

def xywh_to_xyxy(x, y, w, h, img_w, img_h):
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return x1, y1, x2, y2

def compute_iou(box1, boxes):
    x1 = torch.max(box1[0], boxes[:, 0])
    y1 = torch.max(box1[1], boxes[:, 1])
    x2 = torch.min(box1[2], boxes[:, 2])
    y2 = torch.min(box1[3], boxes[:, 3])
    inter_area = torch.clamp(x2 - x1, 0) * torch.clamp(y2 - y1, 0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box1_area + boxes_area - inter_area
    return inter_area / (union + 1e-6)

def non_max_suppression(boxes, confs, iou_threshold):
    idxs = torch.argsort(confs, descending=True)
    keep = []
    while idxs.numel() > 0:
        i = idxs[0].item()
        keep.append(i)
        if idxs.numel() == 1:
            break
        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious <= iou_threshold]
    return keep

def decode_yolo_output(output, anchors, stride, conf_thresh=0.5, iou_thresh=0.5):
    B, C, H, W = output.shape
    num_anchors = len(anchors)
    num_classes = C // num_anchors - 5

    output = output.view(B, num_anchors, 5 + num_classes, H, W)
    output = output.permute(0, 1, 3, 4, 2).contiguous()

    anchors = torch.tensor(anchors).to(output.device).float().view(num_anchors, 1, 1, 2)
    boxes = []

    for b in range(B):
        preds = output[b]
        conf = preds[..., 4].sigmoid()

        mask = conf > conf_thresh
        if mask.sum() == 0:
            boxes.append([])
            continue

        x = preds[..., 0].sigmoid()
        y = preds[..., 1].sigmoid()
        w = preds[..., 2].exp()
        h = preds[..., 3].exp()

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

        box_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).view(-1, 4)
        conf = conf.view(-1)
        scores = conf

        keep = conf > conf_thresh
        box_xyxy = box_xyxy[keep]
        scores = scores[keep]

        if box_xyxy.shape[0] > 0:
            nms_idx = nms(box_xyxy, scores, iou_thresh)
            boxes.append(box_xyxy[nms_idx].cpu())
        else:
            boxes.append([])

    return boxes

def draw_boxes(boxes, original_img, iou_thresh=0.5):
    H, W, _ = original_img.shape
    boxes_xyxy = []
    confs = []
    for box in boxes:
        x_c, y_c, w, h, conf = box
        x1, y1, x2, y2 = xywh_to_xyxy(x_c, y_c, w, h, W, H)
        boxes_xyxy.append([x1, y1, x2, y2])
        confs.append(conf)

    boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
    confs_tensor = torch.tensor(confs)
    keep = non_max_suppression(boxes_tensor, confs_tensor, iou_thresh)

    for i in keep:
        x1, y1, x2, y2 = boxes_xyxy[i]
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original_img, f"{confs[i]:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return original_img
