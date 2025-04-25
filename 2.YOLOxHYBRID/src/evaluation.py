import torch

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    Both boxes are expected to be in [x1, y1, x2, y2] format.
    """
    box1 = box1.tolist()
    box2 = box2.tolist()

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou


def calculate_map_score(predictions, targets, iou_threshold=0.5):
    """
    Approximate mAP using average of precision and recall (F1-like metric).
    iou threshold kept 0.5 for avoiding noise and efficiency
    """
    tp, fp, fn = 0, 0, 0

    for pred_boxes, gt_boxes in zip(predictions, targets):
        matched = []
        for pred in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for i, gt in enumerate(gt_boxes):
                iou = compute_iou(pred[:4], gt[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_threshold and best_gt_idx not in matched:
                tp += 1
                matched.append(best_gt_idx)
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    map_score = (precision * recall) / (precision + recall + 1e-6)  

    return map_score  



if __name__ == "__main__":
    preds = [[torch.tensor([10., 10., 50., 50.])]]
    gts = [[torch.tensor([12., 12., 48., 48.])]]
    print("Approx mAP:", calculate_map_score(preds, gts))
