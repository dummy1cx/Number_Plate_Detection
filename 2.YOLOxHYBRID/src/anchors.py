import math
import torch
import torch.nn.functional as F
import os
import numpy as np
from sklearn.cluster import KMeans
## ---------------------------------------------------------------------------------------------
## The code is referenced from
## https://medium.com/%40yerdaulet.zhumabay/generating-anchor
## https://github.com/decanbay/YOLOv3-Calculate-Anchor-Boxes/blob/master/YOLOv3_get_anchors.py
## This code helps implimenting anchor boxes refering to the labels file for th co-ordinates
## ----------------------------------------------------------------------------------------------

def load_yolo_labels(label_folder):
    boxes = []
    for file in os.listdir(label_folder):
        if file.endswith('.txt'):
            with open(os.path.join(label_folder, file), 'r') as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.strip().split())
                    boxes.append([w, h])
    return np.array(boxes)

def get_kmeans_anchors(boxes, n_clusters=9):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300)
    kmeans.fit(boxes)
    anchors = kmeans.cluster_centers_
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
    return anchors