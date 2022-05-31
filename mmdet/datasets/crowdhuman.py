import json
import os
import os.path as osp

import numpy as np
from PIL import Image
from scipy.io import loadmat
from sklearn.metrics import average_precision_score

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


class CrowdHumanLoader:

    def __init__(self, dataset_root, split):
        self.cls_num = None
        self.dataset_root = dataset_root
        self.split = split
        if self.split == "test":
            self.split = "val"
        assert self.split in ["train", "val", "mix"]
        self.images_root = osp.join(self.dataset_root, "Images")
        self.box = "vbox"
        self.roidb = self.load_roidb()

    @property
    def num_images(self):
        return len(self.roidb)

    def image_path_at(self, i):
        image_path = self.roidb[i]["img_path"]
        assert osp.isfile(image_path), "Path does not exist: %s" % image_path
        return image_path

    def load_roidb(self):
        """Load the image indexes for training / test."""
        # import ipdb; ipdb.set_trace()
        anno = []
        if self.split == "train" or self.split == "mix":
            with open(os.path.join(self.dataset_root, "annotation_train.odgt")) as f:
                anno_train = [json.loads(line.strip()) for line in f.readlines()]
                anno.extend(anno_train)
        if self.split == "val" or self.split == "mix":
            with open(os.path.join(self.dataset_root, "annotation_val.odgt")) as f:
                anno_test = [json.loads(line.strip()) for line in f.readlines()]
                anno.extend(anno_test)
        roidb = []
        for item in anno:
            img_path = os.path.join(self.images_root, item["ID"] + ".jpg")
            size = Image.open(img_path).size
            gt_ids = []
            gt_boxes = []
            for b in item["gtboxes"]:
                if b["tag"] != "person":
                    continue
                x, y, w, h = b[self.box]
                if h < 50:
                    continue
                gt_boxes.append([x, y, x + w, y + h])
                gt_ids.append(-1)
            if len(gt_boxes) > 0:
                gt_boxes = np.asarray(gt_boxes, dtype=np.int32)
                gt_ids = np.asarray(gt_ids, dtype=np.int32)
                roidb.append({
                    "gt_boxes": gt_boxes,
                    "gt_pids": gt_ids,
                    "img_path": img_path,
                    "height": size[1],
                    "width": size[0]
                })
        return roidb


@DATASETS.register_module()
class CrowdHuman(CustomDataset):
    """
    Conventions for label:
    suppose we have K person with identity label, the cls num for the dataset set is K+1 (K + unlabeled)
    the cls id for the K person with identity label is [0, K)
    the cls id for the person without identity label is K
    the cls id for the background is K+1
    """
    CLASSES = None

    def __init__(self, **kwargs):
        dataset_root = kwargs.pop("dataset_root")
        self.split = kwargs.pop("split")
        self.loader = CrowdHumanLoader(dataset_root, self.split)
        super(CrowdHuman, self).__init__(**kwargs)
        print("Loaded CrowdHuman dataset (split=%s) from %s" % (self.split, dataset_root))

    def load_annotations(self, ann_file):
        data_infos = []
        for i, roidb in enumerate(self.loader.roidb):
            roidb = self.loader.roidb[i]
            new_entry = {
                "id": os.path.basename(roidb["img_path"]),
                "filename": roidb["img_path"],
                "width": roidb["width"],
                "height": roidb["height"],
            }
            if self.split == "train" or self.split == "mix":
                labels = roidb["gt_pids"].astype(np.int64)
                num_unlbl = np.sum(labels == -1)
                labels[labels == -1] = np.arange(-1, -1 - num_unlbl, -1, dtype=np.int64)
                new_entry["ann"] = {
                    "bboxes": roidb["gt_boxes"].astype(np.float32),
                    "labels": labels,
                }
            data_infos.append(new_entry)
        return data_infos

    def load_proposals(self, proposal_file):
        proposals = []
        for _ in self.loader.roidb:
            proposals.append(np.array([[0, 0, 0, 0]], dtype=np.float32))
        return proposals

    def evaluate(self, results, metric='mAP', logger=None, proposal_nums=(100, 300, 1000), iou_thr=0.5,
                 scale_ranges=None, gallery_size=100):
        print("\n")
        import torch
        torch.save(results, "inf_result_ch.pth")
        gallery_det = []
        for each in results:
            gallery_det.append(each[0])

        self.evaluate_detection(gallery_det)

    @staticmethod
    def compute_iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter * 1.0 / union

    def evaluate_detection(self, gallery_det, threshold=0.5, iou_thresh=0.5, labeled_only=False):
        """
        gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
        threshold (float): filter out gallery detections whose scores below this
        iou_thresh (float): treat as true positive if IoU is above this threshold
        labeled_only (bool): filter out unlabeled background people
        """
        assert self.loader.num_images == len(gallery_det)

        roidb = self.loader.roidb
        y_true, y_score = [], []
        count_gt, count_tp = 0, 0
        for gt, det in zip(roidb, gallery_det):
            gt_boxes = gt["gt_boxes"]
            if labeled_only:
                inds = np.where(gt["gt_pids"].ravel() > 0)[0]
                if len(inds) == 0:
                    continue
                gt_boxes = gt_boxes[inds]
            det = np.asarray(det)
            inds = np.where(det[:, 4].ravel() >= threshold)[0]
            det = det[inds]
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
            if num_det == 0:
                count_gt += num_gt
                continue
            ious = np.zeros((num_gt, num_det), dtype=np.float32)
            for i in range(num_gt):
                for j in range(num_det):
                    ious[i, j] = self.compute_iou(gt_boxes[i], det[j, :4])
            tfmat = ious >= iou_thresh
            # for each det, keep only the largest iou of all the gt
            for j in range(num_det):
                largest_ind = np.argmax(ious[:, j])
                for i in range(num_gt):
                    if i != largest_ind:
                        tfmat[i, j] = False
            # for each gt, keep only the largest iou of all the det
            for i in range(num_gt):
                largest_ind = np.argmax(ious[i, :])
                for j in range(num_det):
                    if j != largest_ind:
                        tfmat[i, j] = False
            for j in range(num_det):
                y_score.append(det[j, -1])
                y_true.append(tfmat[:, j].any())
            count_tp += tfmat.sum()
            count_gt += num_gt

        det_rate = count_tp * 1.0 / count_gt
        ap = average_precision_score(y_true, y_score) * det_rate

        print("{} detection:".format("Labeled only" if labeled_only else "All"))
        print("  Recall = {:.2%}".format(det_rate))
        if not labeled_only:
            print("  AP = {:.2%}".format(ap))


if __name__ == '__main__':
    dataset = CrowdHuman(split="mix", dataset_root="/home/chenzhicheng/GitRepo/mmdetection/data/ch", ann_file=None, pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ])
    import ipdb; ipdb.set_trace()
    import torch

    a = torch.load("/home/chenzhicheng/GitRepo/mmdetection/inf_result.pth")
    dataset.evaluate(a, gallery_size=100)
