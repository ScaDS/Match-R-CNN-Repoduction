import os

import torch
import torch.utils.data
from PIL import Image
from pycocotools.coco import COCO


from src.data import create_all_pairs


class MakeDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None, path=os.path.join('data', 'processed', 'train_pairs.pkl')):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        # self.ids = list(sorted(self.coco.imgs.keys()))
        # self.ids = make_pairs.pairs(1000)
        self.ids = create_all_pairs.load_pairs(path)

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id1 = self.ids[index][0]
        img_id2 = self.ids[index][1]
        # List: get annotation id from coco
        ann_ids1 = coco.getAnnIds(imgIds=[img_id1])
        ann_ids2 = coco.getAnnIds(imgIds=[img_id2])
        # Dictionary: target coco_annotation file for an image
        coco_annotation1 = coco.loadAnns(ann_ids1)
        coco_annotation2 = coco.loadAnns(ann_ids2)
        img1 = Image.open(os.path.join(self.root, img_id1 + '.jpg'))
        img2 = Image.open(os.path.join(self.root, img_id2 + '.jpg'))
        # number of objects in the image
        num_objs1 = len(coco_annotation1)
        num_objs2 = len(coco_annotation2)

        # Size of bbox (Rectangular)
        areas1 = []
        for i in range(num_objs1):
            areas1.append(coco_annotation1[i]["area"])
        areas1 = torch.as_tensor(areas1, dtype=torch.float32)
        areas2 = []
        for i in range(num_objs2):
            areas2.append(coco_annotation2[i]["area"])
        areas2 = torch.as_tensor(areas2, dtype=torch.float32)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes1 = []
        for i in range(num_objs1):
            xmin = coco_annotation1[i]["bbox"][0]
            ymin = coco_annotation1[i]["bbox"][1]
            xmax = xmin + coco_annotation1[i]["bbox"][2]
            ymax = ymin + coco_annotation1[i]["bbox"][3]
            boxes1.append([xmin, ymin, xmax, ymax])
        boxes1 = torch.as_tensor(boxes1, dtype=torch.float32)
        boxes2 = []
        for i in range(num_objs2):
            xmin = coco_annotation2[i]["bbox"][0]
            ymin = coco_annotation2[i]["bbox"][1]
            xmax = xmin + coco_annotation2[i]["bbox"][2]
            ymax = ymin + coco_annotation2[i]["bbox"][3]
            boxes2.append([xmin, ymin, xmax, ymax])
        boxes2 = torch.as_tensor(boxes2, dtype=torch.float32)

        # Category ID
        for i in range(num_objs1):
            a = torch.zeros(13, dtype=torch.int64)
            a[coco_annotation1[i]["category_id"] - 1] = 1
        cat_id1 = a
        for i in range(num_objs2):
            b = torch.zeros(13, dtype=torch.int64)
            b[coco_annotation2[i]["category_id"] - 1] = 1
        cat_id2 = b

        # ID
        ident1 = []
        for i in range(num_objs1):
            ident1.append(coco_annotation1[i]["id"])
        ident1 = torch.as_tensor(ident1, dtype=torch.float32)
        ident2 = []
        for i in range(num_objs2):
            ident2.append(coco_annotation2[i]["id"])
        ident2 = torch.as_tensor(ident2, dtype=torch.float32)

        # Pair_ID
        pair_id1 = []
        for i in range(num_objs1):
            pair_id1.append(coco_annotation1[i]["pair_id"])
        pair_id1 = torch.as_tensor(pair_id1, dtype=torch.float32)
        pair_id2 = []
        for i in range(num_objs2):
            pair_id2.append(coco_annotation2[i]["pair_id"])
        pair_id2 = torch.as_tensor(pair_id2, dtype=torch.float32)

        # Tensorise img_id
        img_id1 = torch.tensor([int(img_id1)])
        img_id2 = torch.tensor([int(img_id2)])

        # Iscrowd
        iscrowd1 = torch.zeros((num_objs1,), dtype=torch.int64)
        iscrowd2 = torch.zeros((num_objs2,), dtype=torch.int64)

        # Style
        style1 = []
        for i in range(num_objs1):
            style1.append(coco_annotation1[i]["style"])
        style1 = torch.as_tensor(style1, dtype=torch.float32)
        style2 = []
        for i in range(num_objs2):
            style2.append(coco_annotation2[i]["style"])
        style2 = torch.as_tensor(style2, dtype=torch.float32)

        # Number of Keypoints
        num_keypoints1 = []
        for i in range(num_objs1):
            num_keypoints1.append(coco_annotation1[i]["num_keypoints"])
        num_keypoints1 = torch.as_tensor(num_keypoints1, dtype=torch.float32)
        num_keypoints2 = []
        for i in range(num_objs2):
            num_keypoints2.append(coco_annotation2[i]["num_keypoints"])
        num_keypoints2 = torch.as_tensor(num_keypoints2, dtype=torch.float32)

        keypoints1 = []
        for i in range(num_objs1):
            keypoints1.append(coco_annotation1[i]["keypoints"])
        keypoints1 = torch.as_tensor(keypoints1, dtype=torch.float32)
        keypoints2 = []
        for i in range(num_objs2):
            keypoints2.append(coco_annotation2[i]["keypoints"])
        keypoints2 = torch.as_tensor(keypoints2, dtype=torch.float32)

        # Annotation is in dictionary format
        my_annotation1 = {"area": areas1,
                          "boxes": boxes1,
                          "labels": cat_id1,
                          "id": ident1,
                          "pair_id": pair_id1,
                          "image_id": img_id1,
                          "iscrowd": iscrowd1,
                          "style": style1,
                          "num_keypoints": num_keypoints1,
                          "keypoints": keypoints1}
        my_annotation2 = {"area": areas2,
                          "boxes": boxes2,
                          "labels": cat_id2,
                          "id": ident2,
                          "pair_id": pair_id2,
                          "image_id": img_id2,
                          "iscrowd": iscrowd2,
                          "style": style2,
                          "num_keypoints": num_keypoints2,
                          "keypoints": keypoints2}

        if self.transforms is not None:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        return img1, img2, my_annotation1, my_annotation2

    def __len__(self):
        return len(self.ids)
