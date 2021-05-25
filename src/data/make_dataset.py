import os

import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO


from src.data import make_pairs
from src.features.get_features import get_features
from src.models.matching_network import MatchingNet


class MakeDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        # self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = make_pairs.pairs(10)

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
            a = torch.zeros(13, dtype=torch.int64)
            a[coco_annotation2[i]["category_id"] - 1] = 1
        cat_id2 = a

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
            style1.append(coco_annotation2[i]["style"])
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

        if self.transforms is not None:
            img2 = self.transforms(img2)

        return img1, img2, my_annotation1, my_annotation2

    def __len__(self):
        return len(self.ids)


# In my case, just added ToTensor
def get_transform():
    custom_transforms = [torchvision.transforms.ToTensor()]
    return torchvision.transforms.Compose(custom_transforms)


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


# def get_model_instance_segmentation(num_classes):
#     # load an instance segmentation model pre-trained pre-trained on COCO
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
#     # get number of input features for the classifier
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     # replace the pre-trained head with a new one
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#
#     return model


# path to your own data and coco file
train_data_dir = os.path.join('data', 'raw', 'train', 'image')
train_coco = os.path.join('data', 'processed', 'deepfashion2_coco_train.json')

# Batch size
train_batch_size = 1

# Params for dataloader
train_shuffle_dl = True
num_workers_dl = 4

# Params for training

num_classes = 13
num_epochs = 10

lr = 0.005
momentum = 0.9
weight_decay = 0.005

print("Torch version:", torch.__version__)

# create own Dataset
my_dataset = MakeDataset(
    root=train_data_dir, annotation=train_coco, transforms=get_transform()
)

# own DataLoader
data_loader = torch.utils.data.DataLoader(
    my_dataset,
    batch_size=train_batch_size,
    shuffle=train_shuffle_dl,
    num_workers=num_workers_dl,
    collate_fn=collate_fn,
)

# select device (whether GPU or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# DataLoader is iterable over Dataset
for imgs1, imgs2, annotations1, annotations2 in data_loader:
    imgs1 = list(img1.to(device) for img1 in imgs1)
    imgs2 = list(img1.to(device) for img1 in imgs1)
    annotations1 = [{k: v.to(device) for k, v in t.items()} for t in annotations1]
    print(annotations1)
    annotations2 = [{k: v.to(device) for k, v in t.items()} for t in annotations2]
    print(annotations2)

# model = get_model_instance_segmentation(num_classes)
model = MatchingNet().to(device=device)

# move model to the right device
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, lr=lr, momentum=momentum, weight_decay=weight_decay
)

len_dataloader = len(data_loader)

# Training
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}/{num_epochs}")
    model.train()
    i = 0
    for imgs1, imgs2, annotations1, annotations2 in data_loader:
        i += 1
        imgs1 = list(img1.to(device) for img1 in imgs1)
        imgs2 = list(img2.to(device) for img2 in imgs2)
        annotations1 = [{k: v.to(device) for k, v in t.items()} for t in annotations1]
        annotations2 = [{k: v.to(device) for k, v in t.items()} for t in annotations2]
        loss_dict = model(get_features(imgs1), get_features(imgs2), annotations1, annotations2)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f"Iteration: {i}/{len_dataloader}, Loss: {losses}")
