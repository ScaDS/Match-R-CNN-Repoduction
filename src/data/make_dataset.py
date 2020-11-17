import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# copied from: https://github.com/tkshnkmr/frcnn_medium_sample/blob/master/utils.py
class MakeDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        img = Image.open(os.path.join(self.root, img_id + '.jpg'))
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Category ID
        for i in range(num_objs):
            a = torch.zeros(13, dtype=torch.int64)
            a[coco_annotation[i]["category_id"] - 1] = 1
        cat_id = a

        # ID
        ident = []
        for i in range(num_objs):
            ident.append(coco_annotation[i]["id"])
        ident = torch.as_tensor(ident, dtype=torch.float32)

        # Pair_ID
        pair_id = []
        for i in range(num_objs):
            pair_id.append(coco_annotation[i]["pair_id"])
        pair_id = torch.as_tensor(pair_id, dtype=torch.float32)

        # Tensorise img_id
        img_id = torch.tensor([int(img_id)])

        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Style
        style = []
        for i in range(num_objs):
            style.append(coco_annotation[i]["style"])
        style = torch.as_tensor(style, dtype=torch.float32)

        # Number of Keypoints
        num_keypoints = []
        for i in range(num_objs):
            num_keypoints.append(coco_annotation[i]["num_keypoints"])
        num_keypoints = torch.as_tensor(num_keypoints, dtype=torch.float32)

        keypoints = []
        for i in range(num_objs):
            keypoints.append(coco_annotation[i]["keypoints"])
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)

        # Annotation is in dictionary format
        my_annotation = {"area": areas,
                         "boxes": boxes,
                         "labels": cat_id,
                         "id": ident,
                         "pair_id": pair_id,
                         "image_id": img_id,
                         "iscrowd": iscrowd,
                         "style": style,
                         "num_keypoints": num_keypoints,
                         "keypoints": keypoints}

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


# In my case, just added ToTensor
def get_transform():
    custom_transforms = [torchvision.transforms.ToTensor()]
    return torchvision.transforms.Compose(custom_transforms)


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# path to your own data and coco file
train_data_dir = "../data/raw/test/image/"
train_coco = "../data/processed/deepfashion2_coco_test.json"

# Batch size
train_batch_size = 1

# Params for dataloader
train_shuffle_dl = True
num_workers_dl = 4

# Params for training

# Two classes; Only target class or background
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
for imgs, annotations in data_loader:
    imgs = list(img.to(device) for img in imgs)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    print(annotations)

model = get_model_instance_segmentation(num_classes)

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
    for imgs, annotations in data_loader:
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f"Iteration: {i}/{len_dataloader}, Loss: {losses}")
