import os
from datetime import datetime

import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.nn import CrossEntropyLoss

from src.data.make_dataset import MakeDataset
from src.features.get_features import get_features
from src.models.matching_network import MatchingNet


# TODO:
#  Training loop
#  scheduler:
#  LR start at 0.02 -> decreased by 0.01
#                   -> first time after 8 epochs
#                   -> then 11 epochs
#                   -> terminates at 12 epochs

def get_transform():
    custom_transforms = [torchvision.transforms.Resize(800),
                         torchvision.transforms.ToTensor()]
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

train_data_dir = os.path.join('data', 'raw', 'train', 'image')
train_coco = os.path.join('data', 'processed', 'deepfashion2_coco_train.json')
validation_data_dir = os.path.join('data', 'raw', 'validation', 'image')
validation_coco = os.path.join('data', 'processed', 'deepfashion2_coco_validation.json')


train_batch_size = 6
train_shuffle_dl = True
num_workers_dl = 4
num_epochs = 12
lr = 0.02
momentum = 0.9
weight_decay = 0.005

print("Torch version:", torch.__version__)

my_dataset = MakeDataset(
    root=train_data_dir,
    annotation=train_coco,
    transforms=get_transform())

my_validationset = MakeDataset(
    root=validation_data_dir,
    annotation=validation_coco,
    transforms=get_transform(),
    path=os.path.join('data', 'processed', 'validation_pairs.pkl'))

train_loader = torch.utils.data.DataLoader(
    my_dataset,
    batch_size=train_batch_size,
    shuffle=train_shuffle_dl,
    num_workers=num_workers_dl,
    collate_fn=collate_fn,)

validation_loader = torch.utils.data.DataLoader(
    my_validationset,
    batch_size=train_batch_size,
    shuffle=train_shuffle_dl,
    num_workers=num_workers_dl,
    collate_fn=collate_fn,)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = MatchingNet().to(device=device)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
# params = model.parameters()
optimizer = torch.optim.SGD(
    params,
    lr=lr,
    momentum=momentum,
    weight_decay=weight_decay)

len_trainloader = len(train_loader)


def make_target(anno1, anno2):
    pair1 = anno1[0]['pair_id']
    pair2 = anno2[0]['pair_id']
    style1 = anno1[0]['style']
    style2 = anno2[0]['style']
    target = torch.tensor([0])
    for p1, s1 in zip(pair1, style1):
        for p2, s2 in zip(pair2, style2):
            if p1 == p2 and s1 == s2:
                if s1 != torch.tensor([0]).to(device):
                    target = torch.tensor([1])
    return target.repeat(600)


def training_loop(num_epochs, opt, mod, loss_function, train_loader):
    min_loss_val = np.inf
    for epoch in range(1, num_epochs + 1):
        loss_train = 0.0
        mod.train()
        for imgs1, imgs2, annotations1, annotations2 in train_loader:
            # imgs1 = list(img1.to(device) for img1 in imgs1)
            # imgs2 = list(img2.to(device) for img2 in imgs2)
            imgs1 = get_features(imgs1).to(device)
            imgs2 = get_features(imgs2).to(device)
            annotations1 = [{k: v.to(device) for k, v in t.items()} for t in annotations1]
            annotations2 = [{k: v.to(device) for k, v in t.items()} for t in annotations2]
            outputs = mod(imgs1, imgs2)
            target = make_target(annotations1, annotations2).to(device)
            loss = loss_function(outputs, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_train += loss.item()

        loss_val = 0.0
        mod.eval()
        with torch.no_grad():
            for imgs1, imgs2, annotations1, annotations2 in validation_loader:
                imgs1 = get_features(imgs1).to(device)
                imgs2 = get_features(imgs2).to(device)
                annotations1 = [{k: v.to(device) for k, v in t.items()} for t in annotations1]
                annotations2 = [{k: v.to(device) for k, v in t.items()} for t in annotations2]
                outputs = mod(imgs1, imgs2)
                target = make_target(annotations1, annotations2).to(device)
                loss = loss_function(outputs, target)
                loss_val += loss.item()

        print(f'Epoch {epoch + 1} \t\t '
              f'Training Loss: {loss_train / len(train_loader)} \t\t '
              f'Validation Loss: {loss_val / len(validation_loader)}')
        if min_loss_val > loss_val:
            print(f'Validation Loss Decreased({min_loss_val:.6f}--->{loss_val:.6f}) \t Saving The Model')
            min_loss_val = loss_val
            torch.save(model.state_dict(), os.path.join('data', 'results', 'final_model.pth'))

        torch.save(model.state_dict(), os.path.join('data', 'results', str(epoch + 1) + '_trained_model.pth'))


training_loop(num_epochs,
              optimizer,
              model,
              CrossEntropyLoss(),
              train_loader)

