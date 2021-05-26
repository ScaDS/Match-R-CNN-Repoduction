import os
from datetime import datetime

import torch
import torch.utils.data
import torchvision
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

from src.data import make_dataset
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


# device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
# print(f"Training on device {device}.")
# model = MatchingNet().to(device=device)
# learning_rate = 0.02
# n_epochs = 5000
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# loss_fn = CrossEntropyLoss()
#
# train_data_dir = "../data/raw/test/image/"
# train_coco = "../data/processed/deepfashion2_coco_test.json"
#
# train_shuffle_dl = True
# num_workers_dl = 4
#
# train_batch_size = 8
#
# dataset = make_dataset.MakeDataset(root=train_data_dir,
#                                    annotation=train_coco,
#                                    transforms=make_dataset.get_transform())
#
#
# data_loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=train_batch_size,
#     shuffle=train_shuffle_dl,
#     num_workers=num_workers_dl,
#     collate_fn=make_dataset.collate_fn,
# )
#
#
# # TODO:
# #   - write train_loader
# #   - write val_loader
# #   - write main
#
#
# # partly copied from the book Deep Learning with Pytorch
# def training_loop(num_epochs, opt, mod, loss_function, train_loader):
#     for epoch in range(1, num_epochs + 1):
#         loss_train = 0.0
#         for features, labels in train_loader:
#             features = features.to(device=device)
#             labels = labels.to(device=device)
#             outputs = mod(features)
#             loss = loss_function(outputs, labels)
#
#             opt.zero_grade()
#             loss.backward()
#             opt.step()
#
#             loss_train += loss.item()
#
#         if epoch == 1 or epoch % 10 == 0:
#             print('{} Epoch {}, Training loss {}'.format(
#                 datetime.now(), epoch,
#                 loss_train / len(train_loader)))
#
#
# def validate(mod, validation_loader, val_loader):
#     accdict = {}
#     for name, loader in [("train", validation_loader), ("val", val_loader)]:
#         correct = 0
#         total = 0
#
#         with torch.no_grad():
#             for features, labels in loader:
#                 features = features.to(device=device)
#                 labels = labels.to(device=device)
#                 outputs = mod(features)
#                 _, predicted = torch.max(outputs, dim=1)
#                 total += labels.shape[0]
#                 correct += int((predicted == labels).sum())
#
#         print("Accuracy {}: {:.2f}".format(name, correct / total))
#         accdict[name] = correct / total
#     return accdict
#
#
# training_loop(
#     num_epochs=n_epochs,
#     opt=optimizer,
#     mod=model,
#     loss_function=loss_fn,
#     train_loader=data_loader
# )

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
