from datetime import datetime

import torch
import torch.utils.data
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

from src.data import make_dataset
from src.models.matching_network import MatchingNet


# TODO:
#  Training loop
#  scheduler:
#  LR start at 0.02 -> decreased by 0.01
#                   -> first time after 8 epochs
#                   -> then 11 epochs
#                   -> terminates at 12 epochs

class PairDataset(Dataset):
    def __init__(self, dataset_1, dataset_2):
        self.dataset1 = dataset_1
        self.dataset2 = dataset_2

    def __getitem__(self, index):
        x1 = self.dataset1[index]
        x2 = self.dataset2[index]

        return x1, x2

    def __len__(self):
        return len(self.dataset1)


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")
model = MatchingNet().to(device=device)
learning_rate = 0.02
n_epochs = 5000
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = CrossEntropyLoss()

train_data_dir = "../data/raw/test/image/"
train_coco = "../data/processed/deepfashion2_coco_test.json"

train_shuffle_dl = True
num_workers_dl = 4

train_batch_size = 8

dataset = make_dataset.MakeDataset(root=train_data_dir,
                                   annotation=train_coco,
                                   transforms=make_dataset.get_transform())


data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=train_batch_size,
    shuffle=train_shuffle_dl,
    num_workers=num_workers_dl,
    collate_fn=make_dataset.collate_fn,
)


# TODO:
#   - write train_loader
#   - write val_loader
#   - write main


# partly copied from the book Deep Learning with Pytorch
def training_loop(num_epochs, opt, mod, loss_function, train_loader):
    for epoch in range(1, num_epochs + 1):
        loss_train = 0.0
        for features, labels in train_loader:
            features = features.to(device=device)
            labels = labels.to(device=device)
            outputs = mod(features)
            loss = loss_function(outputs, labels)

            opt.zero_grade()
            loss.backward()
            opt.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.now(), epoch,
                loss_train / len(train_loader)))


def validate(mod, validation_loader, val_loader):
    accdict = {}
    for name, loader in [("train", validation_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(device=device)
                labels = labels.to(device=device)
                outputs = mod(features)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name, correct / total))
        accdict[name] = correct / total
    return accdict


training_loop(
    num_epochs=n_epochs,
    opt=optimizer,
    mod=model,
    loss_function=loss_fn,
    train_loader=data_loader
)
