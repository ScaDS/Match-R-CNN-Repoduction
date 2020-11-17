from datetime import datetime

import torch
from torch import optim
from torch.nn import CrossEntropyLoss

from src.models.matching_network import MatchingNet


# TODO:
#  Training loop
#  scheduler:
#  LR start at 0.02 -> decreased by 0.01
#                   -> first time after 8 epochs
#                   -> then 11 epochs
#                   -> terminates at 12 epochs


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")
model = MatchingNet().to(device=device)
learning_rate = 0.02
n_epochs = 5000
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = CrossEntropyLoss()


# partly copied from the book Deep Learning with Pytorch
def training_loop(n_epochs: int, optimizer, model: torch.tensor, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for rois, labels in train_loader:
            rois = rois.to(device=device)
            labels = labels.to(device=device)
            outputs = model(rois)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grade()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.now(), epoch,
                loss_train / len(train_loader)))


def validate(model: torch.tensor, train_loader, val_loader):
    accdict = {}
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for rois, labels in loader:
                rois = rois.to(device=device)
                labels = labels.to(device=device)
                outputs = model(rois)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name, correct / total))
        accdict[name] = correct / total
    return accdict
