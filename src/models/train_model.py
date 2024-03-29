from datetime import datetime
import os

import numpy as np
import torch
import torch.utils.data
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch.multiprocessing

from src.data.make_dataset import MakeDataset

from src.models.matching_network import MatchingNet

# from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def get_transform():
    custom_transforms = [torchvision.transforms.Resize(800),
                         torchvision.transforms.ToTensor()]
    return torchvision.transforms.Compose(custom_transforms)


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


train_pairs_path = os.path.join('data', 'results', 'final_training_item_pairs', 'train_item_pairs.pkl')
train_features_path = os.path.join('data', 'results', 'pooled_features', 'train')
validation_pairs_path = os.path.join('data', 'results', 'final_training_item_pairs', 'validation_item_pairs.pkl')
validation_features_path = os.path.join('data', 'results', 'pooled_features', 'validation')

train_batch_size = 128
train_shuffle_dl = True
# num_workers_dl = 4
num_workers_dl = 0
num_epochs = 12
lr = 0.02
momentum = 0.9
weight_decay = 0.00001

print("Torch version:", torch.__version__)

my_dataset = MakeDataset(
    item_pairs_list=train_pairs_path,
    features_dir=train_features_path
)

my_validationset = MakeDataset(
    item_pairs_list=validation_pairs_path,
    features_dir=validation_features_path
)

train_loader = torch.utils.data.DataLoader(
    my_dataset,
    batch_size=train_batch_size,
    shuffle=train_shuffle_dl,
    num_workers=num_workers_dl,
    collate_fn=collate_fn,
    drop_last=True, )

validation_loader = torch.utils.data.DataLoader(
    my_validationset,
    batch_size=train_batch_size,
    shuffle=train_shuffle_dl,
    num_workers=num_workers_dl,
    collate_fn=collate_fn,
    drop_last=True, )

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = MatchingNet()
model = nn.DataParallel(model)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
# params = model.parameters()
optimizer = torch.optim.SGD(
    params,
    lr=lr,
    momentum=momentum,
    weight_decay=weight_decay)

len_trainloader = len(train_loader)


def training_loop(n_epochs, opt, mod, loss_function, trainloader, validationloader):
    min_loss_val = np.inf

    train_loss_values = []
    validation_loss_values = []
    ap_list = []

    y_true = np.array([])
    y_scores = np.array([])

    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        loss_val = 0.0

        mod.train()
        print('train epoch: ', epoch)
        for feature1, feature2, annotation in tqdm(trainloader):
            feat1 = torch.cat([torch.unsqueeze(f1, 0) for f1 in feature1], 0).to(device)
            feat2 = torch.cat([torch.unsqueeze(f2, 0) for f2 in feature2], 0).to(device)
            anno = torch.cat([torch.unsqueeze(a, 0) for a in annotation]).to(device)
            outputs = mod(feat1, feat2)

            loss = loss_function(outputs, anno.squeeze(1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_train += loss.item() * ((feat1.size(0) + feat2.size(0)) / 2)


        mod.eval()
        with torch.no_grad():
            print('validation epoch: ', epoch)
            for feature1, feature2, annotation in tqdm(validationloader):
                feat1 = torch.cat([torch.unsqueeze(f1, 0) for f1 in feature1], 0).to(device)
                feat2 = torch.cat([torch.unsqueeze(f2, 0) for f2 in feature2], 0).to(device)
                anno = torch.cat([torch.unsqueeze(a, 0) for a in annotation]).to(device)
                outputs = mod(feat1, feat2)

                loss = loss_function(outputs, anno.squeeze(1))
                loss_val += loss.item() * ((feat1.size(0) + feat2.size(0)) / 2)

                y_true = np.append(y_true, anno.to('cpu'))
                y_scores = np.append(y_scores, outputs.to('cpu'))

        print(f'Epoch {epoch + 1} \t\t '
              f'Training Loss: {loss_train / len(train_loader)} \t\t '
              f'Validation Loss: {loss_val / len(validation_loader)}')
        if min_loss_val > loss_val:
            print(f'Validation Loss Decreased({min_loss_val:.6f}--->{loss_val:.6f}) \t Saving The Model')
            min_loss_val = loss_val
            torch.save(model.state_dict(), os.path.join('data', 'results', 'model', 'final_model.pth'))
        torch.save(model.state_dict(), os.path.join('data', 'results', 'model', str(epoch + 1) + '_trained_model.pth'))

        train_loss_values.append(loss_train / len(trainloader))
        validation_loss_values.append(loss_val / len(validation_loader))

        am_y_scores = np.array([])

        for i, k in zip(y_scores[0::2], y_scores[1::2]):
            t = torch.tensor([i, k])
            am_y_scores = np.append(am_y_scores, torch.argmax(t, dim=0))

        ap = average_precision_score(y_true, am_y_scores)
        ap_list.append(ap)
        print('average precision: ' + str(ap))

        if epoch == 9 or epoch == 11:
            opt.param_groups[0]['lr'] = opt.param_groups[0]['lr'] * 0.1


    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

    plt.plot(train_loss_values, 'r', label='train_loss')
    plt.plot(validation_loss_values, 'b', label='validation_loss')
    # plt.plot(ap_list, 'g', label='average_precision')
    plt.legend(loc="upper right")
    plt.xlabel("Epochen")
    plt.ylabel("Fehler")
    plt.savefig(dt_string + '_dropout_0_2'  + '.png')


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    training_loop(num_epochs,
                  optimizer,
                  model,
                  CrossEntropyLoss(),
                  train_loader,
                  validation_loader)


if __name__ == '__main__':
    main()
