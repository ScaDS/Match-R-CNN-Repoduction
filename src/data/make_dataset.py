import os
import pickle

import torch
import torch.utils.data


class MakeDataset(torch.utils.data.Dataset):
    """ Dataset to get pairwise trained features"""

    def __init__(self, item_pairs_list, features_dir, transform=None):
        with open(item_pairs_list, 'rb') as f:
            self.item_pairs_list = pickle.load(f)
            f.close
        self.features_dir = features_dir
        self.transform = transform

    def __len__(self):
        return len(self.item_pairs_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pair = self.item_pairs_list[idx]
        with open(os.path.join(self.features_dir, pair[0] + '.pkl'), 'rb') as f:
            feature_1 = pickle.load(f)
            f.close()
        with open(os.path.join(self.features_dir, pair[1] + '.pkl'), 'rb') as f:
            feature_2 = pickle.load(f)
            f.close()

        target = torch.tensor([1])
        if pair[2]:
            target = torch.tensor([0])
        my_annotation = target

        return feature_1[0], feature_2[0], my_annotation
