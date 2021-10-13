import torch
import torch.utils.data


class MakeDataset(torch.utils.data.Dataset):
    """ Dataset to get pairwise trained features"""

    def __init__(self, pairs_feature_list, transform=None):
        """
        Args:
            pairs_feature_list (list): List with all features.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.pairs_feature_list = pairs_feature_list
        self.transform = transform

    def __len__(self):
        return len(self.pairs_feature_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        feature_1 = self.pairs_feature_list[idx][0][1]
        feature_2 = self.pairs_feature_list[idx][1][1]
        pair = self.pairs_feature_list[idx][2]

        target = torch.tensor([0])
        if pair:
            target = torch.tensor([1])
        my_annotation = target

        return feature_1, feature_2, my_annotation
