import argparse
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm


def save_features(file_path):
    f = open(file_path, "r")
    lines_list = f.readlines()

    features_dict = defaultdict(list)

    for line in tqdm(lines_list):
        ident = line[0:6]
        feature = re.findall('\[.+?\]', line)[0][1:-1]
        feature = feature.split(',')
        feature = [float(i) for i in feature]
        feature_tensor = torch.FloatTensor(feature)

        features_dict[ident].append(feature_tensor)

    for key in tqdm(features_dict):
        target_file = Path(os.path.join('data', 'results', 'features', key + '.pkl'))
        with open(target_file, 'wb') as f:
            pickle.dump(features_dict[key], f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--filepath',
                        help='filepath',
                        type=str,
                        default=os.path.join('data', 'results', 'box_features_train.txt'))
    args = parser.parse_args()
    file_path = args.filepath

    save_features(file_path)


if __name__ == '__main__':
    main()
