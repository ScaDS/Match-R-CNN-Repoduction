import argparse
import os
import re

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--path',
                        help='filepath',
                        type=str,
                        default=os.path.join('data', 'results', 'box_features_train.txt'))
    args = parser.parse_args()
    file_path = args.path

    f = open(file_path, "r")
    lines_list = f.readlines()

    features_list = []

    for line in lines_list:
        ident = line[0:6]
        feature = re.findall('\[.+?\]', line)[0][1:-1]
        feature = feature.split(',')
        feature = [float(i) for i in feature]
        feature_tensor = torch.FloatTensor(feature)
        features_list.append((ident, feature_tensor))

    return features_list


if __name__ == '__main__':
    main()
