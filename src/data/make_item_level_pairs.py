import argparse
import json
import os
import pickle
import re

import torch
from tqdm import tqdm


def parse_line_to_feature_tensor_list(line):
    feature_str_list = re.findall('\[.+?\]', line[1:-2])
    feature_tensor_list = []

    for feature in feature_str_list:
        feature = feature.split(',')
        feature = feature[1:-1]
        feat = [float(i) for i in feature]
        feature_tensor_list.append(torch.FloatTensor(feat))

    return feature_tensor_list


def parse_category_list(partner):
    file = partner[1:]
    category_list = [s.strip() for s in str(file).split('pred_classes: tensor([')[1].split(']')[0].split(',')]
    if category_list != ['']:
        category_list = [int(x) for x in category_list]

    return category_list


def get_style_cat(json_file_path):
    style_cat_list = []

    with open(json_file_path) as json_file:
        data = json.load(json_file)

        for k, v in data.items():
            if type(v) is dict:
                style_cat_list.append((v['style'], v['category_id'] - 1))

    return style_cat_list


def add_style_to_cat(cat_list, style_cat_list):
    style_added_list = []

    for cat in cat_list:
        old_len = len(style_added_list)
        for sc in style_cat_list:
            if cat == sc[1]:
                style_added_list.append((cat, sc[0]))
        if old_len == len(style_added_list):
            style_added_list.append((cat, 0))

    return style_added_list


def check_cat_duplicates(list_to_check):
    return len(list_to_check) == len(set(list_to_check))


def make_item_level_pairs(pairs_file_path, feature_dir_path, json_dir_path, target_path):
    pairs_pred_instances_features = []

    with open(pairs_file_path, 'rb') as f:
        pairs_file = pickle.load(f)

    count = 0
    for pair in tqdm(pairs_file):
        f = open(os.path.join(feature_dir_path, pair[0]), "r")
        a_partner = f.readlines()
        f = open(os.path.join(feature_dir_path, pair[1]), "r")
        b_partner = f.readlines()

        a_feature_list = parse_line_to_feature_tensor_list(a_partner[0])
        b_feature_list = parse_line_to_feature_tensor_list(b_partner[0])

        # parsed from pred_instances
        a_cat_list = parse_category_list(a_partner)
        b_cat_list = parse_category_list(b_partner)

        # from annotation json file
        a_style_cat_list = get_style_cat(os.path.join(json_dir_path, pair[0] + '.json'))
        b_style_cat_list = get_style_cat(os.path.join(json_dir_path, pair[1] + '.json'))

        if check_cat_duplicates(a_cat_list) and check_cat_duplicates(b_cat_list):
            a_annos_list = add_style_to_cat(a_cat_list, a_style_cat_list)
            b_annos_list = add_style_to_cat(b_cat_list, b_style_cat_list)

            a_item_list = []
            for annos, features in zip(a_annos_list, a_feature_list):
                a_item_list.append((annos, features))

            b_item_list = []
            for annos, features in zip(b_annos_list, b_feature_list):
                b_item_list.append((annos, features))

            for a_item in a_item_list:
                for b_item in b_item_list:
                    if a_item[0][0] == b_item[0][0] and a_item[0][1] == b_item[0][1]:
                        # item ((cat, style), feature)
                        pair_bool = count % 2 == 0
                        pairs_pred_instances_features.append((a_item, b_item, pair_bool))
        count += 1

    # dumps list with pairs [(((a_cat, a_style), a_feature), ((b_cat, b_style), b_feature), pair_bool), ...]
    with open(target_path, 'wb') as f:
        pickle.dump(pairs_pred_instances_features, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--pairs',
                        help='path to pairs pickle file',
                        type=str,
                        default=os.path.join('data', 'processed', 'train_pairs.pkl'))
    parser.add_argument('-d',
                        '--feat_dir',
                        help='path to features directory',
                        type=str,
                        default=os.path.join('data', 'results', 'features'))
    parser.add_argument('-j',
                        '--json_dir',
                        help='path to json directory',
                        type=str,
                        default=os.path.join('data', 'raw', 'train', 'annos'))
    parser.add_argument('-t',
                        '--target_path',
                        help='path to target',
                        type=str,
                        default=os.path.join('data', 'processed', 'item_level_pairs.pkl'))
    args = parser.parse_args()

    pairs_file_path = args.pairs
    feature_dir_path = args.feat_dir
    json_dir_path = args.json_dir
    target_path = args.target_path

    make_item_level_pairs(pairs_file_path, feature_dir_path, json_dir_path, target_path)


if __name__ == '__main__':
    main()
