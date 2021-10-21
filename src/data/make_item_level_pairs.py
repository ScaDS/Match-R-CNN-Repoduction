import argparse
import json
import os
import pickle

from tqdm import tqdm


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


def add_style(feat, style_cat_list):
    feat_plus_style = ()

    old_len = len(feat_plus_style)
    for sc in style_cat_list:
        if feat == sc[1]:
            feat_plus_style += (feat, sc[0])
    if old_len == len(feat_plus_style):
        feat_plus_style += (feat, 0)

    return feat_plus_style


def make_item_pairs(feature_dict_path, annos_dir, target, pairs_file_path):
    with open(pairs_file_path, 'rb') as f:
        pairs_file = pickle.load(f)
        f.close()

    with open(feature_dict_path, 'rb') as f:
        feature_dict = pickle.load(f)
        f.close()

    item_pairs = []

    for pair in tqdm(pairs_file):
        a_features = feature_dict[pair[0]]
        b_features = feature_dict[pair[1]]
        pair_cat = pair[2]

        a_style_cat_list = get_style_cat(os.path.join(annos_dir, pair[0] + '.json'))
        b_style_cat_list = get_style_cat(os.path.join(annos_dir, pair[1] + '.json'))

        if check_cat_duplicates([t[1] for t in a_features]) and check_cat_duplicates([t[1] for t in b_features]):
            for a_feat in a_features:
                for b_feat in b_features:
                    if a_feat[1] == b_feat[1]:
                        a_feature_style = add_style(int(a_feat[1]), a_style_cat_list)
                        b_feature_style = add_style(int(b_feat[1]), b_style_cat_list)

                        if pair_cat is None:
                            if a_feature_style[1] == b_feature_style[1]:
                                pair_bool = False
                                item_pairs.append((a_feat[0], b_feat[0], pair_bool))
                        else:
                            pair_cat -= 1  # die category_ids der annos gehen von 1-13 bei den preds von 0-12
                            if pair_cat == a_feat[1]:
                                if a_feature_style[1] == b_feature_style[1]:
                                    pair_bool = True
                                    item_pairs.append((a_feat[0], b_feat[0], pair_bool))

    with open(target, 'wb') as f:
        pickle.dump(item_pairs, f)
        f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--pairs',
                        help='path to pairs pickle file',
                        type=str,
                        default=os.path.join('data', 'processed', 'train_pairs.pkl'))
    parser.add_argument('-d',
                        '--feat_dict',
                        help='path to features directory',
                        type=str,
                        default=os.path.join('data', 'results', 'train_feature_index_class_dict.pkl'))
    parser.add_argument('-j',
                        '--json_dir',
                        help='path to json directory',
                        type=str,
                        default=os.path.join('data', 'raw', 'train', 'annos'))
    parser.add_argument('-t',
                        '--target',
                        help='path to target file',
                        type=str,
                        default=os.path.join('data',
                                             'results', 'final_training_item_pairs', 'train_item_pairs.pkl'))
    args = parser.parse_args()

    pairs_file_path = args.pairs
    feature_dir_path = args.feat_dict
    json_dir_path = args.json_dir
    target = args.target

    make_item_pairs(feature_dir_path, json_dir_path, target, pairs_file_path)


if __name__ == '__main__':
    main()
