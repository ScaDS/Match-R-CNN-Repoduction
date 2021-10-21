import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Tuple, List
from itertools import chain
from tqdm import tqdm


def shop_user_lists(path):

    shop_images_list = []
    user_images_list = []

    print('make shop- and user-list:')
    for file in tqdm(os.listdir(path)):
        with open(os.path.join(path, file), mode='r') as json_file:
            anno_file = json.load(json_file)
        if anno_file.get('source') == 'shop':
            shop_images_list.append(Path(file).stem)
        if anno_file.get('source') == 'user':
            user_images_list.append(Path(file).stem)

    return shop_images_list, user_images_list


def create_category_dict(coco_file: dict) -> dict:

    category_list = []
    for i in tqdm(coco_file.get('annotations')):
        category_list.append((i.get('image_id'), i.get('category_id')))

    category_dict = {}
    for i in tqdm(category_list):
        category_dict.setdefault(i[0], []).append(i[1])

    return category_dict


def create_all_positive_pairs(coco_file, shop, user):

    positive_pairs = []
    user_set = set(user)
    shop_set = set(shop)

    print('creating all positive pairs:')
    for i in tqdm(coco_file.get('annotations')):
        if i.get('image_id') in user_set:
            if i.get('style') != 0:
                for j in coco_file.get('annotations'):
                    if j.get('image_id') in shop_set:
                        if i.get('style') == j.get('style'):
                            if i.get('pair_id') == j.get('pair_id'):
                                if i.get('category_id') == j.get('category_id'):
                                    positive_pairs.append((i.get('image_id'), j.get('image_id'), i.get('category_id')))

    return list(set(positive_pairs))


def create_negative_pairs(coco_file, shop, user, positive_pairs):

    negative_pairs = []
    user_set = set(user)
    shop_set = set(shop)

    print('creating all negative pairs:')
    for i in tqdm(coco_file.get('annotations')):
        if len(negative_pairs) >= len(positive_pairs):
            break
        else:
            if i.get('image_id') in user_set:
                for j in coco_file.get('annotations'):
                    if j.get('image_id') in shop_set:
                        if i.get('style') != j.get('style'):
                            if i.get('pair_id') == j.get('pair_id'):
                                if i.get('category_id') == j.get('category_id'):
                                    negative_pairs.append((i.get('image_id'), j.get('image_id'), None))

    return list(set(negative_pairs))


def load_pairs(pairs_pkl: str) -> List[Tuple[str, str]]:
    with open(pairs_pkl, 'rb') as f:
        pairs = pickle.load(f)
    return pairs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--set',
                        help='choose between training and validation',
                        choices=['train', 'validation', 'test'],
                        nargs='?',
                        default='validation')
    args = parser.parse_args()

    file = os.path.join('data', 'processed', 'deepfashion2_coco_' + args.set + '.json')

    with open(file, mode='r') as json_file:
        coco_file = json.load(json_file)

    shop_pkl = Path(os.path.join('data', 'processed', args.set + '_shop_list.pkl'))
    user_pkl = Path(os.path.join('data', 'processed', args.set + '_user_list.pkl'))
    if shop_pkl.is_file() and user_pkl.is_file():
        with open(shop_pkl, 'rb') as f:
            shop = pickle.load(f)
        with open(user_pkl, 'rb') as f:
            user = pickle.load(f)
    else:
        shop, user = shop_user_lists(os.path.join('data', 'raw', args.set, 'annos'))
        with open(shop_pkl, 'wb') as f:
            pickle.dump(shop, f)
        with open(user_pkl, 'wb') as f:
            pickle.dump(user, f)

    positive_pkl = Path(os.path.join('data', 'processed', args.set + '_positive_pairs.pkl'))
    if positive_pkl.is_file():
        with open(positive_pkl, 'rb') as f:
            positive_pairs = pickle.load(f)
    else:
        positive_pairs = create_all_positive_pairs(coco_file, shop, user)
        with open(positive_pkl, 'wb') as f:
            pickle.dump(positive_pairs, f)

    negative_pairs = create_negative_pairs(coco_file, shop, user, positive_pairs)
    with open(os.path.join('data', 'processed', args.set + '_negative_pairs.pkl'), 'wb') as f:
        pickle.dump(negative_pairs, f)

    training_pairs = list(chain(*zip(positive_pairs, negative_pairs)))
    with open(os.path.join('data', 'processed', args.set + '_pairs.pkl'), 'wb') as f:
        pickle.dump(training_pairs, f)

    print('length pos pairs: ', len(positive_pairs))
    print(positive_pairs[:10])
    print('length neg pairs: ', len(negative_pairs))
    print(negative_pairs[:10])
    print('length pairs: ', len(training_pairs))
    print(training_pairs[:10])


if __name__ == '__main__':
    main()
