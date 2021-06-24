import json
import os
import pickle
from pathlib import Path
from typing import Tuple, List

from tqdm import tqdm


def shop_user_lists(path: str) -> Tuple[List[str], List[str]]:

    shop_images_list = []
    user_images_list = []

    print('make shop- and user-list')
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


def create_all_positive_pairs(coco_file: dict, shop: List[str], user: List[str]) -> List[Tuple[str, str]]:

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
                                positive_pairs.append((i.get('image_id'), j.get('image_id')))

    return positive_pairs


def create_negative_pairs(shop: List[str],
                          user: List[str],
                          coco_file: dict,
                          positive_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:

    negative_pairs = []
    user_set = set(user)
    shop_set = set(shop)

    print('creating negative pairs:')
    while len(negative_pairs) < len(positive_pairs):
        for i in tqdm(coco_file.get('annotations')):
            if i.get('image_id') in user_set:
                for j in coco_file.get('annotations'):
                    if j.get('image_id') in shop_set:
                        if i.get('category_id') == j.get('category_id'):
                            if i.get('pair_id') != j.get('pair_id'):
                                negative_pairs.append((i.get('image_id'), j.get('image_id')))

    return negative_pairs


def main():

    file = os.path.join('data', 'processed', 'deepfashion2_coco_train.json')
    with open(file, mode='r') as json_file:
        coco_file = json.load(json_file)

    shop, user = shop_user_lists(os.path.join('data', 'raw', 'train', 'annos'))

    # positve_pairs = create_all_positive_pairs(coco_file, shop, user)
    # with open('positive_pairs.pkl', 'wb') as f:
    #     pickle.dump(positve_pairs, f)

    with open('positiv_pairs.pkl', 'rb') as f:
        positive_pairs = pickle.load(f)
    negative_pairs = create_negative_pairs(shop, user, coco_file, positive_pairs)

    with open('negative_pairs', 'wb') as f:
        pickle.dump(negative_pairs, f)


if __name__ == '__main__':
    main()
