import json
import os
import random
import shutil

import numpy as np
from tqdm import tqdm

import settings


def make_image_dict(annotations_dir: str) -> dict:
    pair_ids = []
    for filename in tqdm(os.listdir(annotations_dir)):
        with open(os.path.join(annotations_dir, filename)) as json_file:
            data = json.load(json_file)
        pair_ids.append(data['pair_id'])
    pair_ids = np.unique(pair_ids)

    image_dict = {}
    for i in pair_ids:
        image_dict.update({i: []})

    for filename in tqdm(os.listdir(annotations_dir)):
        with open(os.path.join(annotations_dir, filename)) as json_file:
            data = json.load(json_file)
        if data['pair_id'] in image_dict:
            image_dict[data['pair_id']].append(filename)

    return image_dict


def split_dataset(data_dir: str, factor: float, seed: int, pairs_dict: dict):
    test_dir = os.path.join(os.path.dirname(data_dir), 'test')
    image_dir, annotations_dir = os.path.join(test_dir, 'image'), os.path.join(test_dir, 'annos')
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True), os.makedirs(annotations_dir, exist_ok=True)

    if len(os.listdir(os.path.join(test_dir, 'image'))) != len(os.listdir(os.path.join(data_dir, 'image'))):
        pair_ids = list(pairs_dict.keys())
        random.seed(seed)
        random.shuffle(pair_ids)

        selected = []
        for pair_id in pair_ids[:int(len(pair_ids) * factor)]:
            selected.extend(pairs_dict[pair_id])

        selected_annos = [os.path.join(data_dir, 'annos', i) for i in selected]
        selected_imgs = [os.path.join(data_dir, 'image', i.split(".")[0] + '.jpg') for i in selected]

        for img, anno in zip(selected_imgs, selected_annos):
            shutil.move(img, os.path.join(image_dir, os.path.split(img)[-1]))
            shutil.move(anno, os.path.join(annotations_dir, os.path.split(anno)[-1]))


p_dict = make_image_dict(os.path.join(settings.DATA_DIR, 'raw', 'train', 'annos'))
split_dataset(os.path.join(settings.DATA_DIR, 'raw', 'train'), 0.3, 7, p_dict)
