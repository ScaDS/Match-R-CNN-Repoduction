import json
import os
import random
import shutil

import numpy as np
from tqdm import tqdm

import settings


def get_pair_ids(annotations_dir):
    pair_ids = []
    for filename in tqdm(os.listdir(annotations_dir)):
        with open(os.path.join(annotations_dir, filename)) as json_file:
            data = json.load(json_file)
        pair_ids.append(data['pair_id'])

    return np.unique(pair_ids)


def get_image(pair_ids, annotations_dir):
    images = []
    for pair_id in tqdm(pair_ids):
        for filename in os.listdir(annotations_dir):
            with open(os.path.join(annotations_dir, filename)) as json_file:
                data = json.load(json_file)
            if data['pair_id'] == pair_id:
                images.append(filename)

    return images


def split_dataset(data_dir, factor, seed):

    test_dir = os.path.join(os.path.dirname(data_dir), 'test')
    image_dir, annotations_dir = os.path.join(test_dir, 'image'), os.path.join(test_dir, 'annos')
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True), os.makedirs(annotations_dir, exist_ok=True)

    if len(os.listdir(os.path.join(test_dir, "image"))) != len(os.listdir(os.path.join(data_dir, "image"))):
        pair_ids = get_pair_ids(os.path.join(settings.DATA_DIR,
                                             'raw',
                                             'train',
                                             'annos'))
        random.seed(seed)
        random.shuffle(pair_ids)
        selected = get_image(pair_ids[:int(len(pair_ids) * factor)],
                             os.path.join(settings.DATA_DIR,
                                          'raw',
                                          'train',
                                          'annos'))
        selected_imgs = [os.path.join(data_dir, 'image', i) for i in selected]
        selected_annos = [os.path.join(data_dir, 'annos', i.split(".")[0] + '.json') for i in selected]

        for img, anno in zip(selected_imgs, selected_annos):
            shutil.move(img, os.path.join(image_dir, os.path.split(img)[-1]))
            shutil.move(anno, os.path.join(annotations_dir, os.path.split(anno)[-1]))
            print(f"Moved {len(selected)} images and annos from {data_dir} to {test_dir}.")


split_dataset(os.path.join(settings.DATA_DIR, 'raw', 'train'), 0.3, 7)
