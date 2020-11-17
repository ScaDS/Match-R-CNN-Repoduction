import json
import os
import random
import shutil
from collections import defaultdict

from tqdm import tqdm

import settings
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-s',
                    '--seed',
                    help='seed for random split',
                    type=int,
                    default=7)
parser.add_argument('-p',
                    '--path',
                    help='path to the record to be split',
                    type=str,
                    default=os.path.join(settings.DATA_DIR, 'raw', 'train'))
parser.add_argument('-f',
                    '--factor',
                    help='factor for random split',
                    type=float,
                    default=0.3)
args = parser.parse_args()


def make_image_dict(annotations_dir: str) -> dict:
    image_dict = defaultdict(list)
    for filename in tqdm(os.listdir(annotations_dir)):
        with open(os.path.join(annotations_dir, filename)) as json_file:
            data = json.load(json_file)
        image_dict[data['pair_id']].append(filename)

    return image_dict


def split_dataset(data_dir: str, factor: float, seed: int, pairs_dict: dict):
    test_dir = os.path.join(os.path.dirname(data_dir), 'test')
    image_dir = os.path.join(test_dir, 'image')
    annotations_dir = os.path.join(test_dir, 'annos')
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True), os.makedirs(annotations_dir, exist_ok=True)

    pair_ids = list(pairs_dict.keys())
    random.seed(seed)
    random.shuffle(pair_ids)

    selected = []
    for pair_id in pair_ids[:int(len(pair_ids) * factor)]:
        selected.extend(pairs_dict[pair_id])

    selected_annotations = [os.path.join(data_dir, 'annos', i) for i in selected]
    selected_images = [os.path.join(data_dir, 'image', i.split(".")[0] + '.jpg') for i in selected]

    for img, annotation in zip(selected_images, selected_annotations):
        shutil.move(img, os.path.join(image_dir, os.path.split(img)[-1]))
        shutil.move(annotation, os.path.join(annotations_dir, os.path.split(annotation)[-1]))


p_dict = make_image_dict(os.path.join(args.path, 'annos'))
split_dataset(args.path, args.factor, args.seed, p_dict)
