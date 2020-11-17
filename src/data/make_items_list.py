import argparse

from PIL import Image

import settings
import os
import json

from src.features import get_box_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--coco_file',
                        help='path to coco_file',
                        type=str,
                        required=True,
                        defaul=os.path.join(settings.DATA_DIR, 'processed', 'deepfashion2_coco_train.json'))
    parser.add_argument('-i',
                        '--img_dir',
                        help='path to image dir',
                        type=str,
                        required=True,
                        defaul=os.path.join(settings.DATA_DIR, 'raw', 'train', 'image'))

    args = parser.parse_args()
    coco_file = args.coco_file
    img_dir = args.img_dir

    items_list = make_items_list(coco_file)
    for item in items_list:
        img = get_item(img_dir, item)
        get_box_features.get_box_features(img)


def make_items_list(file: str) -> list:
    with open(file) as json_file:
        dataset = json.load(json_file)
        items = []
        for i in dataset.get('annotations'):
            bbox = i.get('bbox')
            item_id = i.get('id')
            pair_id = i.get('pair_id')
            image_id = i.get('image_id')
            category_id = i.get('category_id')
            style = i.get('style')
            items.append((item_id, bbox, pair_id, image_id, category_id, style))
        return items


def get_item(img_dir, items):
    item = items[1]
    print('image-id: ' + items[3])
    print('style ' + str(items[5]))
    x_1 = item[0]
    y_1 = item[1]
    w = item[2]
    h = item[3]
    right = x_1 + w
    lower = y_1 + h
    img = Image.open(os.path.join(img_dir, format(item[2], '06') + '.jpg'))
    img = img.crop((x_1, y_1, right, lower))
    return img


if __name__ == '__main__':
    main()
