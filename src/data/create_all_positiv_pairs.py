import json
import os
import pickle

from tqdm import tqdm


def create_all_positiv_pairs_df(file):

    with open(file, mode='r') as json_file:
        coco_file = json.load(json_file)

    positiv_pairs = []

    print('creating all positive pairs:')
    for i in tqdm(coco_file.get('annotations')):
        for j in coco_file.get('annotations'):
            if i.get('image_id') != j.get('image_id'):
                if i.get('style') != 0:
                    if i.get('style') == j.get('style'):
                        if i.get('pair_id') == j.get('pair_id'):
                            positiv_pairs.append((i.get('image_id'), j.get('image_id')))
                            print(i.get('image_id'), j.get('image_id'))

    return positiv_pairs


def main():
    coco_file = os.path.join('data', 'processed', 'deepfashion2_coco_train.json')
    positv_pairs = create_all_positiv_pairs_df(coco_file)

    with open('positiv_pairs.pkl', 'wb') as f:
        pickle.dump(positv_pairs, f)


if __name__ == '__main__':
    main()
