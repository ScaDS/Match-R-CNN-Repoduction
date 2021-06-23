import argparse
import json
import os
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--pairs',
                        help='number of pairs',
                        type=int,
                        default=6)
    # parser.add_argument('-p',
    #                     '--path',
    #                     help='path to coco_file',
    #                     type=str,
    #                     default=os.path.join(settings.DATA_DIR, 'processed', 'deepfashion2_coco_train.json'))
    args = parser.parse_args()

    # coco_file = args.path
    coco_file = os.path.join('data', 'processed', 'deepfashion2_coco_train.json')
    n_pairs = args.pairs

    df = make_image_df(coco_file)
    print(df)
    print(create_pairs(df, n_pairs))

    return create_pairs(df, n_pairs)


# def pairs(n):
#     coco_file = os.path.join('data', 'processed', 'deepfashion2_coco_train.json')
#     df = make_image_df(coco_file)
#     return create_pairs(df, n)


def make_image_df(file):
    with open(file, mode='r') as json_file:
        coco_file = json.load(json_file)

    image_dict = defaultdict(list)

    print('creating image-dataframe:')
    for i in tqdm(coco_file.get('annotations')):
        image_dict['image_id'].append(i.get('image_id'))
        image_dict['pair_id'].append(i.get('pair_id'))
        image_dict['style'].append(i.get('style'))
        image_dict['category_id'].append(i.get('category_id'))

    return pd.DataFrame.from_dict(image_dict)


def create_pairs(df, number_of_pairs):
    pairs = []
    i = 0
    while i < number_of_pairs:
        positive_example1 = df.sample(n=1)
        while positive_example1['style'].item() == 0:
            positive_example1 = df.sample(n=1)

        pair_df = df[df['pair_id'] == positive_example1['pair_id'].item()]
        style_fitting_df = pair_df[pair_df['style'] == positive_example1['style'].item()]
        positive_example2 = style_fitting_df.sample(n=1)

        category_fitting_df = df[df['category_id'] == positive_example1['category_id'].item()]
        negative_example1 = category_fitting_df.sample(n=1)
        negative_example2 = category_fitting_df[
            category_fitting_df['image_id'] != positive_example1['image_id'].item()].sample(n=1)

        # appends tuple with (pos1, pos2)
        pairs.append((positive_example1['image_id'].values[0],
                      positive_example2['image_id'].values[0],))
        # appends tuple with (neg1, neg2)
        pairs.append((negative_example1['image_id'].values[0],
                      negative_example2['image_id'].values[0]))
        i += 1

    return pairs


# def create_all_positiv_pairs_df(df):
#     positiv_pairs = []
#     print('creating all positive pairs:')
#     for index1, row1 in tqdm(df.iterrows()):
#         for index2, row2 in df.iterrows():
#             if row1['image_id'] != row2['image_id']:
#                 if row1['style'] != 0:
#                     if row1['pair_id'] == row2['pair_id']:
#                         positiv_pairs.append((index1, index2))
#
#     return positiv_pairs


def create_all_negativ_pairs_df(df):
    negativ_pairs = []
    for index, row in df.iterrows():
        print(row)
    return negativ_pairs


if __name__ == '__main__':
    main()
