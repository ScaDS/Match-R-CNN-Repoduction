# Match-R-CNN-Repoduction
Reproducing the fashion image categorization and retrieval baseline approach from https://github.com/switchablenorms/DeepFashion2 / https://arxiv.org/pdf/1901.07973.pdf

## Installation
* conda install --file requirements.txt

## Procedure

###Data-preparation
* download deepfashion2 dataset
* run src/data/split_dataset.py on train dataset
* run src/data/deepfashion2_to_coco.py with train, validation and test dataset
* run src/data/create_all_pairs.py with train, validation and test dataset
  
### Train feature-network
* run src/models/train_feature_network.py with train dataset

### Feature and pair creation
* run src/features/get_features.py with train, validation and test dataset
* run src/data/create_all_pairs.py with train, validation and test dataset
* run src/make_item_level_pairs.py with train, validation and test dataset

### Train matching-network
* run src/models/train_model.py
