# SDD-PIQA: Unsupervised Face Image Quality Assessment with Similarity Distribution Distance

## Get a 

## Generation of Quality Pseudo-Labels
1. Run './generate_pseudo_labels/gen_datalist.py' to obtain the data list.
2. Run './generate_pseudo_labels/extract_embedding/extract_feats.py' to extract the embeddings of face image.
3. Run './generate_pseudo_labels/gen_pseudo_labels.py' to calculate the quality pseudo-labels.


## Training of Quality Regression Model
1. Replace the data path with your local path on `train_confing.py`.
2. Run local_train.sh.

## Prediction of PIQA
Run './eval.py' to predict face quality score.
We provide the pre-trained model on the refined MS1M dataset with IR50: [googledrive](https://drive.google.com/file/d/1AM0iWVfSVWRjCriwZZ3FXiUGbcDzkF25/view?usp=sharing)
