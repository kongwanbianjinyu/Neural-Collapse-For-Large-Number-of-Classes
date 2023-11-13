from argparse import ArgumentParser
from ModelModule import *
from DataModule import *
from pytorch_lightning import seed_everything
import pytorch_lightning as pl

import torch.nn as nn

import scipy.linalg as scilin
from args import *
import torch
import os
from torch.utils.data import Dataset
from sklearn.neighbors import NearestCentroid



def collect_features(args, model, dataloader):
    feature_list = []
    target_list = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            features = model.encoder(inputs)
            #features = model.resnet(inputs)

        features = nn.functional.normalize(features, dim=1, p=2)
        features = features.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()


        feature_list.append(features)
        target_list.append(targets)

    X = np.vstack(feature_list)
    y = np.hstack(target_list)

    return X, y


def main(args):
    seed_everything(0)
    print(args)

    data = DataModule(args)

    # model: resnet
    model = ModelModule(args).to(args.device)

    NCC_acc_train_list = []
    NCC_acc_test_list = []

    for i in range(20):
        checkpoint_path = args.ckpt_path
        print(f"Loading checkpoint from {args.ckpt_path}: epoch={i * 10 + 9}.ckpt")
        ckpt_path = os.path.join(args.ckpt_path, f"epoch={i * 10 + 9}.ckpt")
        model = ModelModule.load_from_checkpoint(ckpt_path).to(args.device)

        print("Collecting Layerwise Features...")
        X_train, y_train = collect_features(args, model, data.train_dataloader())
        X_test, y_test = collect_features(args, model, data.val_dataloader())
        # print(X_train.shape)
        # print(y_train.shape)

        # Creating the Nearest Centroid Classifier
        model = NearestCentroid()

        # Training the classifier
        model.fit(X_train, y_train)
        NCC_acc_train = model.score(X_train, y_train) * 100
        NCC_acc_test = model.score(X_test, y_test) * 100
        NCC_acc_train_list.append(NCC_acc_train)
        NCC_acc_test_list.append(NCC_acc_test)
        print(f"Training Set Score : {NCC_acc_train} %")
        print(f"Testing Set Score : {NCC_acc_test} %")

    print("NCC train acc:" , NCC_acc_train_list)
    print("NCC test acc:", NCC_acc_test_list)


if __name__ == '__main__':
    args = get_eval_arguments()
    main(args)