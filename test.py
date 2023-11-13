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

def main(args):
    seed_everything(0)
    print(args)

    data = DataModule(args)
    dataloader = data.val_dataloader()

    # model: resnet
    model = ModelModule(args).to(args.device)


    test_acc_list = []
    accuracy = MulticlassAccuracy(num_classes=args.num_classes).to(args.device)

    for i in range(20):
        checkpoint_path = args.ckpt_path
        print(f"Loading checkpoint from {args.ckpt_path}: epoch={i * 10 + 9}.ckpt")
        ckpt_path = os.path.join(args.ckpt_path, f"epoch={i * 10 + 9}.ckpt")
        model = ModelModule.load_from_checkpoint(ckpt_path).to(args.device)

        for batch_idx, (X,y) in enumerate(dataloader):
            batch = (X.to(args.device), y.to(args.device))

            with torch.no_grad():
                #_, acc = model(batch,"val")
                x, y = batch
                features = model.encoder(x)
                #features = model.resnet(x)
                features = F.normalize(features)

                weights = F.normalize(model.linear.weight)

                logits = (features @ weights.t()) * args.temperature
                acc = accuracy(logits, y)
        print("Acc:", acc.item() * 100)

        test_acc_list.append(acc.item())
    print("Test acc list:", test_acc_list)


if __name__ == '__main__':
    args = get_eval_arguments()
    main(args)