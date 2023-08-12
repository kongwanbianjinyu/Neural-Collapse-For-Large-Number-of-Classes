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
from tqdm import tqdm

CIFAR10_TRAIN_SAMPLES = 10 * (5000,)
CIFAR100_TRAIN_SAMPLES = 100 * (500,)
TINY = 200 * (500,)
FACE = 10000 * (50,)



def compute_info(args, model, dataloader):
    mu_G = 0
    mu_c_dict = dict()
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            #features = model.resnet(inputs)
            features = model.encoder(inputs)

        features = nn.functional.normalize(features, dim=1, p=2)
        features = features.detach()

        mu_G += torch.sum(features, dim=0)

        for b in range(len(targets)):
            y = targets[b].item()
            if y not in mu_c_dict:
                mu_c_dict[y] = features[b, :]
            else:
                mu_c_dict[y] += features[b, :]

    if args.dataset == "cifar10":
        mu_G /= sum(CIFAR10_TRAIN_SAMPLES)
        for i in range(len(CIFAR10_TRAIN_SAMPLES)):
            mu_c_dict[i] /= CIFAR10_TRAIN_SAMPLES[i]
    elif args.dataset == "cifar100":
        mu_G /= sum(CIFAR100_TRAIN_SAMPLES)
        for i in range(len(CIFAR100_TRAIN_SAMPLES)):
            mu_c_dict[i] /= CIFAR100_TRAIN_SAMPLES[i]
    elif args.dataset == "tiny_imagenet":
        mu_G /= sum(TINY)
        for i in range(len(TINY)):
            mu_c_dict[i] /= TINY[i]
    elif args.dataset == "face":
        mu_G /= sum(FACE)
        for i in range(len(FACE)):
            mu_c_dict[i] /= FACE[i]

    return mu_G, mu_c_dict

def compute_Sigma_W(args, model, mu_c_dict, dataloader):

    Sigma_W = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            #outputs = model.resnet(inputs)
            outputs = model.encoder(inputs)

        features = nn.functional.normalize(outputs, dim=1, p=2)

        for b in range(len(targets)):
            y = targets[b].item()
            Sigma_W += (features[b, :] - mu_c_dict[y]).unsqueeze(1) @ (features[b, :] - mu_c_dict[y]).unsqueeze(0)

    if args.dataset == 'cifar10':
        Sigma_W /= sum(CIFAR10_TRAIN_SAMPLES)
    elif args.dataset == 'cifar100':
        Sigma_W /= sum(CIFAR100_TRAIN_SAMPLES)
    elif args.dataset == "tiny_imagenet":
        Sigma_W /= sum(TINY)
    elif args.dataset == "face":
        Sigma_W /= sum(FACE)


    return Sigma_W.cpu().numpy()


def compute_Sigma_B(mu_c_dict, mu_G):
    Sigma_B = 0
    K = len(mu_c_dict)
    for i in range(K):
        Sigma_B += (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    Sigma_B /= K

    return Sigma_B.cpu().numpy()


def main(args):
    seed_everything(0)
    print(args)

    data = DataModule(args)
    train_dataloader = data.train_dataloader()

    nc1_list = []
    for i in range(20):
        print(f"Loading checkpoint from {args.ckpt_path}: epoch={i * 10 + 9}.ckpt")
        ckpt_path = os.path.join(args.ckpt_path, f"epoch={i * 10 + 9}.ckpt")
        model = ModelModule.load_from_checkpoint(ckpt_path).to(args.device)
        #model = ModelModule(args).to(args.device)

        mu_G, mu_c_dict= compute_info(args = args,  model = model, dataloader = train_dataloader)

        Sigma_W = compute_Sigma_W(args = args, model = model, mu_c_dict = mu_c_dict, dataloader = train_dataloader)
        Sigma_B = compute_Sigma_B(mu_c_dict = mu_c_dict, mu_G = mu_G)

        NC1 = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict)

        nc1_list.append(NC1)
        print(f"epoch: {i * 10 + 9}, NC1: {NC1}, log(NC1): {np.log(NC1)}")

    print("NC1 list:", nc1_list)



if __name__ == '__main__':
    args = get_eval_arguments()
    main(args)
