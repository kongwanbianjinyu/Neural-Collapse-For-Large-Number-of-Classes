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
    print(args.device)
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



def duality_distance(mu_c_dict, mu_G, w):
    K = w.shape[0]
    h_mean = torch.zeros_like(w)
    for k,v in mu_c_dict.items():
        h_mean[k,:] = v - mu_G

    h_mean_norm = torch.norm(h_mean).item()
    w_norm = torch.norm(w).item()

    h_mean = nn.functional.normalize(h_mean, dim=1, p=2)
    w = nn.functional.normalize(w, dim=1, p=2)

    # distance = torch.norm(h_mean - w).item()
    distance = 0
    for k in range(K):
        distance += 1 - torch.dot(h_mean[k,:],w[k,:]).item()
    distance = distance / K

    return h_mean_norm, w_norm, distance


def main(args):
    seed_everything(0)
    print(args)

    data = DataModule(args)
    train_dataloader = data.train_dataloader()
    distance_list = []
    w_list = []
    h_list = []

    # initial checkpoint NC:
    i = -1
    print("Initial checkpoint:")
    model = ModelModule(args).to(args.device)

    print("computing info...")
    mu_G, mu_c_dict = compute_info(args=args, model=model, dataloader=train_dataloader)

    w = model.linear.weight.detach().cpu()

    print("duality distance...")
    h_mean_norm, w_norm, distance = duality_distance(mu_c_dict, mu_G, w)
    distance_list.append(distance)
    w_list.append(w_norm)
    h_list.append(h_mean_norm)

    print(f"epoch: {i * 10 + 9},h_mean_norm:{h_mean_norm}, w_norm:{w_norm},distance:{distance}, ")

    # for i in range(20):
    #     print(f"Loading checkpoint from {args.ckpt_path}: epoch={i * 10 + 9}.ckpt")
    #     ckpt_path = os.path.join(args.ckpt_path, f"epoch={i * 10 + 9}.ckpt")
    #     model = ModelModule.load_from_checkpoint(ckpt_path).to(args.device)
    #     #model = ModelModule(args).to(args.device)
    #
    #     print("computing info...")
    #     mu_G, mu_c_dict = compute_info(args = args,  model = model, dataloader = train_dataloader)
    #
    #     w = model.linear.weight.detach().cpu()
    #
    #     print("duality distance...")
    #     h_mean_norm, w_norm, distance = duality_distance(mu_c_dict, mu_G, w)
    #     distance_list.append(distance)
    #     w_list.append(w_norm)
    #     h_list.append(h_mean_norm)
    #
    #     print(f"epoch: {i * 10 + 9},h_mean_norm:{h_mean_norm}, w_norm:{w_norm},distance:{distance}, ")

    print("h_mean_norm:", h_list)
    print("w_norm:", w_list)
    print("Duality Distance:", distance_list)

if __name__ == '__main__':
    args = get_eval_arguments()
    main(args)