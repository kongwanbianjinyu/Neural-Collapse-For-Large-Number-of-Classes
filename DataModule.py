import random
import torchvision
import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from PIL import Image


class DataModule():
    def __init__(self, args):
        self.args = args
        if self.args.dataset == "cifar10":
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2471, 0.2435, 0.2616)
        elif self.args.dataset == "cifar100":
            self.mean = (0.5071, 0.4865, 0.4409)
            self.std = (0.2673, 0.2564, 0.2762)
        elif self.args.dataset == "tiny_imagenet":
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
        elif self.args.dataset == "face":
            self.mean = (0.5, 0.5, 0.5)
            self.std = (0.5, 0.5, 0.5)

        self.transform_aug = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ToTensor(),
                T.Normalize(self.mean, self.std)
            ]
        )
        if self.args.dataset == "face":
            self.transform_no_aug = T.Compose(
                [
                    T.Resize([50, 50]),
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std)
                ]
            )
        else:
            self.transform_no_aug = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std)
                ]
            )

        self.transform_aug_64 = T.Compose(
            [
                T.RandomCrop(64, padding=8),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ToTensor(),
                T.Normalize(self.mean, self.std)
            ]
        )

        self.transform_aug_50 = T.Compose(
            [
                T.Resize([50, 50]),
                T.RandomCrop(50, padding=4),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ToTensor(),
                T.Normalize(self.mean, self.std)
            ]
        )

    def train_dataloader(self):
        # whether use data augmentation
        if self.args.data_aug:
            # choose augmentation methods
            if self.args.aug_method == "regular":
                transform = self.transform_aug
            elif self.args.aug_method =="regular64":
                transform = self.transform_aug_64
            elif self.args.aug_method =="regular50":
                transform = self.transform_aug_50
            # choose dataset
            if self.args.dataset == "cifar10":
                dataset = CIFAR10(root="./data", download=True, train=True, transform=transform)
            elif self.args.dataset == "cifar100":
                dataset = CIFAR100(root="./data", download=True, train=True, transform=transform)
            elif self.args.dataset == "tiny_imagenet":
                train_dir = './data/tiny-imagenet-200/train'
                dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
            elif self.args.dataset == "face":
                train_dir = './data/BUPT-CBFace-50/images'
                dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)

        else:
            if self.args.dataset == "cifar10":
                dataset = CIFAR10(root="./data", download=True, train=True, transform=self.transform_no_aug)
            elif self.args.dataset == "cifar100":
                dataset = CIFAR100(root="./data", download=True, train=True, transform=self.transform_no_aug)
            elif self.args.dataset == "tiny_imagenet":
                train_dir = './data/tiny-imagenet-200/train'
                dataset = torchvision.datasets.ImageFolder(train_dir, transform=self.transform_no_aug)
            elif self.args.dataset == "face":
                train_dir = './data/BUPT-CBFace-50/images'
                dataset = torchvision.datasets.ImageFolder(train_dir, transform=self.transform_no_aug)


        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        if self.args.dataset == "cifar10":
            dataset = CIFAR10(root="./data", download=True, train=False, transform=self.transform_no_aug)
        elif self.args.dataset == "cifar100":
            dataset = CIFAR100(root="./data", download=True, train=False, transform=self.transform_no_aug)
        elif self.args.dataset == "tiny_imagenet":
            val_dir = './data/tiny-imagenet-200/val'
            dataset = torchvision.datasets.ImageFolder(val_dir, transform=self.transform_no_aug)
        elif self.args.dataset == "face":
            dataset = CIFAR10(root="./data", download=True, train=False, transform=self.transform_no_aug)


        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size, #250
            num_workers=self.args.num_workers,
        )
        return dataloader

    def test_dataloader(self):
        if self.args.dataset == "cifar10":
            dataset = CIFAR10(root="./data", download=True, train=False, transform=self.transform_no_aug)
        elif self.args.dataset == "cifar100":
            dataset = CIFAR100(root="./data", download=True, train=False, transform=self.transform_no_aug)
        elif self.args.dataset == "tiny_imagenet":
            test_dir = './data/tiny-imagenet-200/test'
            dataset = torchvision.datasets.ImageFolder(test_dir, transform=self.transform_no_aug)
        elif self.args.dataset == "face":
            dataset = CIFAR10(root="./data", download=True, train=False, transform=self.transform_no_aug)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size, #250
            num_workers=self.args.num_workers,
        )
        return dataloader



