#python NCC.py --ckpt_path ./saved_models/resnet18-cifar100-d10-tau10.0-epoch200-lr0.1/ --device cuda:0 --dataset cifar100 --data_aug --aug_method regular
#python NCC.py --ckpt_path ./saved_models/densenet121-cifar100-d10-tau10.0-epoch200-lr0.1/ --device cuda:0 --dataset cifar100 --data_aug --aug_method regular
#python NCC.py --ckpt_path ./saved_models/resnext50-cifar100-d10-tau10.0-epoch200-lr0.1/ --device cuda:1 --dataset cifar100 --data_aug --aug_method regular

#python NCC.py --ckpt_path ./saved_models/resnet18-tiny_imagenet-d10-tau20.0-epoch200-lr0.1/ --device cuda:0 --dataset tiny_imagenet --data_aug --aug_method regular64
#python NCC.py  --ckpt_path ./saved_models/resnext50-tiny_imagenet-d10-tau20.0-epoch200-lr0.1/ --device cuda:0 --dataset tiny_imagenet --data_aug --aug_method regular64
python NCC.py --ckpt_path ./saved_models/densenet121-tiny_imagenet-d10-tau10.0-epoch200-lr0.1/ --device cuda:1 --dataset tiny_imagenet --data_aug --aug_method regular64

#python NCC.py --ckpt_path ./saved_models/resnet18-cifar100-d10-tau1.0-epoch200-lr0.1/ --device cuda:0 --dataset cifar100 --data_aug --aug_method regular