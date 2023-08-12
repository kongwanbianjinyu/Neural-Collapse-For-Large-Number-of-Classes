#python GNC1.py --ckpt_path ./saved_models/resnet18-cifar100-d10-tau10.0-epoch200-lr0.1/ --device cuda:2 --dataset cifar100
#python GNC1.py --ckpt_path ./saved_models/densenet121-cifar100-d10-tau10.0-epoch200-lr0.1/ --device cuda --dataset cifar100
#python GNC1.py -ckpt_path ./saved_models/resnext50-cifar100-d10-tau10.0-epoch200-lr0.1/ --device cuda:5 --dataset cifar100

#python GNC1.py --ckpt_path ./saved_models/resnet18-tiny_imagenet-d10-tau20.0-epoch200-lr0.1/ --device cuda --dataset tiny_imagenet
#python GNC1.py --ckpt_path ./saved_models/resnext50-tiny_imagenet-d10-tau20.0-epoch200-lr0.1/ --device cuda --dataset tiny_imagenet
#python GNC1.py --ckpt_path ./saved_models/densenet121-tiny_imagenet-d10-tau10.0-epoch200-lr0.1/ --device cuda:1 --dataset tiny_imagenet


python GNC1.py --ckpt_path ./saved_models/resnext50-face-d512-tau50.0-epoch200-lr0.1/ --device cuda:6 --dataset face --data_aug --aug_method regular50



