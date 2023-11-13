#python GNC3.py --ckpt_path ./saved_models/resnet18-cifar100-d10-tau10.0-epoch200-lr0.1/ --device cuda:4 --dataset cifar100
#python GNC3.py --ckpt_path ./saved_models/densenet121-cifar100-d10-tau10.0-epoch200-lr0.1/ --device cuda --dataset cifar100
#python GNC3.py --ckpt_path ./saved_models/resnext50-cifar100-d10-tau10.0-epoch200-lr0.1/ --device cuda:3 --dataset cifar100
#python GNC3.py --ckpt_path ./saved_models/resnet18-tiny_imagenet-d10-tau20.0-epoch200-lr0.1/ --device cuda --dataset tiny_imagenet
#python GNC3.py --ckpt_path ./saved_models/resnext50-tiny_imagenet-d10-tau20.0-epoch200-lr0.1/ --device cuda --dataset tiny_imagenet
#python GNC3.py --ckpt_path ./saved_models/densenet121-tiny_imagenet-d10-tau10.0-epoch200-lr0.1/ --device cuda --dataset tiny_imagenet

#python GNC3.py --ckpt_path ./saved_models/densenet121-face-d512-tau50.0-epoch200-lr0.1/ --device cuda:3 --dataset face
#python GNC3.py --ckpt_path ./saved_models/resnext50-face-d512-tau50.0-epoch200-lr0.1/ --device cuda:3 --dataset face --data_aug --aug_method regular50 --encoder resnext50 --feature_dim 512 --num_classes 10000 --temperature 50
python GNC3.py --device cuda:3 --dataset tiny_imagenet --encoder resnext50 --feature_dim 10 --num_classes 200 --temperature 10
