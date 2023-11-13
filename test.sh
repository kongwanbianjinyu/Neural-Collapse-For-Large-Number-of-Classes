#python test.py --ckpt_path ./saved_models/resnet18-tiny_imagenet-d10-tau20.0-epoch200-lr0.1/ --device cuda:2 --dataset tiny_imagenet --num_classes 200 --temperature 20
python test.py --ckpt_path ./saved_models/resnet18-cifar100-d10-tau10.0-epoch200-lr0.1/ --device cuda:2 --dataset cifar100 --num_classes 100 --temperature 10
