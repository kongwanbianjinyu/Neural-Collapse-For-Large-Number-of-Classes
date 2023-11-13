#python main.py --dataset cifar100 --data_aug --aug_method regular --encoder resnet18 --feature_dim 10 --max_epochs 200 --batch_size 250 --temperature 10 --learning_rate 0.1 --loss Hardmax


#nohup python main.py --dataset tiny_imagenet --data_aug --aug_method regular64 --encoder resnext50 --feature_dim 10 --num_classes 200 --batch_size 250 --max_epochs 200 --temperature 10 --learning_rate 0.1 > train_resnext_val.log 2>&1 &

#python main.py --dataset face --data_aug --aug_method regular50 --encoder densenet121 --feature_dim 512 --num_classes 10000 --batch_size 250 --max_epochs 400 --temperature 50 --learning_rate 0.1 


#python main.py --CMFClassifier --CMF_momentum 0.9 --dataset cifar100 --data_aug --aug_method regular --encoder resnext50 --feature_dim 10 --max_epochs 200 --batch_size 250 --temperature 10 --learning_rate 0.1

#python main.py --dataset cifar100 --data_aug --aug_method regular --encoder resnet18 --feature_dim 10 --max_epochs 200 --batch_size 250 --temperature 10 --learning_rate 0.1 --loss CE --devices "0,1" --no_normalization
python main.py --dataset cifar100 --data_aug --aug_method regular --encoder resnet18 --feature_dim 10 --max_epochs 200 --batch_size 250 --temperature 7 --learning_rate 0.1 --loss CE --devices "2,3"