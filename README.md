# Neural-Collapse-For-Large-Number-of-Classes


## Setup

1. Create conda environment and activate it :
```conda create -n gnc python=3.10```
```conda activate gnc```

2. Install pytorch according to your cuda version: https://pytorch.org/get-started/locally/, for cuda 12.1:
``` conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia ```

3. Install pytorch-lightning 1.9.0(Must use this version):
```pip install pytorch-lightning==1.9.0```

## Train

1. Train with normalization of features and classifiers:
   
```python main.py --dataset cifar100 --data_aug --aug_method regular --encoder resnet18 --feature_dim 10 --max_epochs 200 --batch_size 250 --temperature 7 --learning_rate 0.1 --loss CE --devices "0,1"```

3. Train without normalization(standard):
   
```python main.py --dataset cifar100 --data_aug --aug_method regular --encoder resnet18 --feature_dim 10 --max_epochs 200 --batch_size 250 --temperature 7 --learning_rate 0.1 --loss CE --devices "0,1" --no_normalization```

4. Train with CMF(Class-Mean Feature) classifier:

   ```python main.py --dataset cifar100 --data_aug --aug_method regular --encoder resnet18 --feature_dim 10 --max_epochs 200 --batch_size 250 --temperature 7 --learning_rate 0.1 --loss CE --devices "0,1" --CMFClassifier --CMF_momentum 0.9```

   The checkpoint file .ckpt would be saved in folder "./saved_models" and the experiment name would be the format of {encoder}-{dataset}-d{feature_dim}-tau{}-epoch{}-lr{}.

## Compute GNC1/GNC2/GNC3
Pass the checkpoint file path to --ckpt_file and run:

```GNC1.sh GNC2.sh GNC3.sh```

## Solve Softmax Code problem Numerically
run:

```hardmax_solution.sh```




 
