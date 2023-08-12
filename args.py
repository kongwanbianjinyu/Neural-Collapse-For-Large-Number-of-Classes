from argparse import ArgumentParser
import pytorch_lightning as pl

def get_train_arguments():
    """Defines command-line arguments, and parses them.
    """
    parser = ArgumentParser()

    # data
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "tiny_imagenet", "face"])
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--aug_method', type=str, default="regular", choices=["regular", "regular64", "regular50"])

    # model
    parser.add_argument("--encoder", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnext50", "resnext101",
                                                                            "densenet121", "inceptionv3", "ViT", "mobilenet", "mobilenetv2"])
    parser.add_argument("--feature_dim", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=100)

    # CMF(Class-Mean Feature Classifier)
    parser.add_argument('--CMFClassifier', action='store_true')
    parser.add_argument('--CMF_momentum', type=float, default=0.9)

    # optimization
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)#

    parser.add_argument('--temperature', type=float, default=7)


    # Trainer Arguments
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument('--save_dir', type=str, default='./saved_models/',
                        help='base directory for saving PyTorch model. (default: ./saved_models/)')
    parser.add_argument("--save_every_n_epochs", type=int, default=10)

    args = parser.parse_args()
    return args

def get_eval_arguments():
    """Defines command-line arguments, and parses them.
    """
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--ckpt_path', type=str, default="ckpt_path")

    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "tiny_imagenet", "face"])
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--aug_method', type=str, default="regular", choices=["regular", "regular64", "regular50"])

    parser.add_argument("--encoder", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "resnext50", "resnext101",
                                 "densenet121", "inceptionv3", "ViT", "mobilenet", "mobilenetv2"])
    parser.add_argument("--feature_dim", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=100)


    args = parser.parse_args()
    return args
