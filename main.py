import os
import random
from argparse import ArgumentParser
import pytorch_lightning as pl
from DataModule import *
from ModelModule import *
import json
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint,RichProgressBar
from pytorch_lightning.strategies.ddp import DDPStrategy
from args import *


def main(args):
    # args
    pl.seed_everything(0)
    print(args)

    # lightning module
    data = DataModule(args)
    model = ModelModule(args)

    # wandb logger, logger is saved in ./Three_Loss_Layerwise/
    if args.CMFClassifier:
        logger_name = 'CMF-{}-{}-d{}-tau{}-epoch{}-lr{}-momentum-{}'.format(args.encoder, args.dataset, args.feature_dim,
                                                            args.temperature, args.max_epochs, args.learning_rate, args.CMF_momentum)
    else:
        logger_name = '{}-{}-d{}-tau{}-epoch{}-lr{}'.format(args.encoder, args.dataset, args.feature_dim,
                                                            args.temperature, args.max_epochs, args.learning_rate)
    wandb_logger = WandbLogger(project='NCLargeNumClasses',
                               log_model=False,
                               name=logger_name)

    # save checkpoint, checkpoints are saved in ./saved_models/
    model_dir = os.path.join(args.save_dir, logger_name)
    checkpoint_callback = ModelCheckpoint(dirpath = model_dir,
                                          filename = '{epoch}',
                                          save_last = True,
                                          save_top_k = -1,
                                          every_n_epochs = args.save_every_n_epochs)
    # trainer
    trainer = pl.Trainer(devices=[6,7],
                         accelerator="gpu",
                         precision=16,
                         strategy = DDPStrategy(find_unused_parameters=False),
                         max_epochs = args.max_epochs,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback, RichProgressBar()],
                         #resume_from_checkpoint="./saved_models/resnet18-face-d512-tau10.0-epoch200-lr0.1/last.ckpt"
                         )
                         #resume_from_checkpoint="./saved_models/SimCLR-resnet_cifar_compact-cifar100_pair-epoch1000/epoch=199.ckpt")
    # train
    trainer.fit(model = model,
                train_dataloaders = data.train_dataloader(),
                val_dataloaders = data.val_dataloader())
    trainer.test(dataloaders=data.test_dataloader())
    wandb.finish()

if __name__ == '__main__':
    args = get_train_arguments()
    main(args)