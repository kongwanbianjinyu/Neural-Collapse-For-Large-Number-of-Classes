import pytorch_lightning as pl
import torchmetrics
import torch
from torchmetrics.classification import MulticlassAccuracy
import torchvision
from encoders import *
import torch.nn.functional as F


class CMFWeights(nn.Module):
    def __init__(
        self,
        num_classes  = 10,
        feature_dim = 512
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # weights size is (K,d)
        self.register_buffer("weight", torch.zeros(num_classes, feature_dim))

    @torch.no_grad()
    def update(self, features, labels, momentum) -> torch.Tensor:
        """
        Returns l2-normalized class weights by averaging the features from the same class
        """
        # add each feature to weights accroding to its labels
        self.weight = momentum * self.weight
        self.weight.index_add_(0, labels, (1 - momentum) * features)
        # normlize to unit norm
        self.weight = F.normalize(self.weight)
        return self.weight

class ModelModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        # load resnet to extract features
        if args.encoder == "resnet18":
            self.encoder = resnet18(args.feature_dim)
            #self.resnet = resnet18(args.feature_dim)
        elif args.encoder == "resnet34":
            self.encoder = resnet34(args.feature_dim)
        elif args.encoder == "resnet50":
            self.encoder = resnet50(args.feature_dim)
        elif args.encoder == "resnext50":
            self.encoder = resnext50(args.feature_dim)
        elif args.encoder == "resnext101":
            self.encoder = resnext101(args.feature_dim)
        elif args.encoder == "densenet121":
            self.encoder = densenet121(args.feature_dim)
        elif args.encoder == "inceptionv3":
            self.encoder =  inceptionv3(args.feature_dim)
        elif args.encoder == "ViT":
            self.encoder = vit(args.feature_dim)
        elif args.encoder == "mobilenet":
            self.encoder = mobilenet(args.feature_dim)
        elif args.encoder == "mobilenetv2":
            self.encoder = mobilenetv2(args.feature_dim)

        use CMF classifier or linear classifier
        if args.CMFClassifier:
            self.CMFweights = CMFWeights(num_classes=args.num_classes, feature_dim=args.feature_dim)
        else:
            self.linear = nn.Linear(args.feature_dim, args.num_classes, bias=False)

        self.criterion = torch.nn.CrossEntropyLoss()
        # define acc
        self.accuracy = MulticlassAccuracy(num_classes=args.num_classes)

    def forward(self, batch, stage):
        x,y = batch
        features = self.encoder(x)
        features = F.normalize(features)

        if self.args.CMFClassifier:
            if stage == "train":
                self.CMFweights.update(features=features, labels=y, momentum=self.args.CMF_momentum)
            weights = self.CMFweights.weight
        else:
            weights = F.normalize(self.linear.weight)

        logits = (features @ weights.t()) * self.args.temperature
        #logits = self.linear(features)
        loss = self.criterion(logits, y)
        accuracy = self.accuracy(logits, y)

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, "train")
        self.log("loss/train", loss, prog_bar=True)
        self.log("acc/train", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, "val")
        self.log("loss/val", loss, prog_bar=True)
        self.log("acc/val", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, "test")
        self.log("loss/test", loss, prog_bar=True)
        self.log("acc/test", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.args.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.args.learning_rate,
                                        momentum=0.9,
                                        weight_decay=self.args.weight_decay,
                                        nesterov=True)
        elif self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.max_epochs)
        return [optimizer], [scheduler]