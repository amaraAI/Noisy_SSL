import pytorch_lightning as pl
import torch
import torch.nn as nn
import lightly
import torchvision


class SimsiamModel(pl.LightningModule):
    def __init__(self,batch_size=512,input_size=32,lr=0.06,num_ftrs=512,max_epochs=300, backbone_type='resnet18', momentum=0.9, weight_decay=5e-4):
        super().__init__()
        self.num_ftrs= num_ftrs
        self.lr = lr #* batch_size / input_size
        self.proj_hidden_dim = num_ftrs
        self.pred_hidden_dim = int(num_ftrs/4)
        self.out_dim = num_ftrs
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.max_epochs = max_epochs

        # create a ResNet backbone and remove the classification head
        if backbone_type == 'resnet18':
            resnet = torchvision.models.resnet18()
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
        )

        # create a simsiam based on ResNet
        self.resnet_simsiam = \
            model = lightly.models.SimSiam(backbone,num_ftrs=self.num_ftrs,proj_hidden_dim=self.proj_hidden_dim,pred_hidden_dim=self.pred_hidden_dim,out_dim=self.out_dim)
        # create our loss with the optional
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()

    def forward(self, x):
        self.resnet_simsiam(x)

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        #print("THE BATCH SIZE IS", batch)
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_simsiam(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simsiam.parameters(), lr=self.lr,
                                momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]



class BYOLModel(pl.LightningModule):
    def __init__(self,batch_size=512,input_size=32,lr=0.06,num_ftrs=512,max_epochs=300, backbone_type='resnet18', momentum=0.9, weight_decay=5e-4):
        super().__init__()
        self.num_ftrs= num_ftrs
        self.lr = lr #* batch_size / input_size
        self.hidden_dim = num_ftrs *2
        self.out_dim = num_ftrs
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.max_epochs = max_epochs

        # create a ResNet backbone and remove the classification head
        if backbone_type == 'resnet18':
            resnet = torchvision.models.resnet18()
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
        )

        # create a simsiam based on ResNet
        self.resnet_byol = \
            model = lightly.models.BYOL(backbone,num_ftrs=self.num_ftrs,hidden_dim=self.hidden_dim,out_dim= self.out_dim,m=0.9)
        # create our loss with the optional
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()

    def forward(self, x):
        self.resnet_byol(x)

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        #print("THE BATCH SIZE IS", batch)
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_byol(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_byol.parameters(), lr=self.lr,
                                momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]


class MocoModel(pl.LightningModule):
    def __init__(self,batch_size=512,input_size=32,lr=0.06,num_ftrs=512,max_epochs=300,temperature=0.1,memory_bank_size=4096, momentum=0.9, weight_decay=5e-4):
        super().__init__()
        self.num_ftrs= num_ftrs
        self.lr = lr #* batch_size / input_size
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.max_epochs = max_epochs
        self.memory_bank_size = memory_bank_size
        self.temperature = temperature
        self.m= 0.99

        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18', 1, num_splits=8)
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # create a moco based on ResNet
        self.resnet_moco = \
            lightly.models.MoCo(backbone, num_ftrs=self.num_ftrs, m=self.m, batch_shuffle=True)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=self.temperature,
            memory_bank_size=self.memory_bank_size)

    def forward(self, x):
        self.resnet_moco(x)

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_moco(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()


    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_moco.parameters(), lr=self.lr,
                                momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,self.max_epochs)
        return [optim], [scheduler]