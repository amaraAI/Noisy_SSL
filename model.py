import pytorch_lightning as pl
import torch
import torch.nn as nn
import lightly
import torchvision


class SimsiamModel(pl.LightningModule):
    def __init__(self, backbone_type='resnet-18', max_epochs=100, num_ftrs=512, proj_hidden_dim=512, pred_hidden_dim=128, out_dim = 512, num_mlp_layers=2,batch_shuffle=True, lr=0.05 * 128 /256, momentum=0.9, weight_decay=5e-4):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.max_epochs = max_epochs

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
        )

        # create a simsiam based on ResNet
        self.resnet_simsiam = \
            model = lightly.models.SimSiam(backbone,num_ftrs=num_ftrs,proj_hidden_dim=proj_hidden_dim,pred_hidden_dim=pred_hidden_dim,out_dim=out_dim,num_mlp_layers=num_mlp_layers)
        # create our loss with the optional memory bank
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
