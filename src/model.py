import pytorch_lightning as pl
from config import Config
from efficientnet_pytorch import EfficientNet
import torchvision
import torch.nn as nn
# * Encoder/Features extractor: esnet, efficientnet, densenet)
# * Decoder/Classifier : inear layer (in_tures, n_classes)
# * loss_fn : CrossEntropyLoss
# * Optimize metrics : accuracy
# * Optimizer : Adam, AdamW, SGD
# * learning rate : (3e-5...1e-1)
# * lr scheduler : linear with warmup, ReduceLROnPlateau
# * pretrained : Always true


class Model(pl.LightningModule):
    def __init__(self, config: Config,
                 len_train_ds: int,
                 steps_per_epoch=None,
                 class_w=None
                 ):
        super(Model, self).__init__()
        config_dict = config.__dict__.items()
        config_dict = dict(
            [item for item in config_dict if '__' not in item[0]])
        self.config = config_dict
        self.len_train_ds = len_train_ds

        # save config as hparam
        self.save_hyperparameters(self.config)

        if steps_per_epoch is None:
            self.steps_per_epoch = self.len_train_ds // self.hparams.train_batch_size
        else:
            self.steps_per_epoch = steps_per_epoch
        # load pretrained model from torchvision or anywheare else
        try:
            self.encoder = getattr(
                torchvision.models, self.hparams.base_model)(pretrained=True)
        except:
            self.encoder = getattr(
                EfficientNet.from_pretrained, self.hparams.base_model)

        if "resnet" in self.hparams.base_model:
            try:
                # get num output features from features extractor
                self.num_ftrs = self.encoder.fc.out_features
            except:
                self.num_ftrs = self.encoder._fc.out_features

        elif "efficientnet" in self.hparams.base_model:
            try:
                self.num_ftrs = self.encoder.fc.out_features
            except:
                self.num_ftrs = self.encoder._fc.out_features

        elif "densenet" in self.hparams.base_model:
            try:
                self.num_ftrs = self.encoder.classifier.out_features
            except:
                pass

        if self.hparams.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=.35)
        self.decoder = nn.Linear(
            in_features=self.num_ftrs, out_features=self.hparams.num_classes)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_w)

        # optimization stuff
        self.warmup_steps = self.steps_per_epoch // 3
        self.total_steps = self.steps_per_epoch * \
            self.hparams.num_epochs - self.warmup_steps

    #########
    # methods
    #########
    def forward(self, images, targets=None):
        # extract features
        out = self.encoder(images)
        # apply dropout
        out = self.dropout(out)
        # apply classifier
        out = self.decoder(out)

        return out

    def configure_optimizers(self):
        if self.hparams.freeze:
            opt = optim.AdamW(self.classifier.parameters(),
                              lr=self.hparams.lr,
                              eps=1e-8,
                              weight_decay=0.01
                              )
        else:
            opt = optim.AdamW(self.parameters(),
                              lr=self.hparams.lr,
                              eps=1e-8,
                              weight_decay=0.01
                              )

        scheduler = get_linear_schedule_with_warmup(
            optimizer=opt,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )

        return [opt], [scheduler]

    def training_step(self, batch, batch_idx):
        xs, ys = batch['images'], batch['targets']
        # make predictions
        predictions = self(xs)

        # compute metrics
        # loss
        loss = self.get_loss(preds=predictions, targets=ys)

        # accuracy
        acc = self.get_acc(preds=predictions, targets=ys)

        # logging stuff
        self.log('train_acc', acc, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss,
                'accuracy': acc,
                "predictions": predictions,
                'targets': ys}

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Train",
                                          avg_acc,
                                          self.current_epoch)

    def validation_step(self, batch, batch_idx):
        xs, ys = batch['images'], batch['targets']
        # make predictions
        predictions = self(xs)

        # compute metrics
        # loss
        loss = self.get_loss(preds=predictions, targets=ys)

        # accuracy
        acc = self.get_acc(preds=predictions, targets=ys)

        # logging stuff
        self.log('val_acc', acc, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss,
                'accuracy': acc,
                "predictions": predictions,
                'targets': ys}

    def validation_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Validation",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Validation",
                                          avg_acc,
                                          self.current_epoch)

    def predict(self, dataloader, batch_size=1):
        if batch_size == 1:
            try:
                preds = self(dataloader.unsqueeze(0))
            except:
                preds = self(dataloader)
        else:
            preds = self(dataloader)

        return preds.detach().cpu().numpy().flatten()

    def get_loss(self, preds, targets):
        preds = preds.cpu()
        targets = targets.cpu()
        return self.loss_fn(input=preds, target=targets)

    def get_acc(self, preds, targets):
        preds = preds.cpu()
        targets = targets.cpu()
        return accuracy(pred=preds, target=targets)
