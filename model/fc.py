import torch
import torchmetrics
from torch.nn import functional as F

from model.base import ModelBase
from properties import APPLICATION_PROPERTIES


class FullyConnectedLayer(ModelBase):

    def __init__(self, *args, **kwargs):
        super(FullyConnectedLayer, self).__init__()
        self.save_hyperparameters()
        self._set_metrics(num_classes=10)

        # Layers
        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

        self.init_weight()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))

        return x

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_batch_pred = self(x_batch)  # Hypothesis

        prob_batch = F.softmax(input=y_batch_pred)
        loss = F.nll_loss(input=torch.log(prob_batch), target=y_batch)

        # Logging
        self.log(APPLICATION_PROPERTIES.TRAIN_LOSS_REPR, loss)
        self.log(APPLICATION_PROPERTIES.TRAIN_ACCURACY_REPR, self.train_acc(preds=prob_batch, target=y_batch))
        self.log(APPLICATION_PROPERTIES.TRAIN_PRECISION_REPR, self.train_precision(preds=prob_batch, target=y_batch))
        self.log(APPLICATION_PROPERTIES.TRAIN_RECALL_REPR, self.train_recall(preds=prob_batch, target=y_batch))
        self.log(APPLICATION_PROPERTIES.TRAIN_F1_SCORE_REPR, self.train_f1(preds=prob_batch, target=y_batch))

        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_batch_pred = self(x_batch)

        prob_batch = F.softmax(input=y_batch_pred)
        loss = F.nll_loss(input=torch.log(prob_batch), target=y_batch)

        # Logging
        self.log(APPLICATION_PROPERTIES.VAL_LOSS_REPR, loss)
        self.log(APPLICATION_PROPERTIES.VAL_ACCURACY_REPR, self.validation_acc(preds=prob_batch, target=y_batch))
        self.log(APPLICATION_PROPERTIES.VAL_PRECISION_REPR, self.validation_precision(preds=prob_batch, target=y_batch))
        self.log(APPLICATION_PROPERTIES.VAL_RECALL_REPR, self.validation_recall(preds=prob_batch, target=y_batch))
        self.log(APPLICATION_PROPERTIES.VAL_F1_SCORE_REPR, self.validation_f1(preds=prob_batch, target=y_batch))

    def test_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_batch_pred = self(x_batch)

        prob_batch = F.softmax(input=y_batch_pred)
        loss = F.nll_loss(input=torch.log(prob_batch), target=y_batch)

        # # Logging
        self.log(APPLICATION_PROPERTIES.TEST_LOSS_REPR, loss)
        self.log(APPLICATION_PROPERTIES.TEST_ACCURACY_REPR, self.test_acc(preds=prob_batch, target=y_batch))
        self.log(APPLICATION_PROPERTIES.TEST_PRECISION_REPR, self.test_precision(preds=prob_batch, target=y_batch))
        self.log(APPLICATION_PROPERTIES.TEST_RECALL_REPR, self.test_recall(preds=prob_batch, target=y_batch))
        self.log(APPLICATION_PROPERTIES.TEST_F1_SCORE_REPR, self.test_f1(preds=prob_batch, target=y_batch))

    def configure_optimizers(self):
        optimizer = self.get_optimizer(optimizer_name=self.hparams.optimizer)
        scheduler = self.get_lr_scheduler(scheduler_name=self.hparams.lr_scheduler, optimizer=optimizer)

        optimizer_output = optimizer

        if scheduler:
            optimizer_output = [optimizer], [scheduler]

        return optimizer_output
