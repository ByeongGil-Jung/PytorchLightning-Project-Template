import pytorch_lightning as pl
import torchmetrics
from torch import nn
from torch import optim


class ModelBase(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super(ModelBase, self).__init__(*args, **kwargs)

    def _set_metrics(self, *args, **kwargs):
        num_classes = kwargs["num_classes"] if "num_classes" in kwargs else None

        # Train
        self.train_acc = torchmetrics.Accuracy()
        self.train_precision = torchmetrics.Precision()
        self.train_recall = torchmetrics.Recall()
        self.train_f1 = torchmetrics.F1(num_classes=num_classes) if num_classes else None

        # Validation
        self.validation_acc = torchmetrics.Accuracy()
        self.validation_precision = torchmetrics.Precision()
        self.validation_recall = torchmetrics.Recall()
        self.validation_f1 = torchmetrics.F1(num_classes=num_classes) if num_classes else None

        # Test
        self.test_acc = torchmetrics.Accuracy()
        self.test_precision = torchmetrics.Precision()
        self.test_recall = torchmetrics.Recall()
        self.test_f1 = torchmetrics.F1(num_classes=num_classes) if num_classes else None

    def init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight.data)

    def get_optimizer(self, optimizer_name):
        optimizer = None

        if optimizer_name == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        # Add ...

        return optimizer

    def get_lr_scheduler(self, scheduler_name, optimizer):
        scheduler = None

        if scheduler_name == "cosine_anealing_warm_restart":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.hparams.T_0)
        # Add ...

        return scheduler
