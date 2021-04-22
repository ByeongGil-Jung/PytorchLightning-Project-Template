from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback, EarlyStopping

from logger import logger
import copy


class StatusListenerCallback(Callback):

    def __init__(self, *args, **kwargs):
        super(StatusListenerCallback, self).__init__(*args, **kwargs)

    @classmethod
    def append(cls, callback_list, arg_dict):
        pass

    def on_train_start(self, trainer, pl_module: LightningModule) -> None:
        logger.info("Start to train ...")

    def on_train_end(self, trainer, pl_module: LightningModule) -> None:
        trainer.model_metadata.init()
        logger.info("Finished the training.")

    def on_validation_end(self, trainer, pl_module: LightningModule) -> None:
        # metrics_dict = copy.copy(trainer.callback_metrics)
        metrics_dict = trainer.callback_metrics

        train_loss = metrics_dict["train_loss"] if "train_loss" in metrics_dict else None
        train_accuracy = metrics_dict["train_accuracy"] if "train_accuracy" in metrics_dict else None
        train_precision = metrics_dict["train_precision"] if "train_precision" in metrics_dict else None
        train_recall = metrics_dict["train_recall"] if "train_recall" in metrics_dict else None
        train_f1_score = metrics_dict["train_f1_score"] if "train_f1_score" in metrics_dict else None

        val_loss = metrics_dict["val_loss"] if "val_loss" in metrics_dict else None
        val_accuracy = metrics_dict["val_accuracy"] if "val_accuracy" in metrics_dict else None
        val_precision = metrics_dict["val_precision"] if "val_precision" in metrics_dict else None
        val_recall = metrics_dict["val_recall"] if "val_recall" in metrics_dict else None
        val_f1_score = metrics_dict["val_f1_score"] if "val_f1_score" in metrics_dict else None

        print_format = f"[Epoch {trainer.current_epoch}]\n" \
                       f" Train - Loss : {train_loss} | " \
                       f"Accuracy : {train_accuracy} | " \
                       f"Precision : {train_precision} | " \
                       f"Recall : {train_recall} | " \
                       f"F1 : {train_f1_score}\n" \
                       f" Val - Loss : {val_loss} | " \
                       f"Accuracy : {val_accuracy} | " \
                       f"Precision : {val_precision} | " \
                       f"Recall : {val_recall} | " \
                       f"F1 : {val_f1_score}"

        print(print_format)

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:
        logger.info("Start to test ...")

    def on_test_end(self, trainer, pl_module: LightningModule) -> None:
        trainer.model_metadata.init()
        logger.info("Finished the testing.")

    def on_keyboard_interrupt(self, trainer, pl_module: LightningModule) -> None:
        logger.info("Keyboard interrupted.")


class EarlyStoppingCallback(EarlyStopping):

    def __init__(self, *args, **kwargs):
        super(EarlyStoppingCallback, self).__init__(*args, **kwargs)
        self.best_epoch = 0
        self.is_saved_checkpoint = True

    @classmethod
    def append(cls, callback_list, arg_dict):
        pass

    def on_save_checkpoint(self, trainer, pl_module):
        super().on_save_checkpoint(trainer, pl_module)

    def on_load_checkpoint(self, checkpointed_state):
        super().on_load_checkpoint(checkpointed_state)
        print("Load checkpoint")

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)

        if self.wait_count == 0:
            self.best_epoch = trainer.current_epoch

        print(f"Early Stopping [{self.wait_count}/{self.patience}] - Best Epoch : {self.best_epoch} | Best Score : {self.best_score}")




