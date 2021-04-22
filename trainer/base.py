from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from domain.metadata import ModelMetadata
from properties import APPLICATION_PROPERTIES
from logger import logger
from trainer.callback import EarlyStoppingCallback, StatusListenerCallback


class TrainerBase(pl.Trainer):

    def __init__(self, model_metadata: ModelMetadata, *args, **kwargs):
        self.model_metadata = model_metadata
        self._convert_arguments(kwargs=kwargs)
        super(TrainerBase, self).__init__(*args, **kwargs)

    def _convert_arguments(self, kwargs):
        model_metadata = self.model_metadata
        arg_dict = kwargs

        # Callback
        if isinstance(arg_dict["callbacks"], list):
            new_callback_list = list()

            # Status listener
            if "status_listener_callback" in arg_dict["callbacks"]:
                new_callback_list.append(StatusListenerCallback())
            # Early stopping
            if "early_stopping_callback" in arg_dict["callbacks"]:
                early_stopping_monitor = APPLICATION_PROPERTIES.VAL_LOSS_REPR
                early_stopping_min_delta = APPLICATION_PROPERTIES.EARLY_STOPPING_MIN_DELTA_DEFAULT
                early_stopping_patience = APPLICATION_PROPERTIES.EARLY_STOPPING_PATIENCE_DEFAULT
                early_stopping_verbose = APPLICATION_PROPERTIES.EARLY_STOPPING_VERBOSE_DEFAULT
                early_stopping_mode = APPLICATION_PROPERTIES.EARLY_STOPPING_MODE_DEFAULT
                early_stopping_strict = APPLICATION_PROPERTIES.EARLY_STOPPING_STRICT_DEFAULT

                if "early_stopping_monitor" in arg_dict:
                    early_stopping_monitor = arg_dict["early_stopping_monitor"]
                    del arg_dict["early_stopping_monitor"]
                if "early_stopping_min_delta" in arg_dict:
                    early_stopping_min_delta = arg_dict["early_stopping_min_delta"]
                    del arg_dict["early_stopping_min_delta"]
                if "early_stopping_patience" in arg_dict:
                    early_stopping_patience = arg_dict["early_stopping_patience"]
                    del arg_dict["early_stopping_patience"]
                if "early_stopping_verbose" in arg_dict:
                    early_stopping_verbose = arg_dict["early_stopping_verbose"]
                    del arg_dict["early_stopping_verbose"]
                if "early_stopping_mode" in arg_dict:
                    early_stopping_mode = arg_dict["early_stopping_mode"]
                    del arg_dict["early_stopping_mode"]
                if "early_stopping_strict" in arg_dict:
                    early_stopping_strict = arg_dict["early_stopping_strict"]
                    del arg_dict["early_stopping_strict"]

                early_stopping_callback = EarlyStoppingCallback(
                    monitor=early_stopping_monitor,
                    min_delta=early_stopping_min_delta,
                    patience=early_stopping_patience,
                    verbose=early_stopping_verbose,
                    mode=early_stopping_mode,
                    strict=early_stopping_strict
                )

                new_callback_list.append(early_stopping_callback)

            arg_dict["callbacks"] = new_callback_list
        else:
            arg_dict["callbacks"] = [StatusListenerCallback()]

        # Logger
        if arg_dict["logger"] == "tensor_board":
            arg_dict["logger"] = TensorBoardLogger(
                save_dir=APPLICATION_PROPERTIES.MODEL_RESULT_DIRECTORY_PATH,
                name=model_metadata.model_name
            )
        else:
            arg_dict["logger"] = TensorBoardLogger(
                save_dir=APPLICATION_PROPERTIES.MODEL_RESULT_DIRECTORY_PATH,
                name=model_metadata.model_name
            )

        # Auto lr find
        self.is_auto_lr_find = arg_dict["is_auto_lr_find"]
        del arg_dict["is_auto_lr_find"]

    def lr_find(self, model, train_loader, val_loader):
        logger.info(f"Start to find the optimal learning rate ...")
        self.lr_finder = self.tuner.lr_find(model=model, train_dataloader=train_loader, val_dataloaders=val_loader)

        optimal_lr = self.lr_finder.suggestion()
        model.hparams.lr = optimal_lr
        logger.info(f"Finished finding the optimal learning rate : {optimal_lr}")
