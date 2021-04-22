import os

from domain.base import Factory, Yaml
from properties import APPLICATION_PROPERTIES
from logger import logger


class HyperparameterFactory(Factory):

    def __init__(self, datamodule_params, trainer_params, model_params):
        super(HyperparameterFactory, self).__init__()
        self.datamodule_params = datamodule_params
        self.trainer_params = trainer_params
        self.model_params = model_params

    @classmethod
    def create(cls, data_name, model_name):
        datamodule_yaml_filename = ""
        trainer_yaml_filename = ""
        model_yaml_filename = ""

        # Datamodule
        if data_name == "mnist":
            datamodule_yaml_filename = "MNIST-datamodule.yaml"

        # Trainer
        if model_name == "fc":
            trainer_yaml_filename = "FC-trainer.yaml"

        # Model
        if model_name == "fc":
            model_yaml_filename = "FC.yaml"

        # Get yaml and transform to arguments
        datamodule_params = Yaml(
            path=os.path.join(APPLICATION_PROPERTIES.DATASET_HYPERPARAMETER_DIRECTORY_PATH, datamodule_yaml_filename)
        ).to_hyperparameters()
        trainer_params = Yaml(
            path=os.path.join(APPLICATION_PROPERTIES.TRAINER_HYPERPARAMETER_DIRECTORY_PATH, trainer_yaml_filename)
        ).to_hyperparameters()
        model_params = Yaml(
            path=os.path.join(APPLICATION_PROPERTIES.MODEL_HYPERPARAMETER_DIRECTORY_PATH, model_yaml_filename)
        ).to_hyperparameters()

        logger.info(f"Selected datamodule arguments ({datamodule_yaml_filename}) : \n"
                    f"{datamodule_params}")
        logger.info(f"Selected trainer arguments ({trainer_yaml_filename}) : \n"
                    f"{trainer_params}")
        logger.info(f"Selected model arguments ({model_yaml_filename}) : \n"
                    f"{model_params}")

        return cls(datamodule_params=datamodule_params, trainer_params=trainer_params, model_params=model_params)

