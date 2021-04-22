import os

from domain.base import Domain, Hyperparameters
from properties import APPLICATION_PROPERTIES


class Metadata(Domain):

    def __init__(self, *args, **kwargs):
        super(Metadata, self).__init__(*args, **kwargs)


class Information(Hyperparameters):

    def __init__(self, *args, **kwargs):
        super(Information, self).__init__(*args, **kwargs)


class ModelMetadata(Metadata):

    def __init__(self, model_name, information=None, *args, **kwargs):
        super(ModelMetadata, self).__init__(*args, **kwargs)
        self.model_name = model_name
        self.information = information
        self.model_file_metadata = ModelFileMetadata(model_name=model_name)

    def __repr__(self):
        return f"{self.model_name}"

    def init(self):
        self.model_file_metadata = ModelFileMetadata(model_name=self.model_name)


class ModelFileMetadata(Metadata):

    def __init__(self, model_name, plot_ext=".png", *args, **kwargs):
        super(ModelFileMetadata, self).__init__(*args, **kwargs)
        self.model_name = model_name
        self.model_dir_path = os.path.join(APPLICATION_PROPERTIES.MODEL_RESULT_DIRECTORY_PATH, self.model_name)
        self.model_latest_version_dir_path = os.path.join(self.model_dir_path, self.get_latest_version())
        self.optimal_lr_plot_path = os.path.join(self.model_latest_version_dir_path, "optimal_lr.png")

        # Extension
        self.plot_ext = plot_ext

    def get_version_list(self):
        version_list = list()

        if os.path.isdir(self.model_dir_path):
            for version in os.listdir(self.model_dir_path):
                if os.path.isdir(os.path.join(self.model_dir_path, version)):
                    version_list.append(version)

        return version_list

    def get_latest_version(self):
        if self.get_version_list():
            return sorted(self.get_version_list(), key=lambda version: int(version[-1]))[-1]
        else:
            return ""

    def get_latest_version_number(self):
        if self.get_latest_version():
            return int(self.get_latest_version()[-1])
        else:
            return -1
