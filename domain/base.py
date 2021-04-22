from collections.abc import MutableMapping

import yaml


class Domain(object):

    def __init__(self, *args, **kwargs):
        pass


class Factory(Domain):

    def __init__(self, *args, **kwargs):
        super(Factory, self).__init__(*args, **kwargs)

    @classmethod
    def create(cls, *args, **kwargs):
        pass


class Hyperparameters(Domain):

    def __init__(self, *args, **kwargs):
        super(Hyperparameters, self).__init__(*args, **kwargs)
        self.__dict__.update(kwargs)

        # Converting
        self._convert()

    def __repr__(self):
        return self.__dict__.__repr__()

    def __call__(self, *args, **kwargs):
        return self.__dict__

    def to_dict(self):
        return self.__dict__

    def _convert(self):
        _dict = self.__dict__


class Yaml(Domain):

    def __init__(self, path, *args, **kwargs):
        super(Yaml, self).__init__(*args, **kwargs)
        self.__dict__ = self._read_yaml(path)

    def __repr__(self):
        return self.__dict__.__repr__()

    def to_hyperparameters(self):
        params = Hyperparameters()
        params.__dict__ = self.__dict__

        return params

    def _read_yaml(self, path):
        with open(path, "rb") as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

        return yaml_dict