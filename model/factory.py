from model.fc import FullyConnectedLayer
from domain.base import Factory


class ModelFactory(Factory):

    def __init__(self):
        super(ModelFactory, self).__init__()

    @classmethod
    def create(cls, model_name, model_params):
        model = None

        if model_name == "fc":
            model = FullyConnectedLayer

        return model(**model_params)
