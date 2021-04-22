import torch

from domain.base import Domain


class Criterion(Domain):

    def __init__(self, *args, **kwargs):
        super(Criterion, self).__init__(*args, **kwargs)

    @classmethod
    def mse_loss(cls, x, x_hat):
        return torch.mean(torch.pow((x - x_hat), 2))

    @classmethod
    def reconstruction_error(cls, x, x_hat):
        return ((x_hat - x) ** 2).mean(axis=1)
