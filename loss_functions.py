import numpy as np


class MSE:
    @staticmethod
    def forward(prediction, output):
        return np.sum((prediction - output) ** 2)

    @staticmethod
    def back(prediction, output):
        return 2 * (prediction - output)


class LogLoss:
    @staticmethod
    def forward(prediction, output):
        return (np.dot(output, -np.log(prediction)) + np.dot(1 - output, -np.log((1 - prediction))))[0][0]

    @staticmethod
    def back(prediction, output):
        return np.multiply(output, -1 / prediction) + np.multiply(1 - output, 1 / (1 - prediction))
