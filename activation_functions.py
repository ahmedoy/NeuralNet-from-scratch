import numpy as np


class Relu:
    @staticmethod
    def forward(array_x):
        zeros_arr = np.zeros(array_x.shape, dtype=array_x.dtype)
        return np.maximum(array_x, zeros_arr)

    @staticmethod
    def back(output):
        output[output > 0] = 1
        return output


class Sigmoid:

    @staticmethod
    def forward(array_x):
        return 1 / (1 + np.exp(-array_x))

    @staticmethod
    def back(output):
        return output * (1 - output)


class Tanh:
    @staticmethod
    def forward(array_x):
        return np.tanh(array_x)

    @staticmethod
    def back(output):
        return 1 - (output ** 2)


class NoActivation:
    @staticmethod
    def forward(array_x):
        return array_x

    @staticmethod
    def back(output):
        return np.ones(output.shape, dtype=output.dtype)


activation_functions = {"relu": Relu, "sigmoid": Sigmoid, "tanh": Tanh, "none": NoActivation}
