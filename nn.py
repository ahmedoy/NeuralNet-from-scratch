import numpy as np
from activation_functions import *


class NN:
    def __init__(self, input_size, output_size, output_activation, loss_function):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_gen = [(input_size, 'none')]
        self.layers = []
        self.layer_count = 0
        self.output_activation = output_activation
        self.loss_function = loss_function

    def add_layer(self, layer_size, activation):
        self.layer_gen.append((layer_size, activation))

    def generate_layers(self):
        self.layer_gen.append((self.output_size, self.output_activation))
        for i in range(len(self.layer_gen) - 1):
            self.layers.append(
                Layer(input_size=self.layer_gen[i][0], output_size=self.layer_gen[i + 1][0], layer_idx=i + 1,
                      activation_name=self.layer_gen[i + 1][1]))
            self.layer_count += 1

    def show_layers(self, verbose=True):
        for layer in self.layers:
            layer.show(verbose)

    def forward(self, input_vector, verbose=False):
        output = input_vector
        for layer in self.layers:
            if verbose:
                output_preact = layer.process_no_activation(output)
                output = layer.process(output)
                print(f'\nOutput Before activation {layer.idx}:\n', output_preact, '\n')
                print(f'\nOutput After activation{layer.idx}:\n', output, '\n')

            else:
                output = layer.process(output)

        return output

    def forward_with_activations(self, input_vector):
        # returns output along with all intermediate activations. (Used in the "back" method)
        output = input_vector
        activations = [output]
        for layer in self.layers:
            output = layer.process(output)
            activations.append(output)
        return activations

    def get_cost(self, input_vector, output):
        return self.loss_function.forward(self.forward(input_vector), output)

    def back(self, input_vector, output):
        activations = self.forward_with_activations(input_vector)
        output_activation = activations[-1]
        output_activation_derivative = self.loss_function.back(output_activation, output).reshape(
            output_activation.shape)

        for layer in self.layers[::-1]:  # iterate over all layers from last to first
            input_activation = activations[-2]
            bias_gradient = np.multiply(layer.activation.back(output_activation), output_activation_derivative)
            weight_gradient = np.matmul(bias_gradient, np.transpose(input_activation))
            layer.add_gradients(weight_gradient, bias_gradient)
            activations.pop()  # remove last layer in list
            output_activation = input_activation
            output_activation_derivative = np.matmul(np.transpose(layer.weights), bias_gradient)

    def back_numerical(self, input_vector, output):
        dx = 0.00000001
        for layer in self.layers:
            weight_gradient = np.zeros(layer.weights.shape, dtype=layer.weights.dtype)
            bias_gradient = np.zeros(layer.biases.shape, dtype=layer.biases.dtype)

            for i in range(layer.weights.shape[0]):  # rows
                for j in range(layer.weights.shape[1]):  # columns
                    temp = layer.weights[i][j]

                    layer.weights[i][j] = temp + dx
                    cost1 = self.get_cost(input_vector, output)

                    layer.weights[i][j] = temp - dx
                    cost2 = self.get_cost(input_vector, output)

                    layer.weights[i][j] = temp

                    weight_gradient[i][j] = (cost1 - cost2) / (2 * dx)

            for i in range(layer.biases.shape[0]):
                temp = layer.biases[i][0]

                layer.biases[i][0] = temp + dx
                cost1 = self.get_cost(input_vector, output)

                layer.biases[i][0] = temp - dx
                cost2 = self.get_cost(input_vector, output)

                layer.biases[i][0] = temp

                bias_gradient[i][0] = (cost1 - cost2) / (2 * dx)

            layer.add_gradients(weight_gradient, bias_gradient)

    def train_batch(self, batch_obj, learning_rate, momentum=0, show_performance=True, save_cost=True):
        total_cost = 0

        for sample in batch_obj.train_batch:
            self.back(sample[0], sample[1])  # input and output respectively
            total_cost += self.get_cost(sample[0], sample[1])

        for layer in self.layers:
            layer.update(learning_rate, momentum)

        if show_performance:
            print(f"Train Cost = {total_cost / batch_obj.train_batch_size}")

        if save_cost:
            if not hasattr(self, 'train_costs'):
                self.train_costs = []
            self.train_costs.append(total_cost / batch_obj.train_batch_size)

    def test_batch(self, batch_obj, show_performance=True, save_cost=True):
        total_cost = 0

        for sample in batch_obj.test_batch:
            total_cost += self.get_cost(sample[0], sample[1])

        if show_performance:
            print(f"Test Cost = {total_cost / batch_obj.test_batch_size}")

        if save_cost:
            if not hasattr(self, 'test_costs'):
                self.test_costs = []
            self.test_costs.append(total_cost / batch_obj.test_batch_size)


class Layer:

    def __init__(self, input_size, output_size, layer_idx, activation_name):
        self.weights = np.random.normal(size=(output_size, input_size))
        self.biases = np.random.normal(size=(output_size, 1))
        self.idx = layer_idx
        self.activation_name = activation_name
        self.activation = activation_functions[self.activation_name]
        self.init_gradients()

    def process(self, layer_input):
        return self.activation.forward(np.matmul(self.weights, layer_input) + self.biases)

    def process_no_activation(self, layer_input):
        return np.matmul(self.weights, layer_input) + self.biases

    def init_gradients(self, prev=True):
        self.weight_gradient = np.zeros(self.weights.shape, dtype=self.weights.dtype)
        self.bias_gradient = np.zeros(self.biases.shape, dtype=self.biases.dtype)
        self.gradient_count = 0  # used to get the average of the gradients
        if prev:
            self.prev_weight_gradients = np.zeros(self.weights.shape, dtype=self.weights.dtype)
            self.prev_bias_gradients = np.zeros(self.biases.shape, dtype=self.biases.dtype)

    def add_gradients(self, weight_gradient, bias_gradient):
        if (weight_gradient.shape != self.weights.shape or bias_gradient.shape != self.biases.shape):
            raise ValueError("Gradient Dimension Error")

        self.weight_gradient += weight_gradient
        self.bias_gradient += bias_gradient
        self.gradient_count += 1

    def update(self, learning_rate, momentum):

        self.weight_gradient = learning_rate * self.weight_gradient / self.gradient_count + momentum * self.prev_weight_gradients
        self.bias_gradient = learning_rate * self.bias_gradient / self.gradient_count + momentum * self.prev_bias_gradients
        self.weights -= self.weight_gradient
        self.biases -= self.bias_gradient
        self.prev_weight_gradients = np.copy(self.weight_gradient)
        self.prev_bias_gradients = np.copy(self.bias_gradient)
        self.init_gradients(prev=False)

    def show(self, verbose=False, show_gradients=False):
        print(f"Layer idx:{self.idx}, Activation: {self.activation_name}, Shape: {self.weights.shape}")
        if verbose:
            print("Weights:")
            print(self.weights)
            print("\n")
            print("Biases: ")
            print(self.biases)
            print("\n")
        if show_gradients:
            print("Weight Gradient:")
            print(self.weight_gradient)
            print("\n")
            print("Bias Gradient:")
            print(self.bias_gradient)

        print("\n\n")
