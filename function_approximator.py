import numpy as np
from matplotlib import pyplot as plt
import batch
import nn
import loss_functions

NN_INPUT_SIZE = 1
NN_OUTPUT_SIZE = 1
X_START, X_END = -2, 3  # This is the range of inputs for the single variable function we are trying to approximate


def function(x):  # This is the single variable function we are trying to approximate
    k = 1
    if x ** 2 < 1:
        k = 5
    return (x ** 4 - x ** 3 - x ** 2) * k


def sample_gen():
    x = np.random.uniform(low=X_START, high=X_END, size=1)
    y = np.array(function(x))
    sample = [x.reshape(NN_INPUT_SIZE, 1), y.reshape(NN_OUTPUT_SIZE, 1)]
    return sample


def graph(graph_idx):  # test batch must be sorted according to input before using this function
    x = [batch.test_batch[i][0][0] for i in range(len(batch.test_batch))]
    y = [batch.test_batch[i][1][0] for i in range(len(batch.test_batch))]
    predictions = [net.forward(x[i].reshape(NN_INPUT_SIZE, 1))[0] for i in range(len(batch.test_batch))]

    plt.plot(x, y, c='b', label='true function')
    plt.plot(x, predictions, c='r', label='neural net')
    plt.legend()
    plt.title(f"Fig {graph_idx}")
    plt.savefig(f'graphs\Fig {graph_idx}.png')
    plt.clf()


def plot(averaging_width, graph_idx):
    train_y = net.train_costs
    test_y = net.test_costs
    avg_train_y = []
    avg_test_y = []
    if averaging_width > len(train_y):
        return
    start = 5 * (graph_idx-1)
    for i in range(start, len(train_y) - averaging_width):
        avg_train_y.append(sum([train_y[i + w] for w in range(averaging_width)]) / averaging_width)
        avg_test_y.append(sum([test_y[i + w] for w in range(averaging_width)]) / averaging_width)

    x = [i for i in range(len(avg_train_y))]
    plt.plot(x, avg_train_y, c='r', label='Train')
    plt.plot(x, avg_test_y, c='b', label='Test')
    plt.legend()
    plt.savefig(f'graphs\Graph {graph_idx}.png')
    plt.clf()


net = nn.NN(input_size=NN_INPUT_SIZE, output_size=NN_OUTPUT_SIZE, output_activation='none',
            loss_function=loss_functions.MSE)
net.add_layer(20, "relu")
net.add_layer(20, "relu")
net.add_layer(20, "relu")
net.add_layer(20, "relu")
net.generate_layers()

batch = batch.Batch(train_batch_size=100, test_batch_size=500, sample_gen_function=sample_gen)
batch.gen_test_batch()  # generating test batch once
batch.test_batch.sort(
    key=lambda sample: sample[0])  # sorting the test batch according to input in order to easily graph the batch

learning_rate = 0.00001
epochs = 10000
momentum = 0.1

for e in range(epochs):
    batch.gen_train_batch()  # generating new training batch at every epoch
    print("Epoch: ", e)
    net.train_batch(batch, learning_rate=learning_rate, momentum=momentum)
    net.test_batch(batch)
    print()

    if e % 100 == 0:
        graph(graph_idx=e // 100)
        plot(averaging_width=50, graph_idx=e // 100)
