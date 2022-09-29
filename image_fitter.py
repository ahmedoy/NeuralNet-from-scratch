import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import batch
import nn
import loss_functions

IMG_PATH = "img.png"
NN_INPUT_SIZE = 2
NN_OUTPUT_SIZE = 1

img = np.array(cv2.imread(IMG_PATH, 0))


def sample_gen():
    row = random.randint(0, img.shape[0] - 1)
    column = random.randint(0, img.shape[1] - 1)
    sample = (np.array([[row / img.shape[0], column / img.shape[1]]]).reshape((NN_INPUT_SIZE, 1)),
              np.array([[round(img[row][column] / 255)]]).reshape(NN_OUTPUT_SIZE, 1))
    return sample


def graph(graph_idx):
    img_rows = img.shape[0]
    img_cols = img.shape[1]
    img_ratio = img_cols / img_rows

    gen_img_rows = 100
    gen_img_cols = int(gen_img_rows * img_ratio)
    gen_img = np.zeros((gen_img_rows, gen_img_cols), dtype=int)
    for i in range(gen_img_rows):
        for j in range(gen_img_cols):
            x = np.array([[i / gen_img_rows, j / gen_img_cols]]).reshape(NN_INPUT_SIZE, 1)
            if net.forward(x)[0][0] > 0.5:
                gen_img[i][j] = 1

    colors = ['black', 'white']
    cmap = LinearSegmentedColormap.from_list('', colors, len(colors))
    plt.imshow(gen_img, cmap=cmap, vmin=0, vmax=len(colors) - 1, alpha=1)

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
    for i in range(len(train_y) - averaging_width):
        avg_train_y.append(sum([train_y[i + w] for w in range(averaging_width)]) / averaging_width)
        avg_test_y.append(sum([test_y[i + w] for w in range(averaging_width)]) / averaging_width)

    x = [i for i in range(len(avg_train_y))]
    plt.plot(x, avg_train_y, c='r', label='Train')
    plt.plot(x, avg_test_y, c='b', label='Test')
    plt.legend()
    plt.savefig(f'graphs\Graph {graph_idx}.png')
    plt.clf()


net = nn.NN(input_size=NN_INPUT_SIZE, output_size=NN_OUTPUT_SIZE, output_activation="sigmoid",
            loss_function=loss_functions.LogLoss)
net.add_layer(20, "tanh")
net.add_layer(20, "tanh")
net.add_layer(20, "tanh")
net.add_layer(20, "tanh")
net.add_layer(20, "tanh")
net.add_layer(20, "tanh")
net.generate_layers()

batch = batch.Batch(train_batch_size=100, test_batch_size=500, sample_gen_function=sample_gen)
batch.gen_test_batch()  # generating test batch once

learning_rate = 0.2
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
