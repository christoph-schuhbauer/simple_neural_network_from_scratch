import numpy as np
from math import e
import matplotlib.pyplot as plt
import pickle
import random
import csv

import setup


class CrossEntropy(object):

    @staticmethod
    def fn(self, a, y):

        return np.sum(np.nan_to_num(-y*np.log(a) - (1 - y)*np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):

        return (a - y)


class Network():

    def __init__(self, weights, biases):

        self.weights = weights
        self.biases = biases

        self.activations = []
        self.derivations = []

        self.cost = CrossEntropy()

    def load_network(self):

        with open('network_values.pickle', 'rb') as handle:
            b = pickle.load(handle)

        self.weights = b[0]
        self.biases = b[1]

    def safe_network(self):

        payload = []
        payload.append(self.weights)
        payload.append(self.biases)

        with open('network_values.pickle', 'wb') as handle:
            pickle.dump(payload, handle)


    def predict(self, x):
        self.activations.append(x)

        for w, b in zip(self.weights, self.biases):

            a = np.dot(w, self.activations[-1]) + b
            c = self.calculate_Sigmoid(a)
            self.activations.append(c)

        print(f"predicted result {self.activations[-1]}")
        print(f"np.argmax { np.argmax(self.activations[-1])}")
        prediction = np.argmax(self.activations[-1])
        self.activations.clear()
        return prediction


    #check
    def calculate_Sigmoid(self, x):

        out = 1 / (1 + np.exp(-x))
        return out

    #check
    def calculate_Sigmoid_Derivative(self, x):

        a = 1 - self.calculate_Sigmoid(x)
        b = self.calculate_Sigmoid(x) * a

        return b

    def calc_Sigmoid_ableitung_easy(self, x):

        return x * (1 - x)


    #check
    def feedForward(self, startwerte):


        self.activations.append(startwerte)
        for w, b in zip(self.weights, self.biases):

            a = np.dot(w, self.activations[-1]) + b
            c = self.calculate_Sigmoid(a)

            self.activations.append(c)

        for x in self.activations:

            sol = self.calc_Sigmoid_ableitung_easy(x)
            self.derivations.append(sol)



    def quadratic_cost(self):

        last_error = 0
        return last_error



    #check
    def calculate_error_in_last_layer(self, sollwerte):

        gradient = self.activations[-1] - sollwerte

        ff = self.cost.delta(0, self.activations[-1], sollwerte)

        #last_layer_error = gradient * self.derivations[-1]
        #return last_layer_error
        return ff

    def calculate_error(self, sollwerte):

        all_errors = []
        error_of_last_layer = self.calculate_error_in_last_layer(sollwerte)
        all_errors.append(error_of_last_layer)

        rev_derivations = list(reversed(self.derivations[:-1]))
        rev_weights = list(reversed(self.weights[1:]))

        for w, d in zip(rev_weights, rev_derivations):

            dot = np.dot(w.transpose(), all_errors[-1])
            error = dot * d
            all_errors.append(error)

        errors_from_1st_to_last = list(reversed(all_errors))
        return errors_from_1st_to_last


    def calculate_deltas(self, target_value):

        errors = self.calculate_error(target_value)
        delta_weights = []
        delta_biases = []

        for ee, a in zip(errors, self.activations[:-1]):

            delta = np.outer(ee, a)
            delta_bias = np.outer(ee, 1)
            delta_biases.append(delta_bias)
            delta_weights.append(delta)

        return delta_weights, delta_biases


    def backprob(self, start_value, target_value):

        self.feedForward(start_value)

        delta_w, delta_b = self.calculate_deltas(target_value)

        self.activations.clear()
        self.derivations.clear()

        return delta_w, delta_b


    def update_mini_batch(self, minibatch, eta):

        adding_deltas_weights = [np.zeros(w.shape) for w in self.weights]
        adding_deltas_biases = [np.zeros(b.shape) for b in self.biases]

        batch_size = len(minibatch)
        for x, y in minibatch:

            delta_w, delta_b = self.backprob(x, y)

            for aw, dw in zip(adding_deltas_weights, delta_w):

                aw += dw

            for ab, db in zip(adding_deltas_biases, delta_b):

                ab = ab.reshape(db.shape)
                ab += db

        lamda = 5.0
        new_weights = [ (1-eta*(lamda/50000))* og_w - ( (eta/batch_size) * dw) for og_w, dw in zip(self.weights, adding_deltas_weights) ]
        new_biases = [og_b - ( (eta/batch_size) * db) for og_b, db in zip(self.biases, adding_deltas_biases) ]

        self.weights = new_weights
        self.biases = new_biases


    def create_mini_batches(self,data_set, epochs, batch_size, test_data=None):

        mini_batches = []
        n = len(data_set)
        count_epochs = 0
        if test_data is not None:
            self.evaluate(test_data)
        for i in range(epochs):
            random.shuffle(data_set)
            mini_batches = [data_set[k: k + batch_size] for k in range(0, n, batch_size)]
            count_epochs += 1
            counter = 0

            for mini_batch in mini_batches:
                counter += 1
                self.update_mini_batch(mini_batch, 0.1)

            if test_data is not None:
                self.evaluate(test_data)

            print(f"FINISHED number {i + 1} of {epochs} EPOCHS")


    def evaluate(self, test_data):
        count = 0
        for x, y in test_data:
            self.feedForward(x)
            result = self.activations[-1]
            self.activations.clear()
            if np.argmax(result) == np.argmax(y):
                count += 1

        print(f"{count} / {len(test_data)}")


def create_network(data):

    # data in form of [training-images, test-images, training-labels(one-hot), test-labels(one-hot)]

    train_imgs = data[0]
    test_imgs = data[1]
    train_labels_one_hot = data[2]
    test_labels_one_hot = data[3]

    layers = [784, 30, 10]
    sizes_1 = layers[:-1]
    sizes_2 = layers[1:]
    matrixes_sizes = tuple(zip(sizes_1, sizes_2))

    training_data = []
    test_data = []

    # create trainings-data

    for images, one_hot_labels in zip(train_imgs[:], train_labels_one_hot[:]):
        point = (images, one_hot_labels)
        training_data.append(point)


    # create test-data

    for images, one_hot_labels in zip(test_imgs[:], test_labels_one_hot[:]):
        point = (images, one_hot_labels)
        test_data.append(point)

    weights = [np.round(np.random.randn(y, x), 1) for x, y in matrixes_sizes]
    biases = [np.round(np.random.randn(x), 1) for x in sizes_2]

    network = Network(weights, biases)
    network.create_mini_batches(training_data, 30, 10, test_data)

    if false:
        network.safe_network()
        print("saved")

    return network



def run():
    # setup.prep_digital()
    # data = setup.load_digital_numbers()
    data = setup.load_local_mnist()

    for x in data:
        print(len(x))
        print(type(x))
        print(x[0])

    img = data[1][0]
    img = setup.recreate_img(img)
    img = np.reshape(img, (28, 28))
    plt.imshow(img, interpolation='nearest')
    plt.gray()
    print(data[3][0])
    plt.show()

    network = create_network(data)


if __name__ == '__main__':

    run()