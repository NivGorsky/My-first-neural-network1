import numpy as np
import random
import mnist_loader

class NeuralNetwork:
    def __init__(self, network_structure): #the network structure already contains the bias
        self._network_structure = []
        self._init_network_structure(network_structure)
        self.number_of_layers = len(self._network_structure)
        self.weights_by_layer = self._init_weights()
        self.biases_by_layer = self._init_biases()
        self.epochs = None
        self.mini_batch_size = None
        self.eta = None
        self.training_data = []
        self.test_data = []

    def __predict(self, x):
        for b, w in zip(self.biases_by_layer, self.weights_by_layer): # w is matrix, x is vector -> product gives a vector
            x = self._sigmoid(np.dot(w, x) + b)

        return x

    def _init_network_structure(self, network_structure):
        self._network_structure = network_structure

    def _init_biases(self):
        # biases = []
        # for layer_size in range(1, len(self._network_structure)):
        #     biases.append(np.random.rand(layer_size, 1))


        return [np.random.randn(y, 1) for y in self._network_structure[1:]]

    def _init_weights(self):
        weights = [np.random.randn(y , x)
                   for x, y in zip(self._network_structure[:-1], self._network_structure[1:])]

        # weights = []
        # index_of_last_layer_in_network = len(self._network_structure) - 1
        # index_of_one_before_last_layer_in_network = index_of_last_layer_in_network - 1
        # range_of_layer_spaces_from_first_layer_to_one_before_last_layer = range(0, index_of_one_before_last_layer_in_network)
        #
        # for i in range_of_layer_spaces_from_first_layer_to_one_before_last_layer:
        #     index_of_left_side_of_layer = i
        #     index_of_right_side_of_layer = i+1
        #
        #     number_of_neurons_in_left_side_of_current_layers_space = self._network_structure[index_of_left_side_of_layer]
        #     number_of_neurons_in_right_side_of_current_layer_space = self._network_structure[index_of_right_side_of_layer]
        #     weights_for_current_layers_space_matrix = np.random.rand(number_of_neurons_in_right_side_of_current_layer_space , number_of_neurons_in_left_side_of_current_layers_space)
        #     weights.append(weights_for_current_layers_space_matrix)
        #
        # number_of_neurons_in_right_side_of_current_layers_space = self._network_structure[index_of_last_layer_in_network]
        # number_of_neurons_in_left_side_of_current_layers_space = self._network_structure[index_of_one_before_last_layer_in_network]
        # weights_for_last_layers_space_matrix = np.random.rand(number_of_neurons_in_right_side_of_current_layers_space, number_of_neurons_in_left_side_of_current_layers_space)
        # weights.append(weights_for_last_layers_space_matrix)

        return weights

    def _convert_x_y_to_data(self, x, y):
        tuple_of_x_y = zip(x , y)
        data = []

        for xi, yi in tuple_of_x_y:
            data.append((xi, yi))

        return data

    def __fit__(self, x, y, x_test, y_test): # SGD already receives the trainning data
        #x is a list of ndarray
        #y is a list of ndarray
        self._training_data = self._convert_x_y_to_data(x, y)

        for epoch in range(0, self.epochs):
            random.shuffle(self.training_data)
            mini_batches = self._create_mini_batches_from_training_data()

            for i, single_mini_batch in enumerate(mini_batches):
                self._learn_by_mini_batch(single_mini_batch)

            print("Epoch {} complete".format(epoch))
            print("Epoch {} : {} / {}".format(epoch, self.__score__(x_test, y_test), len(x_test)))

    def _learn_by_mini_batch(self, mini_batch):
        nabla_w = [np.zeros(w.shape) for w in self.weights_by_layer]
        nabla_b = [np.zeros(b.shape) for b in self.biases_by_layer]

        for x, y in mini_batch: #x is the array of data, y is the result
            delta_nabla_w, delta_nabla_b = self._feed_forward_and_back_propagation(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights_by_layer = [w - self.eta * nw
                                     for w, nw in zip(self.weights_by_layer, nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            self.biases_by_layer = [b - self.eta * nb
                       for b, nb in zip(self.biases_by_layer, nabla_b)]

    def _add_bias_to_array(self, np_array):

        return np.insert(np_array, 0, 1, axis=0)

    def _feed_forward_and_back_propagation(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights_by_layer]
        nabla_b = [np.zeros(b.shape) for b in self.biases_by_layer]

        current_activation = x #the first activation is the actual data values x1,x2...xN
        activations = [x]  # list to store all the activations, layer by layer
        all_layers_dot_products = [] # this list holds the dot products of all the neurons in a single layer

        # what im doing is to add the bias every iteration for the current activation..(adding 1 in the index - of the activation array)
        #feedforward
        for biases_of_all_neurons_in_current_layer, weights_of_all_neurons_in_current_layer in zip(self.biases_by_layer, self.weights_by_layer):
            dot_products_for_all_neurons_in_current_layer = np.dot(weights_of_all_neurons_in_current_layer, current_activation) + biases_of_all_neurons_in_current_layer
            all_layers_dot_products.append(dot_products_for_all_neurons_in_current_layer)
            current_activation = self._sigmoid(dot_products_for_all_neurons_in_current_layer)
            activations.append(current_activation)

        #back propagation
        # the last layer partial derivatives equation
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(all_layers_dot_products[-1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta

        for i in range(2, self.number_of_layers):
            dot_products_for_all_neurons_in_current_layer = all_layers_dot_products[-i]
            weights_by_current_layer = self.weights_by_layer[-i+1]
            delta = np.dot(weights_by_current_layer.transpose(), delta) * self.sigmoid_prime(dot_products_for_all_neurons_in_current_layer)
            nabla_w[-i] = np.dot(delta, activations[-i - 1].transpose())
            nabla_b[-i] = delta

        return nabla_w, nabla_b

    def _is_need_to_add_activation_to_vector(self, arr, index):
        return index < len(arr) - 1

    def _create_mini_batches_from_training_data(self):
        n = len(self._training_data)
        mini_batches = [
            self._training_data[k:k + self.mini_batch_size]
            for k in range(0, n, self.mini_batch_size)]

        return mini_batches

    def __score__(self, x, y):
        self._test_data = self._convert_x_y_to_data(x, y)

        test_results = [(np.argmax(self.__predict(x)), y)
                        for (x, y) in self._test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def _sigmoid(self, z):

        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def set_network_parameters(self, epochs, eta, mini_batch_size):
        self.epochs = epochs
        self.eta = eta
        self.mini_batch_size = mini_batch_size

def convert_to_x_y(training_data_to_convert):
    x = []
    y = []

    for xi, yi in training_data_to_convert:
        x.append(xi)
        y.append(yi)

    return x,y

if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    x, y = convert_to_x_y(training_data)
    x_test, y_test = convert_to_x_y(test_data)

    network = NeuralNetwork([784, 30, 10])
    network.set_network_parameters(30, 0.006, 10)
    network.__fit__(x, y, x_test, y_test)
    network.__score__(x_test, y_test)



