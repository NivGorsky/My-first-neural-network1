import numpy as np
import random
import mnist_loader

class NeuralNetwork:
    def __init__(self, network_structure):
        self._network_structure = []
        self._init_network_structure(network_structure)
        self._number_of_layers = len(self._network_structure)
        self._weights_by_spaces_between_layers = self.init_weights()
        self._number_of_epochs = None
        self._mini_batch_size = None
        self._eta = None
        self._training_data = []
        self._test_data = []
        self._index_of_last_layer = -1
        self._index_of_layer_before_last_layer = -1

    def _init_network_structure(self, network_structure):

        for layer in range(0, len(network_structure) - 1):
            self._network_structure.append(network_structure[layer] + 1)

        self._network_structure.append(network_structure[-1])
        self._index_of_last_layer = len(self._network_structure) - 1
        self._index_of_layer_before_last_layer = self._index_of_last_layer - 1

    def init_weights(self):
        weights = []
        index_of_last_layer_in_network = len(self._network_structure) - 1
        range_of_layer_spaces_from_first_layer_to_last_layer = range(0, index_of_last_layer_in_network)

        for i in range_of_layer_spaces_from_first_layer_to_last_layer:
            index_of_left_side_of_layer = i
            index_of_right_side_of_layer = i+1
            number_of_neurons_in_left_side_of_current_layers_space = self._network_structure[index_of_left_side_of_layer]
            number_of_neurons_in_right_side_of_current_layer_space = self._network_structure[index_of_right_side_of_layer]
            weights.append(np.random.randn(number_of_neurons_in_right_side_of_current_layer_space , number_of_neurons_in_left_side_of_current_layers_space))

        for i in range(0, len(weights) - 1):
            weights[i] = np.delete(weights[i], 0, 0)

        return weights
    
    def predict(self, activation):
        activation = self._add_bias_to_array(activation)
        for right_side_space_layer_index in range(1, len(self._network_structure)): # w is matrix, x is vector -> product gives a vector

            left_side_space_layer_index = right_side_space_layer_index - 1
            w = self._weights_by_spaces_between_layers[left_side_space_layer_index]
            activation = self._sigmoid(np.dot(w, activation))

            if right_side_space_layer_index < len(self._network_structure) - 1:
                activation = self._add_bias_to_array(activation)

        return activation

    def _convert_x_y_to_data(self, x, y):
        tuple_of_x_y = zip(x , y)
        training_data = []

        for xi, yi in tuple_of_x_y:
            training_data.append((xi, yi))

        return training_data

    def fit(self, x, y, x_test = None, y_test = None): # SGD already receives the trainning data
        self._training_data = self._convert_x_y_to_data(x, y)

        for epoch in range(0, self._number_of_epochs):
            random.shuffle(self._training_data)
            stochastic_gradient_decent_mini_batches = self._create_mini_batches_from_training_data()

            for i, single_mini_batch in enumerate(stochastic_gradient_decent_mini_batches):
                self._learn_by_mini_batch(single_mini_batch)

            if x_test is not None and y_test is not None:
                print("Results test of epoch number {} : {} / {}".format(epoch, self.score(x_test, y_test), len(x_test)))

    def _learn_by_mini_batch(self, mini_batch):
        weights_delta_for_current_mini_batch = [np.zeros(w.shape) for w in self._weights_by_spaces_between_layers]

        for sample_array, actual_result_array in mini_batch:
            weights_delta_for_current_xi = self._feed_forward_and_back_propagation(sample_array, actual_result_array)
            weights_delta_for_current_mini_batch = self._get_weights_delta_for_current_mini_batch(weights_delta_for_current_mini_batch, weights_delta_for_current_xi)
            self._weights_by_spaces_between_layers = [old_weights - self._eta * new_weights
                                                      for old_weights, new_weights in zip(self._weights_by_spaces_between_layers, weights_delta_for_current_mini_batch)]

    def _get_weights_delta_for_current_mini_batch(self, weights_delta_for_current_mini_batch, weights_delta_for_current_xi):

        result = []
        for old_weight_delta, new_weight_delta in zip(weights_delta_for_current_mini_batch, weights_delta_for_current_xi):
            result.append(old_weight_delta + new_weight_delta)

        return result

    def _add_bias_to_array(self, np_array):

        return np.insert(np_array, 0, 1, axis=0)

    def _feed_forward_and_back_propagation(self, x_data, y_results):
        x_data = self._add_bias_to_array(x_data)
        new_weights_for_current_feedforward_and_back_propagation = [np.zeros(w.shape) for w in self._weights_by_spaces_between_layers]
        current_activation = x_data #the first activation is the actual data values x1,x2...xN
        activations = [x_data]
        all_layers_dot_products = []

        # adding the bias every iteration for the current activation...(adding 1 in the index 0 of the activation array)
        for i, weights_of_all_neurons_in_current_layer in enumerate(self._weights_by_spaces_between_layers):
            dot_products_for_all_neurons_in_current_layer = np.dot(weights_of_all_neurons_in_current_layer, current_activation)
            all_layers_dot_products.append(dot_products_for_all_neurons_in_current_layer)
            current_activation = self._sigmoid(dot_products_for_all_neurons_in_current_layer)

            if self._is_need_to_add_activation_to_vector(self._weights_by_spaces_between_layers, i):
                current_activation = self._add_bias_to_array(current_activation)

            activations.append(current_activation)

        weights_delta = self._derivative_of_activation_function_of_last_layer(activations[-1], y_results) * self._sigmoid_prime(all_layers_dot_products[-1])
        new_weights_for_current_feedforward_and_back_propagation[-1] = np.dot(weights_delta, activations[-2].transpose())

        for i in range(2, self._number_of_layers):
            dot_products_for_all_neurons_in_current_layer = all_layers_dot_products[-i]
            weights_by_current_layer = self._weights_by_spaces_between_layers[-i + 1]
            weights_by_current_layer_without_bias_weights = np.delete(weights_by_current_layer, 0, 1)
            weights_delta = np.dot(weights_by_current_layer_without_bias_weights.transpose(), weights_delta) * self._sigmoid_prime(dot_products_for_all_neurons_in_current_layer)
            new_weights_for_current_feedforward_and_back_propagation[-i] = np.dot(weights_delta, activations[-i - 1].transpose())

        return new_weights_for_current_feedforward_and_back_propagation

    def _is_need_to_add_activation_to_vector(self, arr, index):
        return index < len(arr) - 1

    def _create_mini_batches_from_training_data(self):
        groups_of_inputs = []
        training_data_size = len(self._training_data)
        for i in range(0, training_data_size, self._mini_batch_size):
            groups_of_inputs.append(self._training_data[i: i + self._mini_batch_size])

        return groups_of_inputs

    def score(self, x, y):
        self._test_data = self._convert_x_y_to_data(x, y)
        sum = 0

        for xi , yi in self._test_data:
            prediction = self.predict(xi)
            max_index = np.argmax(prediction)

            if max_index == yi:
                sum += 1

        return sum

        # test_results = [(np.argmax(self.__predict(x)), y)
        #                 for (x, y) in self._test_data]
        #
        # return sum(int(x == y) for (x, y) in test_results)

    def _sigmoid(self, z):

        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_prime(self, z):

        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _derivative_of_activation_function_of_last_layer(self, output_activations, y):

        return (output_activations-y)

    def set_network_parameters(self, epochs, eta, mini_batch_size):
        self._number_of_epochs = epochs
        self._eta = eta
        self._mini_batch_size = mini_batch_size


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
    network.set_network_parameters(30, 0.012, 10)
    network.fit(x, y, x_test, y_test)
    network.score(x_test, y_test)



