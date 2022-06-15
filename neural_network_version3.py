# In this model, I have increased the learning rate, and also increased the number of epochs
from neural_network import NeuralNetwork
import mnist_loader
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
    network.set_network_parameters(6, 0.014, 10)
    network.fit(x, y)
    number_of_examples = len(x_test)
    number_of_examples_were_predicted_incorrectly = network.score(x_test, y_test)

    print('model perdicted {} \ {} examples incorrectly'.format(number_of_examples_were_predicted_incorrectly, number_of_examples))




