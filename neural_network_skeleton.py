# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os
import random

from Neuron.neuron import Neuron
from Layer.layer import Layer


class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # --- PLEASE READ --
        # Use the parameters below to train your feed-forward neural network.

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25

        # This parameter is called the step size, also known as the learning rate (lr).
        # See 18.6.1 in AIMA 3rd edition (page 719).
        # This is the value of α on Line 25 in Figure 18.24.
        # self.lr = 1e-3
        self.lr = 1e-3

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 400

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        # *****************************************************************************************************
        # 
        self.inputDim = input_dim
        self.useHiddenLayer = hidden_layer
        self.layers = np.array([])

        # if self.hiddenLayer:
        #     self.nrOfLayers = 3
        #     inputLayer = Layer(self.inputDim, 25, self.lr)
        #     hiddenLayer = Layer(25, 8, self.lr)
        #     outputLayer = Layer(8, 1, self.lr)
        #     self.layers = np.append(self.layers, [inputLayer, hiddenLayer, outputLayer])
        # else:
        #     self.nrOfLayers = 2
        #     inputLayer = Layer(30, 12, self.lr)
        #     outputLayer = Layer(12, 1, self.lr)
        #     self.layers = np.append(self.layers, [inputLayer, outputLayer])

        if self.useHiddenLayer:
            self.nrOfLayers = 2
            hiddenLayer = Layer(self.inputDim, self.hidden_units, self.lr)
            outputLayer = Layer(self.hidden_units, 1, self.lr)
            self.layers = np.append(self.layers, [hiddenLayer, outputLayer])
        else:
            self.nrOfLayers = 1
            inputLayer = Layer(30, 1, self.lr)
            self.layers = np.append(self.layers, [inputLayer])

    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'data_breast_cancer.p')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network"""
        # TODO: Implement the back-propagation algorithm outlined in Figure 18.24 (page 734) in AIMA 3rd edition.
        # Only parts of the algorithm need to be implemented since we are only going for one hidden layer.

        # Line 6 in Figure 18.24 says "repeat".
        # We are going to repeat self.epochs times as written in the __init()__ method.

        # Line 27 in Figure 18.24 says "return network". Here you do not need to return anything as we are coding
        # the neural network as a class
        
        self.dataSize = self.x_train.shape[0]

        for epoch in range(self.epochs):
            epochError = 0
            for exampleIdx in range(self.dataSize):
                example = self.x_train[exampleIdx]
                label = self.y_train[exampleIdx]
                prediction = self.predict(example)
                error = label - prediction
                epochError += error

                # Feed backwards to calculate deltas
                for layerIdx in range(self.nrOfLayers-1, -1, -1):
                    error = self.layers[layerIdx].feedBackward(error)

                #  Update weights
                for layerIdx in range(self.nrOfLayers-1, -1, -1):
                    self.layers[layerIdx].updateWeights()
                # print("Weights updated")
            successRate = (1-np.abs(epochError[0])/self.dataSize)*100

            
            if epoch % 50 == 0:
                print(f"Completed {epoch}/{self.epochs} epochs with success rate {successRate:.2f}%")

            # print("Total error:\t{}".format(epochError[0]))
            # print("Success rate:\t{}%".format(successRate))
            # if successRate > 99.9:
            #     break

    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """

        out = self.layers[0].feedForward(x)
        for i in range(1,self.nrOfLayers):
            out = self.layers[i].feedForward(out)

        return out


class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')

    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly."""

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')


if __name__ == '__main__':
    unittest.main()
    # nn = NeuralNetwork(30, True)
    # nn.load_data()
    # nn.train()

