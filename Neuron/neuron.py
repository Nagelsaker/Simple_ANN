import numpy as np
import random

class Neuron:
    def __init__(self, nrOfInputs, learningRate):
        '''
        Neuron class, representing the input weights to a neuron, and the calculated output
        In:
            nrOfInputs: (int) Nr of neurons in the previous layer
            learningRate: (float) Learning rate for the network 
        '''
        self.W = np.array([random.uniform(0,1/np.sqrt(nrOfInputs)) for i in range(nrOfInputs+1)])
        self.Xin = np.zeros(nrOfInputs+1)
        self.Xin[0] = -1 # Bias

        self.X = 0
        self.preActivatedX = 0

        self.nrOfInputs = nrOfInputs
        self.delta = 0
        self.W_prev_delta = 0
        self.momentum = 9e-1
        self.lr = learningRate


    def setInputVector(self, inputVector):
        '''
        In:
            inputVector: (nrOfInputs dim array)
        '''
        if inputVector.size == self.nrOfInputs:
            self.Xin[1:] = inputVector
        else:
            raise Exception("Error! Input with dimension {} not the same as specified nr of inputs {}".format(self.Xin.size, self.nrOfInputs))

    def activationFunc(self, value):
        '''
        Activation function implemented using the Sigmoid formula
        '''
        sigmoid = 1 / (1 + np.exp(-value))
        return sigmoid

    def diffActFunc(self, value):
        '''
        Differentiation of the Sigmoid activation function
        '''
        diffSigmoid = self.activationFunc(value) * (1 - self.activationFunc(value))
        return diffSigmoid
    
    def calculateOutput(self):
        '''
        Calculate output from this neuron
        Out:
            (int) output value
        '''
        self.preActivatedX = self.W.T@self.Xin
        self.X = self.activationFunc(self.preActivatedX)
        return self.X

    def updateDelta(self, error):
        '''
        Updates delta value for this neuron
        Input:
            error: (float) The error this neuron contributes to the next layer
        '''
        self.delta = self.diffActFunc(self.preActivatedX) * error

    def updateWeights(self):
        '''
        Updates the input weights for this neuron
        '''
        self.W_delta = self.lr * self.delta * self.Xin + self.momentum * self.W_prev_delta
        self.W += self.W_delta
        self.W_prev_delta = self.W_delta

    def getErrorContrib(self, index):
        '''
        Returns the error that the previous layer, neuron nr "index", contributed to this layer neuron's error.
        '''
        index += 1 # Offset due to bias at pos 0
        return self.W[index] * self.delta