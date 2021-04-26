import numpy as np

from Neuron.neuron import Neuron

class Layer:
    def __init__(self, inputDimension, outputDimension, learningRate):
        '''
        Layer class, representing a vector of neurons.
        '''
        self.inputDim = inputDimension
        self.outputDim = outputDimension
        self.lr = learningRate

        self.neurons = np.array([Neuron(self.inputDim, self.lr) for i in range(self.outputDim)])
        self.output = np.zeros(self.outputDim)


    def feedForward(self, inputVector):
        '''
        Feeds an input vector through the neurons
        In:
            inputVector (inputDim dim array)
        Out:
            (outputDim dim array)
        '''
        for i in range(self.outputDim):
            self.neurons[i].setInputVector(inputVector)
            out = self.neurons[i].calculateOutput()
            self.output[i] = out

        return self.output

    def feedBackward(self, error):
        '''
        Calculates the an error vector for this layer
        In:
            error: (outpuTdim dim array) containing errors for the next layer
        Out:
            errorContribution: (inputDim dim array) containing errors for this layer
        '''
        for i in range(self.outputDim):
            self.neurons[i].updateDelta(error[i])

        errorContributon = np.array([self.getErrorContrib(idx) for idx in range(self.inputDim)])
        return errorContributon


    def getErrorContrib(self, index):
        '''
        Get the error contribution that the input neuron nr index had on the layer
        '''
        error = np.array([self.neurons[i].getErrorContrib(index) for i in range(self.outputDim)]).sum()
        return error

    def updateWeights(self):
        for i in range(self.outputDim):
            self.neurons[i].updateWeights()