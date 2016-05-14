import numpy as np

from util.activation_functions import Activation


class LogisticLayer():
    """
    A layer of neural

    Parameters
    ----------
    n_in: int: number of units from the previous layer (or input data)
    n_out: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    is_classifier_layer: bool:  to do classification or regression

    Attributes
    ----------
    n_in : positive int:
        number of units from the previous layer
    n_out : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activation_string : string
        the name of the activation function
    is_classifier_layer: bool
        to do classification or regression
    deltas : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, n_in, n_out, weights=None,
                 activation='sigmoid', is_classifier_layer=False):

        # Get activation function from string
        self.activation_string = activation
        self.activation = Activation.get_activation(self.activation_string)
	self.derivative = Activation.get_derivative(self.activation_string)

        self.n_in = n_in
        self.n_out = n_out

	self.learningRate = 0.5

        self.inp = np.ndarray((1,n_in+1))
        self.inp[0,0] = 1 #bias
        self.outp = np.ndarray((1,n_out))
        self.deltas = np.zeros((1,n_out))
	# self.inp[0] := 1 -> weight[0] = bias
	# self.inp[1-n_in] -> input vector

        # You can have better initialization here
        if weights is None:
            self.weights = np.random.rand(n_in+1, n_out)/10
        else:
            self.weights = weights
	
	
        self.is_classifier_layer = is_classifier_layer

        # Some handy properties of the layers
        self.size = self.n_out
        self.shape = self.weights.shape

    def forward(self, inp):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        inp : ndarray
            a numpy array (1,n_in + 1) containing the input of the layer

        Change outp
        -------
        outp: ndarray
            a numpy array (1,n_out) containing the output of the layer
        """

        # Here you have to implement the forward pass

	# store input and add 1 in beginning for bias
	self.inp[0,1:] = inp
	self.inp[0,0] = 1.

	#use fire function
	#matrix mult.: (1xn_int+1) x (n_int+1,n_out) -> (1,n_out)
	self.outp =_fire(self.inp)

	pass

    def computeDerivative(self, nextDerivatives, nextWeights):
        """
        Compute the derivatives (backward)

        Parameters
        ----------
        nextDerivatives: ndarray
            a numpy array containing the derivatives from next layer
        nextWeights : ndarray
            a numpy array containing the weights from next layer

        Change deltas
        -------
        deltas: ndarray
            a numpy array containing the partial derivatives on this layer
        """

        # Here the implementation of partial derivative calculation
	
	####
	# output layer: nextDerivatives = target
	####
	if (self.is_classifier_layer == True):
		for j in range(self.size):
			self.deltas[0,j] = (nextDerivatives - self.outp[0,j])*self.outp[0,j]*(1. - self-outp[0,j])
		####
		# hidden layer
		####
		# +1 in nextWeights because weight[0,:] is always the bias which corresponds to no input.
	else:
		for j in range(self.size):
			self.deltas[0,j] = self.outp[0,j] * (1. - self.outp[0,j]) * np.dot(nextDerivatives, nextWeights[j+1,:])

        pass

    def updateWeights(self):
        """
        Update the weights of the layer
        """
        # Here the implementation of weight updating mechanism
	for i in range(self.n_in + 1):
		for j in range(self.size):
			self.weights[i,j] += self.learningRate * self.deltas[0,j] * self.inp[0,i]
        pass

    def _fire(self, inp):
        return Activation.sigmoid(np.dot(np.array(inp), self.weights)
