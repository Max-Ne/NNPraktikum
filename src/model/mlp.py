
import numpy as np

from util.activation_functions import Activation
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier
#from util.loss_functions import CrossEntropyError # not needed...

# for verbose = True, reporting accuracy
from report.evaluator import Evaluator 


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, input_weights=None,
                 output_task='classification', output_activation='sigmoid',
                 cost='crossentropy', learning_rate=0.01, epochs=50):

        """
        A digit-7 recognizer based on logistic regression algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learning_rate : float
        epochs : positive int

        Attributes
        ----------
        training_set : list
        validation_set : list
        test_set : list
        learning_rate : float
        epochs : positive int
        performances: array of floats
        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_task = output_task  # Either classification or regression
        self.output_activation = output_activation
        self.cost = cost

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.nlayers = layers
        self.input_weights = input_weights

        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1,
                                            axis=1)
        self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
                                              axis=1)
        self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)

        # Build up the network from specific layers
        # Here is an example of a MLP acting like the Logistic Regression
        #self.layers = []
        #output_activation = "sigmoid"
        #self.layers.append(LogisticLayer(10, 1, None, output_activation, True))
        
        
        ##########
        #my (first) try to implement it
        ##########
        
        self.activation_values = [0]*self.nlayers
	self.layers = []
	mid_activation = 'sigmoid'

        # number of neurons in each layer
        # for beginning 10 each
        # 0,1,2,....,self.nlayers-1
        #n_outp = [5,10]
        self.n_outp = [10]*(self.nlayers)
        
        #first layer, first input is 10. (should be number of pixles????, len(self.training_set.input[0])
        print train.input.shape[1]
        self.layers.append(LogisticLayer(train.input.shape[1]-1,self.n_outp[0],None,mid_activation, False))
        
        #mid layers
        print self.nlayers-1
        for l in range(self.nlayers-1):
	  if (l == 0):			#(1,2,... self.layers-2)
	    continue;
	  self.layers.append(LogisticLayer(self.n_outp[l-1],self.n_outp[l],None,mid_activation, False))
	
	#last layer, does classification
	self.layers.append(LogisticLayer(self.n_outp[self.nlayers-2],self.n_outp[self.nlayers -1],None,self.output_activation, True));
        

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """


	# - propagate one time through all layers
	# - store all activation values, one array for each layer (necessary??)
	# - nothing to return, just manipulated self.activation_values

	for i in range(self.nlayers): #i=0,1,...,self.nlayers-1
	  outp = self.layers[i].forward(inp);
	  inp = np.insert(outp,0,1)

        return outp

    def _compute_error(self, target):
        """
        Compute the total error of the network

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        
        last_layer = self._get_output_layer();
        
        # target_arr something like [0,0,0,...,1,0,0] for 7, for other digits similiar.
        target_arr = np.zeros(10);
        target_arr[target] += 1.;
        
        last_layer.computeDerivative(np.array(target_arr - last_layer.outp),np.ones((self.n_outp[-1],self.n_outp[-1])))
        
        
        for l in range(self.nlayers-1):
	  r = -l-2			# r = -2, -3, -4, ..., -self.nlayers
	  #print self.layers[r+1].deltas.shape
	  #print self.layers[r+1].weights[1:].shape
	  self.layers[r].computeDerivative(self.layers[r+1].deltas,self.layers[r+1].weights[1:])
        
        
        return last_layer.deltas
        

    def _update_weights(self):
        """
        Update the weights of the layers by propagating back the error
        """
        
        for l in range(self.nlayers):
	    self.layers[l].updateWeights(self.learning_rate)    
        
        pass

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

	####
	
	#evalutation is copy paste from Run.py ...
	evaluator = Evaluator()
	if verbose == True:
	  print "epoch-nr. & Result of the Multi-layer Perceptron recognizer (on test set):"
	
	for epoch in range(self.epochs):
	  
	  self._train_one_epoch();

	  if verbose == True:
	    print epoch
	    evaluator.printAccuracy(self.test_set, self.evaluate())
	    
        pass

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """
        
        #set up random order of learning the different instances.
        random_order = np.arange(len(self.training_set.input))
        np.random.shuffle(random_order)
	
        
        #loop over training samples
	for instance in random_order:
	
	  #feed forward
	  self._feed_forward(self.training_set.input[instance]);
	  
	  #compute error of last layer
	  self._compute_error(self.training_set.label[instance]);
	  
	  #update weights
	  self._update_weights();
	
        pass

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        
        # return index of maximum in output of last layer created by the test instance
        # it is saved in self.activation_values[last entry]
        #self._feed_forward(test_instance)
        #return np.argmax(self.activation_values[self.nlayers-1])
        self._feed_forward(test_instance)
        return np.argmax(self._get_output_layer().outp)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.test_set.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        #print list(map(self.classify, test))
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.training_set.input = np.delete(self.training_set.input, 0, axis=1)
        self.validation_set.input = np.delete(self.validation_set.input, 0,
                                              axis=1)
        self.test_set.input = np.delete(self.test_set.input, 0, axis=1)
