# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
from sklearn.metrics import accuracy_score

from util.activation_functions import Activation
from model.classifier import Classifier

from report.evaluator import Evaluator

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


# Name: Max Neukum
# Matrikelnummer: 1599500
# Email: m.neukum@web.de
# date: 9.may 2016 22:00


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : ndarray
    """
    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
	
        # Initialize the weight vector with small random values
        # around 0 and 0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/10
"""
        # add bias weights at the beginning with the same random initialize
        np.insert(self.weight, 0, np.random.rand()/10)

        # add bias values ("1"s) at the beginning of all data sets
        np.insert(self.trainingSet.input, 0, 1, axis=1)
        np.insert(self.validationSet.input, 0, 1, axis=1)
        np.insert(self.testSet.input, 0, 1, axis=1)
"""
    def train(self, verbose=True):
        """
        Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Here you have to implement the Perceptron Learning Algorithm
        # to change the weights of the Perceptron

	# loop over epochs
	for k in range(self.epochs):
		
		wrong_sets = [0]*len(self.trainingSet.input[0]) 
		# list for sum over miss identificated sets

		#iterate over training sets
		for i in range(len(self.trainingSet.input)):

			# upper left pixel is always greyscale 0
			# -> weight[0] is useless
			# -> use weight[0] as bias and set input[i,0] = 1	
			self.trainingSet.input[i,0] = 1.;

			#sign = 2 * label - 1
			#label = 1 -> sgn = 1
			#label = 0 -> sgn = -1 -> invert (cf. slides)
			sgn = 2*self.trainingSet.label[i] - 1.
		
			# if false identification add to 1 list
			if not((self.fire(sgn * self.trainingSet.input[i]) == 1)):
				wrong_sets = np.add(wrong_sets, sgn * self.trainingSet.input[i])

		# update weights
		for j in range(len(self.weight)):
			self.weight[j] = self.weight[j] + self.learningRate * wrong_sets[j] 
		
		# verbose; validation accuracy after each epoch
		self.evaluator = Evaluator();
		#verbose = False;
		if (verbose == True):
			print "End of epoch nr." + str(k+1) + ": "
			self.evaluator.printAccuracy(self.validationSet, self.evaluate(test=self.validationSet.input))
        pass

"""
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validationSet.label,
                                          self.evaluate(self.validationSet))
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy*100))
                print("-----------------------------")

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        for img, label in zip(self.trainingSet.input, self.trainingSet.label):
            output = self._fire(img)  # real output of the neuron
            error = int(label) - int(output)

            # online learning: updating weights after seeing 1 instance
            self.weight += self.learningRate * error * img

        # if we want to do batch learning, accumulate the error
        # and update the weight outside the loop
>>>>>>> upstream/master
"""
    def classify(self, testInstance):
        """Classify a single instance.

	Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        # Here you have to implement the classification for one instance,
        # i.e., return True if the testInstance is recognized as a 7,
        # False otherwise
	if not(testInstance[0] == 0 or testInstance[0] == 1):
		print 'upper left pixel is not greyscale 0'
	testInstance[0] = 1;
	
	return self.fire(testInstance);
"""	
        return self._fire(testInstance)

>>>>>>> upstream/master
"""
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
            test = self.testSet.input

        # Here is the map function of python - a functional programming concept
        # It applies the "classify" method to every element of "test"
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def _fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        # I already implemented it for you to see how you can work with numpy
        return Activation.sign(np.dot(np.array(input), self.weight))
