#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven # reader for data
#var.: trainingSet, validationSet, testSet
#each var of type DataSet -> .label, .input, .targetDigit, .oneHot, .__iter__()

from model.stupid_recognizer import StupidRecognizer 
#var.: byChance=0.5, train, valid, test
#evaluate is random()<byChance for each el. (50%)

from model.perceptron import Perceptron
#to be implemented

#from model.logistic_regression import LogisticRegression
#does not exist... what is this?!

from report.evaluator import Evaluator
#print InputLabel, ResultLabel, Comparison, ClassiciationResult, ConfusionMatrix, Accuracy
#uses sklearn & __future__
#only for evaluation of the output


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    # see above, data correctly read.
    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    #actually pretty stupid and pompous, only for reference


    # Uncomment this to make your Perceptron evaluated
    myPerceptronClassifier = Perceptron(data.trainingSet,
					data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=30)

    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nStupid Classifier has been training..")
    myStupidClassifier.train()
    print("Done..")

    print("\nPerceptron has been training..")
    myPerceptronClassifier.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    stupidPred = myStupidClassifier.evaluate()
    # Uncomment this to make your Perceptron evaluated
    perceptronPred = myPerceptronClassifier.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    evaluator.printAccuracy(data.testSet, stupidPred)

    print("\nResult of the Perceptron recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    # Uncomment this to make your Perceptron evaluated
    evaluator.printAccuracy(data.testSet, perceptronPred)

    # eval.printConfusionMatrix(data.testSet, pred)
    # eval.printClassificationResult(data.testSet, pred, target_names)

if __name__ == '__main__':
    main()
