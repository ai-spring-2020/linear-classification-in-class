"""
linear_classifier.py
Usage: python3 linear_classifier.py data_file.csv
"""

import csv, sys, random, math


EPOCHS = 1000
ALPHA = 0.05

################################################################################
### Utility functions

def read_data(filename, delimiter=",", has_header=True):
    """Reads data from filename. The optional parameters can allow it
    to read data in different formats. Returns a list of headers and a
    list of lists of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        pair = (example[:-1], example[-1])
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def weights_to_slope_intercept(weights):
    """Turns weights into slope-intercept form for 2D spaces"""

    slope = - weights[1] / weights[2]
    intercept = - weights[0] / weights[2]

    return (slope, intercept)

################################################################################
### Linear classification with a hard threshold

def accuracy_hard_threshold(weights, pairs):
    """Finds the accuracy of these weights on this data."""

    incorrect = 0
    total = len(pairs)
    for (x, y) in pairs:
        if perceptron_hypothesis(weights, x) != y:
            incorrect += 1

    return 1 - (incorrect / total)

def perceptron_hypothesis(weights, x):
    """Gives the hypothesis for this input x given the weights"""

    if dot_product(weights, x) > 0:
        return 1
    return 0

def perceptron_learning_rule(alpha, weights, example):
    """Learning rule for perceptron/hard threshold."""

    # x is list of inputs, y is correct class
    (x, y) = example

    # get the hypothesis
    hyp = perceptron_hypothesis(weights, x)

    # updating the weights
    if hyp != y:
        for i in range(len(weights)):
            weights[i] = weights[i] + (alpha * (y - hyp) * x[i])


def linear_classification_hard_threshold(training):
    """We will implement linear classification with a hard threshold here.
    This use stochastic gradient descent to tune weights"""

    # Define initial weights as random numbers
    weights = [random.random() for _ in range(len(training[0][0]))]

    # Gradient descent
    for e in range(EPOCHS):
        # Choose a random example
        example = random.choice(training)
        old_weights = weights[:]

        perceptron_learning_rule(ALPHA, weights, example)

        # Report on weights when they change
        if weights != old_weights:
            (slope, intercept) = weights_to_slope_intercept(weights)

            print()
            print("Epoch:", e)
            print("y = {} x + {}".format(slope, intercept))
            print("weights =", weights)
            print("accuracy =", accuracy_hard_threshold(weights, training))

    return weights


################################################################################
### Linear classification with logistic regression

def logistic(z):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -z)
    except OverflowError:
        return 0.0
    return 1.0 / denom

def logistic_accuracy(weights, pairs):
    """Finds the accuracy of these weights on these pairs of data."""
    error = 0
    total = len(pairs)

    for (x, y) in pairs:
        error += (logistic_hypothesis(weights, x) - y) ** 2

    return 1 - (error / total)

def logistic_hypothesis(weights, x):
    """Hypothesis for logistic classification"""
    return logistic(dot_product(weights, x))

def logistic_learning_rule(alpha, weights, example):
    """Learning rule for logistic classification."""

    # x is list of inputs, y is correct class
    (x, y) = example

    # get the hypothesis
    hyp = logistic_hypothesis(weights, x)

    # updating the weights
    for i in range(len(weights)):
        weights[i] = weights[i] + (alpha * (y - hyp) * hyp * (1 - hyp) * x[i])

def logistic_classification(training):
    """We will implement linear classification with a logistic threshold here.
    This use stochastic gradient descent to tune weights"""

    # Define initial weights as random numbers
    weights = [random.random() for _ in range(len(training[0][0]))]

    # Gradient descent
    for e in range(EPOCHS):
        # Choose a random example
        example = random.choice(training)
        old_weights = weights[:]

        logistic_learning_rule(ALPHA, weights, example)

        # Report on weights when they change
        if weights != old_weights:
            (slope, intercept) = weights_to_slope_intercept(weights)

            print()
            print("Epoch:", e)
            print("y = {} x + {}".format(slope, intercept))
            print("weights =", weights)
            print("accuracy =", logistic_accuracy(weights, training))

    return weights



def main():
    # Read data from the file provided at command line
    header, data = read_data(sys.argv[1], ",")

    # Convert data into (x, y) tuples
    example_pairs = convert_data_to_pairs(data)

    # Insert 1.0 as first element of each x to work with the dummy weight
    training = [([1.0] + x, y) for (x, y) in example_pairs]

    # See what the data looks like
    for (x, y) in training:
        print("x = {}, y = {}".format(x, y))

    print(training)

    # Run linear classification. This is what you need to implement
    # w = linear_classification_hard_threshold(training)

    # Logistic classification
    w = logistic_classification(training)

    for (x, y) in training:
        lh = logistic_hypothesis(w, x)
        error = (lh - y) ** 2
        print("x = {}, y = {},\th_w(x) = {:0.2f}, error = {:0.2f}".format(x, y, lh, error))


if __name__ == "__main__":
    main()
