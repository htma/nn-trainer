A Neural Network trainer, which can help us observe the training process.

** 1. Generate the training data and test data
This program first generate a sequence of points by using numpy.random.rand function in a rectangle (-1.5, 1.5). The forumlar is as follows:
scaled_value = min + (rand_value*(max-min)).

Second, we check every point whether it is in a unit circle and to label it with 1 and 0.

Third, we write all these points and their labels into a csv file.

