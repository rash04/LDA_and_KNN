"""
COMP 551 (Applied Machine Learning) Assignment 2 Question 2
"LINEAR CLASSIFICATION AND NEAREST NEIGHBOUR CLASSIFICATION"
Name: RASHIK HABIB
McGill University
Date: 10th February, 2018
"""

import numpy as np
import math

"""------------------------------VARIABLES----------------------------------"""
print_on = 1

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

"""----------------------------PROBABILISITIC LDA---------------------------"""
# Use maximum likelihood to determine means and covariance for each class
# Assumes DS1_trainset.txt and DS1_testset.txt has been generated (using the code above)
DS1_trainset = np.genfromtxt("hwk2_datasets_corrected/DS1_trainset.txt", dtype=float, delimiter=',')
DS1_testset = np.genfromtxt("hwk2_datasets_corrected/DS1_testset.txt", dtype=float, delimiter=',')

pos_examples = DS1_trainset[DS1_trainset[:,-1] == 1]
neg_examples = DS1_trainset[DS1_trainset[:,-1] == -1]

pos_count = pos_examples.shape[0]
neg_count = neg_examples.shape[0]
total_count = pos_count + neg_count

# Sample mean determines maximum likelihood for mean value
pos_mean = np.sum(pos_examples[:,:-1], axis=0)/pos_count
neg_mean = np.sum(neg_examples[:,:-1], axis=0)/neg_count

# Weighted average of each of the covariance matrices for the 2 classes...
# ...separately, gives maximum likelihood for the covariance values
pos_X_minus_mu = pos_examples[:,:-1] - pos_mean.transpose()
pos_S = (np.dot(pos_X_minus_mu.transpose(), pos_X_minus_mu))/pos_count

neg_X_minus_mu = neg_examples[:,:-1] - neg_mean.transpose()
neg_S = (np.dot(neg_X_minus_mu.transpose(), neg_X_minus_mu))/neg_count

cov = pos_S*(pos_count/total_count) + neg_S*(neg_count/total_count)

# Determine parameter values for the probabilistic LDA model
pos_prior = pos_count/total_count
neg_prior = neg_count/total_count

pos_w = np.dot(np.linalg.inv(cov), pos_mean)
neg_w = np.dot(np.linalg.inv(cov), neg_mean)

pos_bias = -0.5 * np.dot( np.dot(pos_mean.transpose(), np.linalg.inv(cov)) , pos_mean) + math.log(pos_prior)
neg_bias = -0.5 * np.dot( np.dot(neg_mean.transpose(), np.linalg.inv(cov)) , neg_mean) + math.log(neg_prior)

pos_discriminant = np.dot(DS1_testset[:,:-1], pos_w) + pos_bias 
neg_discriminant = np.dot(DS1_testset[:,:-1], neg_w) + neg_bias

# Use the discriminant to classify the data
classifier = pos_discriminant - neg_discriminant
classifier[classifier >= 0] = 1
classifier[classifier < 0] = -1

# Report classifier performance
for i in range(classifier.size):
    if classifier[i] == 1:
        if DS1_testset[:,-1][i] == 1:
            true_positive = true_positive + 1
        else:
            false_positive = false_positive + 1
    elif classifier[i] == -1:
        if DS1_testset[:,-1][i] == -1:
            true_negative = true_negative + 1
        else:
            false_negative = false_negative + 1

accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)
precision = true_positive/(true_positive + false_positive)
recall = true_positive/(true_positive + false_negative)
F_measure = (2*precision*recall)/(precision + recall)

if print_on:
    print("LDA Performance: ")
    print("Accuracy = " + str(accuracy))
    print("Precision = " + str(precision))
    print("Recall = " + str(recall))
    print("F-measure = " + str(F_measure))