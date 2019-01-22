"""
COMP 551 (Applied Machine Learning) Assignment 2 Question 3
"LINEAR CLASSIFICATION AND NEAREST NEIGHBOUR CLASSIFICATION"
Name: RASHIK HABIB
McGill University
Date: 10th February, 2018
"""

import numpy as np

"""------------------------------VARIABLES----------------------------------"""
print_on = 1

K = 5
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

"""---------------------------K-NEAREST-NEIGHBOURS--------------------------"""
# Assumes DS1_trainset.txt and DS1_testset.txt has been generated (using the code above)
DS1_trainset = np.genfromtxt("hwk2_datasets_corrected/DS1_trainset.txt", dtype=float, delimiter=',')
DS1_testset = np.genfromtxt("hwk2_datasets_corrected/DS1_testset.txt", dtype=float, delimiter=',')

# Take each example from the testset and find the Frobenius Norm to each training example...
# ...and pick the K nearest examples. Check the majority class for these examples...
# ...and classify the test example accordingly
classifier = np.zeros((DS1_testset.shape[0],1))

for i in range(DS1_testset.shape[0]):
    norm = []
    neighbours = []
    test_example = DS1_testset[i,:-1]
    
    for j in range(DS1_trainset.shape[0]):
        train_example = DS1_trainset[j,:-1]
        difference = train_example - test_example
        norm.append((np.linalg.norm(difference), j))
    
    norm.sort()
    
    # Pick K nearest neighbours 
    for p in range(K):
        neighbours.append(norm[p][1])
    
    # If more 1s, then classify as positive, otherwise negative
    if np.sum(DS1_trainset[neighbours,-1]) >= 0:
        classifier[i][0] = 1    
    else:
        classifier[i][0] = -1

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
    print("K = " + str(K))
    print("Accuracy = " + str(accuracy))
    print("Precision = " + str(precision))
    print("Recall = " + str(recall))
    print("F-measure = " + str(F_measure))
        