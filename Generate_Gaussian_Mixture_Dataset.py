"""
COMP 551 (Applied Machine Learning) Assignment 2 Question 4
"LINEAR CLASSIFICATION AND NEAREST NEIGHBOUR CLASSIFICATION"
Name: RASHIK HABIB
McGill University
Date: 10th February, 2018
"""

import numpy as np


"""----------------------------GENERATE DATA--------------------------------"""
pos1_mean = np.genfromtxt("hwk2_datasets_corrected/DS2_c1_m1.txt", dtype=float, delimiter=',')[:-1]
pos2_mean = np.genfromtxt("hwk2_datasets_corrected/DS2_c1_m2.txt", dtype=float, delimiter=',')[:-1]
pos3_mean = np.genfromtxt("hwk2_datasets_corrected/DS2_c1_m3.txt", dtype=float, delimiter=',')[:-1]

neg1_mean = np.genfromtxt("hwk2_datasets_corrected/DS2_c2_m1.txt", dtype=float, delimiter=',')[:-1]
neg2_mean = np.genfromtxt("hwk2_datasets_corrected/DS2_c2_m2.txt", dtype=float, delimiter=',')[:-1]
neg3_mean = np.genfromtxt("hwk2_datasets_corrected/DS2_c2_m3.txt", dtype=float, delimiter=',')[:-1]

cov1 = np.genfromtxt("hwk2_datasets_corrected/DS2_Cov1.txt", dtype=float, delimiter=',')[:, :-1]
cov2 = np.genfromtxt("hwk2_datasets_corrected/DS2_Cov2.txt", dtype=float, delimiter=',')[:, :-1]
cov3 = np.genfromtxt("hwk2_datasets_corrected/DS2_Cov3.txt", dtype=float, delimiter=',')[:, :-1]

# Generate no.of samples based on probability for each of the data being from that Gaussian
pos1_examples = np.random.multivariate_normal(pos1_mean, cov1, 200)
pos2_examples = np.random.multivariate_normal(pos2_mean, cov2, 840)
pos3_examples = np.random.multivariate_normal(pos3_mean, cov3, 960)

neg1_examples = np.random.multivariate_normal(neg1_mean, cov1, 200)
neg2_examples = np.random.multivariate_normal(neg2_mean, cov2, 840)
neg3_examples = np.random.multivariate_normal(neg3_mean, cov3, 960)

# Join and shuffle before splitting into test and training data
pos_examples = np.concatenate((pos1_examples, pos2_examples, pos3_examples), axis=0)
neg_examples = np.concatenate((neg1_examples, neg2_examples, neg3_examples), axis=0)

np.random.shuffle(pos_examples)
np.random.shuffle(neg_examples)

#30-70 split for test-train datasets, for each class, saved as text files
pos_testset = np.concatenate((pos_examples[0:600,:], np.ones((600,1))), axis=1)
pos_trainset = np.concatenate((pos_examples[600:2000,:], np.ones((1400,1))), axis=1)

neg_testset = np.concatenate((neg_examples[0:600,:], np.ones((600,1))*-1), axis=1)
neg_trainset = np.concatenate((neg_examples[600:2000,:], np.ones((1400,1))*-1), axis=1)

DS2_testset = np.concatenate((pos_testset,neg_testset), axis=0)
DS2_trainset = np.concatenate((pos_trainset,neg_trainset), axis=0)

np.savetxt("hwk2_datasets_corrected/DS2_trainset.txt", DS2_trainset, delimiter=',')
np.savetxt("hwk2_datasets_corrected/DS2_testset.txt", DS2_testset, delimiter=',')
