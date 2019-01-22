"""
COMP 551 (Applied Machine Learning) Assignment 2 Question 1
"LINEAR CLASSIFICATION AND NEAREST NEIGHBOUR CLASSIFICATION"
Name: RASHIK HABIB
McGill University
Date: 10th February, 2018
"""

import numpy as np

"""----------------------------GENERATE DATA--------------------------------"""
neg_mean = np.genfromtxt("hwk2_datasets_corrected/DS1_m_0.txt", dtype=float, delimiter=',')[:-1]

pos_mean = np.genfromtxt("hwk2_datasets_corrected/DS1_m_1.txt", dtype=float, delimiter=',')[:-1]

cov = np.genfromtxt("hwk2_datasets_corrected/DS1_Cov.txt", dtype=float, delimiter=',')[:, :-1]

pos_examples = np.random.multivariate_normal(pos_mean, cov, 2000)
neg_examples = np.random.multivariate_normal(neg_mean, cov, 2000)

#30-70 split for test-train datasets, for each class, saved as text files
pos_testset = np.concatenate((pos_examples[0:600,:], np.ones((600,1))), axis=1)
pos_trainset = np.concatenate((pos_examples[600:2000,:], np.ones((1400,1))), axis=1)

neg_testset = np.concatenate((neg_examples[0:600,:], np.ones((600,1))*-1), axis=1)
neg_trainset = np.concatenate((neg_examples[600:2000,:], np.ones((1400,1))*-1), axis=1)

DS1_testset = np.concatenate((pos_testset,neg_testset), axis=0)
DS1_trainset = np.concatenate((pos_trainset,neg_trainset), axis=0)

np.savetxt("hwk2_datasets_corrected/DS1_trainset.txt", DS1_trainset, delimiter=',')
np.savetxt("hwk2_datasets_corrected/DS1_testset.txt", DS1_testset, delimiter=',')

