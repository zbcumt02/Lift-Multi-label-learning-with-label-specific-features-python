import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import scipy.io as sio
from sklearn.metrics import label_ranking_loss
from MLL_metrics import *


file_dir = os.path.dirname(os.path.realpath('__file__'))
data_dir = os.path.join(file_dir, 'data/{}.mat'.format('finaldata'))
data = sio.loadmat(data_dir)

train_data = data['Xapp']
test_data = data['Xgen']
train_target = data['Yapp']
test_target = data['Ygen']

train_data = torch.tensor(train_data, dtype=torch.float32)  # data for train with instance * dim
train_target = torch.tensor(train_target.T, dtype=torch.float32)  # target for train with instance * class

test_data = torch.tensor(test_data, dtype=torch.float32)  # data for test
test_target = torch.tensor(test_target.T, dtype=torch.float32)  # target for test

ratio = 0.1
num_train, dim = train_data.size()
num_class, num_test = test_target.size()

P_Centers = []
N_Centers = []

for i in range(num_class):
    print(f'Performing clustering for the {i+1}/{num_class} class')

    p_idx = (train_target[i] == 1).nonzero().squeeze()  # index for positive class
    n_idx = torch.arange(num_train)[~torch.isin(torch.arange(num_train), p_idx)]  # index for negative class

    p_data = train_data[p_idx]
    n_data = train_data[n_idx]

    k1 = min(int(np.ceil(len(p_idx) * ratio)), int(np.ceil(len(n_idx) * ratio)))
    k2 = k1

    if k1 == 0:
        POS_C = []
        kmeans = KMeans(n_clusters=min(50, num_train), random_state=42)
        NEG_C = kmeans.fit(train_data.numpy()).cluster_centers_
    else:
        POS_C = KMeans(n_clusters=k1, n_init=10, random_state=42).fit(p_data.numpy()).cluster_centers_
        NEG_C = KMeans(n_clusters=k2, n_init=10, random_state=42).fit(n_data.numpy()).cluster_centers_

    P_Centers.append(POS_C)
    N_Centers.append(NEG_C)

Models = []

# SVM training
for i in range(num_class):
    print(f'Building classifiers: {i+1}/{num_class}')

    centers = np.vstack([P_Centers[i], N_Centers[i]])

    num_center = centers.shape[0]

    data = []

    if num_center >= 5000:
        raise ValueError('Too many cluster centers, please try to decrease the number of clusters.')

    blocksize = 5000 - num_center
    num_block = int(np.ceil(num_train / blocksize))

    for j in range(num_block-1):
        low = j * blocksize
        high = (j + 1) * blocksize
        tmp_mat = np.vstack([centers, train_data[low:high]])
        Y = torch.cdist(tmp_mat, tmp_mat)
        tmp_mat = torch.tensor(tmp_mat, dtype=torch.float32)
        data.append(Y[num_center:(num_center + blocksize), :num_center].numpy())

    low = (num_block - 1) * blocksize
    high = num_train
    tmp_mat = np.vstack([centers, train_data[low:high]])
    tmp_mat = torch.tensor(tmp_mat, dtype=torch.float32)
    Y = torch.cdist(tmp_mat, tmp_mat)  # instance * instance
    data.append(Y[num_center:(num_center + high - low), :num_center].numpy())

    training_instance_matrix = np.vstack(data)
    training_label_vector = train_target[i].numpy()

    model = SVC(kernel='linear', probability=True)
    model.fit(training_instance_matrix, training_label_vector)
    Models.append(model)

Pre_Labels = []
Outputs = []

for i in range(num_class):
    centers = np.vstack([P_Centers[i], N_Centers[i]])
    num_center = centers.shape[0]

    data = []

    if num_center >= 5000:
        raise ValueError('Too many cluster centers, please try to decrease the number of clusters.')

    blocksize = 5000 - num_center
    num_block = int(np.ceil(num_test / blocksize))

    for j in range(num_block-1):
        low = j * blocksize
        high = (j + 1) * blocksize
        tmp_mat = np.vstack([centers, test_data[low:high]])
        tmp_mat = torch.tensor(tmp_mat, dtype=torch.float32)
        Y = torch.cdist(tmp_mat, tmp_mat)
        data.append(Y[num_center:(num_center + blocksize), :num_center].numpy())

    low = (num_block - 1) * blocksize
    high = num_test
    tmp_mat = np.vstack([centers, test_data[low:high]])
    tmp_mat = torch.tensor(tmp_mat, dtype=torch.float32)
    Y = torch.cdist(tmp_mat, tmp_mat)
    data.append(Y[num_center:(num_center + high - low), :num_center].numpy())

    testing_instance_matrix = np.vstack(data)
    testing_label_vector = test_target[i].numpy()

    predicted_label = Models[i].predict(testing_instance_matrix)
    prob_estimates = Models[i].predict_proba(testing_instance_matrix)

    if predicted_label.size == 0:
        predicted_label = np.full(num_test, train_target[i, 0])
        Prob_pos = np.ones(num_test) if train_target[i, 0] == 1 else np.zeros(num_test)
        Outputs.append(Prob_pos)
        Pre_Labels.append(predicted_label)
    else:
        pos_index = np.where(Models[i].classes_ == 1)[0][0]
        Prob_pos = prob_estimates[:, pos_index]
        Outputs.append(Prob_pos)
        Pre_Labels.append(predicted_label)

Outputs = torch.tensor(np.array(Outputs), dtype=torch.double)
Pre_Labels = torch.tensor(np.array(Pre_Labels), dtype=torch.float32)

hamming = hamming_loss(test_target, Pre_Labels)
ranking_loss = ranking_loss(Outputs, test_target)
avg_precision = average_precision(Outputs, test_target)
coverage = coverage(Outputs, test_target)
one_error = one_error(Outputs, test_target)

print("Hamming Loss: %.5f\n"
      "Ranking Loss: %.5f\n"
      "Average Precision: %.5f\n"
      "Coverage: %.5f\n"
      "One Error: %.5f" % (hamming, ranking_loss, avg_precision, coverage, one_error))


