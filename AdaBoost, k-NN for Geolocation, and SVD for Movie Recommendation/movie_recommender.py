import os
import scipy.io
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import pandas as pd

# Load training data from MAT file
R = scipy.io.loadmat('/Users/christian/Documents/GitHub/CS189/hw7/movie_data/movie_train.mat')['train']

# Load validation data from CSV
val_data = np.loadtxt('/Users/christian/Documents/GitHub/CS189/hw7/movie_data/movie_validate.txt', dtype=int, delimiter=',')

# dimensions to test
d_values = [2, 5, 10, 20]

# Helper method to get training accuracy
def get_train_acc(R, user_vecs, movie_vecs):
    num_correct, total = 0, 0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if not np.isnan(R[i, j]):
                total += 1
                if np.dot(user_vecs[i], movie_vecs[j])*R[i, j] > 0:
                    num_correct += 1
    return num_correct/total

# Helper method to get validation accuracy
def get_val_acc(val_data, user_vecs, movie_vecs):
    num_correct = 0
    for val_pt in val_data:
        user_vec = user_vecs[val_pt[0]-1]
        movie_vec = movie_vecs[val_pt[1]-1]
        est_rating = np.dot(user_vec, movie_vec)
        if est_rating*val_pt[2] > 0:
            num_correct += 1
    return num_correct/val_data.shape[0]

# Helper method to get indices of all rated movies for each user,
# and indices of all users who have rated that title for each movie
def get_rated_idxs(R):
    user_rated_idxs, movie_rated_idxs = [], []
    for i in range(R.shape[0]):
        user_rated_idxs.append(np.argwhere(~np.isnan(R[i, :])).reshape(-1))
    for j in range(R.shape[1]):
        movie_rated_idxs.append(np.argwhere(~np.isnan(R[:, j])).reshape(-1))
    return np.array(user_rated_idxs, dtype=object), np.array(movie_rated_idxs, dtype=object)

# Part (c): SVD to learn low-dimensional vector representations
def svd_lfm(R):

    # Impute Nan values in R with 0
    idx = np.isnan(R[:,])
    R[idx] = 0

    # Compute eigenvalues and eigenvectors of R.T @ R 
    eigenvalues, eigenvectors = np.linalg.eig(R.T @ R)

    # sort the eigenvalues indices in descending order
    eigenvaluesSortedIdx = np.argsort(eigenvalues)[::-1] 

    # sort eigenvalues and eigenvectors in descending order
    eigenvalues = eigenvalues[eigenvaluesSortedIdx]

    # sort eigenvectors in descending order (right singular vectors: describes the columns of R)
    V= eigenvectors[:, eigenvaluesSortedIdx]

    # compute the SVD of R
    # construct Diagonal matrix D (singular values of R)
    # meaures the amount of variance captured by each corresponding vector in U, and V
    # i.e. the first/largest eigenvalue represents the importance/strength of the pattern associated with the 
    # first columns of U and V 
    D = np.diag(np.sqrt(eigenvalues))

    # compute U (left singular vectors of R: describe the rows of R)
    U = (R @ V @ np.linalg.pinv(D) / np.linalg.norm(R @ V @ np.linalg.pinv(D), axis=0))

    # construct the matrix R = UDV.T
    R_approx = U @ D @ V.T

    return U, V

# Part (d): Compute the training MSE loss of a given vectorization
def get_train_mse(R, user_vecs, movie_vecs):
    # Compute the training MSE loss
    xy = user_vecs @ movie_vecs.T
    xyr = (xy - R) ** 2
    return np.nanmean(xyr)


# Part (e): Compute training MSE and val acc of SVD LFM for various d
d_values = [2, 5, 10, 20]
train_mses, train_accs, val_accs = [], [], []
user_vecs, movie_vecs = svd_lfm(np.copy(R))

for d in d_values:
    train_mses.append(get_train_mse(np.copy(R), user_vecs[:, :d], movie_vecs[:, :d]))
    train_accs.append(get_train_acc(np.copy(R), user_vecs[:, :d], movie_vecs[:, :d]))
    val_accs.append(get_val_acc(val_data, user_vecs[:, :d], movie_vecs[:, :d]))

plt.clf()
plt.plot([str(d) for d in d_values], train_mses, 'o-')
plt.title('Train MSE of SVD-LFM with Varying Dimensionality')
plt.xlabel('d')
plt.ylabel('Train MSE')
plt.savefig(fname='train_mses.png', dpi=600, bbox_inches='tight')
plt.clf()
plt.plot([str(d) for d in d_values], train_accs, 'o-')
plt.plot([str(d) for d in d_values], val_accs, 'o-')
plt.title('Train/Val Accuracy of SVD-LFM with Varying Dimensionality')
plt.xlabel('d')
plt.ylabel('Train/Val Accuracy')
plt.legend(['Train Accuracy', 'Validation Accuracy'])
plt.savefig(fname='trval_accs.png', dpi=600, bbox_inches='tight')


# Part (f): Learn better user/movie vector representations by minimizing loss
best_d = 2
np.random.seed(20)
randomized_user_vecs = np.random.random((R.shape[0], best_d))
randomized_movie_vecs = np.random.random((R.shape[1], best_d))
user_rated_idxs, movie_rated_idxs = get_rated_idxs(np.copy(R))


# Part (f): Function to update user vectors
def update_user_vecs(user_vecs, movie_vecs, R, user_rated_idxs):
    # Update user_vecs to the loss-minimizing value
    out = np.zeros(user_vecs.shape)
    learning_rate = .0001

    for i in range(R.shape[0]):
        temp = np.zeros(user_vecs[i].shape)
        for j in user_rated_idxs[i]:
            temp += (2 * ((user_vecs[i] @ movie_vecs[j] - R[i, j]) * movie_vecs[j]) + (2 * 1/5 * user_vecs[i]))
        out[i] = user_vecs[i] - learning_rate * temp
    
    return out

# Part (f): Function to update user vectors
def update_movie_vecs(user_vecs, movie_vecs, R, movie_rated_idxs):
    
    learning_rate = .0001
    out = np.zeros(movie_vecs.shape)
    for j in range(R.shape[1]):
        temp = np.zeros(movie_vecs[j].shape)   
        for i in movie_rated_idxs[j]:
            temp += 2 * (user_vecs[i] @ movie_vecs[j] - R[i, j]) * user_vecs[i] + (2 * 1/5 * movie_vecs[j])
        out[j] = movie_vecs[j] - learning_rate * temp

    return out

# Part (f): Perform loss optimization using alternating updates
train_mse = get_train_mse(np.copy(R), user_vecs, movie_vecs)
train_acc = get_train_acc(np.copy(R), user_vecs, movie_vecs)
val_acc = get_val_acc(val_data, user_vecs, movie_vecs)

print(f'Start optim, train MSE: {train_mse:.2f}, train accuracy: {train_acc:.4f}, val accuracy: {val_acc:.4f}')

for opt_iter in range(20):
    user_vecs = update_user_vecs(user_vecs, movie_vecs, np.copy(R), user_rated_idxs)
    movie_vecs = update_movie_vecs(user_vecs, movie_vecs, np.copy(R), movie_rated_idxs)
    train_mse = get_train_mse(np.copy(R), user_vecs, movie_vecs)
    train_acc = get_train_acc(np.copy(R), user_vecs, movie_vecs)
    val_acc = get_val_acc(val_data, user_vecs, movie_vecs)
    print(f'Iteration {opt_iter+1}, train MSE: {train_mse:.2f}, train accuracy: {train_acc:.4f}, val accuracy: {val_acc:.4f}')
