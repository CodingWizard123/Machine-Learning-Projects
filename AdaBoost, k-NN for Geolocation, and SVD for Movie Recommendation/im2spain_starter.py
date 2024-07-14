"""
The goal of this assignment is to predict GPS coordinates from image features using k-Nearest Neighbors.
Specifically, have featurized 28616 geo-tagged images taken in Spain split into training and test sets (27.6k and 1k).

The assignment walks students through:
    * visualizing the data
    * implementing and evaluating a kNN regression model
    * analyzing model performance as a function of dataset size
    * comparing kNN against linear regression

Images were filtered from Mousselly-Sergieh et al. 2014 (https://dl.acm.org/doi/10.1145/2557642.2563673)
and scraped from Flickr in 2024. The image features were extracted using CLIP ViT-L/14@336px (https://openai.com/clip/).
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def plot_data(train_feats, train_labels):
    """
    Input:
        train_feats: Training set image features
        train_labels: Training set GPS (lat, lon)

    Output:
        Displays plot of image locations, and first two PCA dimensions vs longitude
    """
    # Plot image locations (use marker='.' for better visibility)
    plt.scatter(train_labels[:, 1], train_labels[:, 0], marker=".")
    plt.title('Image Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    # Run PCA on training_feats
    transformed_feats = StandardScaler().fit_transform(train_feats)
    transformed_feats = PCA(n_components=2).fit_transform(transformed_feats)

    # Plot images by first two PCA dimensions (use marker='.' for better visibility)
    plt.scatter(transformed_feats[:, 0],     # Select first column
                transformed_feats[:, 1],     # Select second column
                c=train_labels[:, 1],
                marker='.')
    plt.colorbar(label='Longitude')
    plt.title('Image Features by Longitude after PCA')
    plt.show()


def grid_search(train_features, train_labels, test_features, test_labels, verbose=True):
    knn = NearestNeighbors().fit(train_features)
    ks = list(range(1, 11)) + [20, 30, 40, 50, 100]
    mean_errors = []

    for k in ks:
        knn.n_neighbors = k
        distances, indices = knn.kneighbors(test_features, return_distance=True)

        errors = []
        for i, nearest_indices in enumerate(indices):
            # Calculate average lat and lon of the k-nearest neighbors
            avg_lat = np.mean(train_labels[nearest_indices, 0])
            avg_lon = np.mean(train_labels[nearest_indices, 1])

            # Calculate displacement error in miles
            lat_error = abs(avg_lat - test_labels[i, 0]) * 69  # latitude conversion to miles
            lon_error = abs(avg_lon - test_labels[i, 1]) * 52  # longitude conversion to miles

            # Euclidean distance in miles as the mean error for this test point
            error = np.sqrt(lat_error**2 + lon_error**2)
            errors.append(error)

        mean_error = np.mean(errors)
        mean_errors.append(mean_error)
        if verbose:
            print(f'{k}-NN mean displacement error (miles): {mean_error:.2f}')

    # Plotting the results
    if verbose:
        plt.plot(ks, mean_errors, marker='o')
        plt.xlabel('Number of Neighbors k')
        plt.ylabel('Mean Displacement Error (miles)')
        plt.title('Mean Displacement Error vs. Number of Neighbors k')
        plt.grid(True)
        plt.show()

    return min(mean_errors)

def weighted_grid_search(train_features, train_labels, test_features, test_labels, verbose=True):
    knn = NearestNeighbors().fit(train_features)
    ks = range(1, 51)  # Adjust range of k as needed
    mean_errors = []

    for k in ks:
        knn.n_neighbors = k
        distances, indices = knn.kneighbors(test_features, return_distance=True)

        errors = []
        for i, (dist, nearest_indices) in enumerate(zip(distances, indices)):
            weights = 1 / (dist + 1e-8)  # Avoid division by zero
            sum_weights = np.sum(weights)
            
            weighted_lat = np.sum(weights * train_labels[nearest_indices, 0]) / sum_weights
            weighted_lon = np.sum(weights * train_labels[nearest_indices, 1]) / sum_weights

            lat_error = abs(weighted_lat - test_labels[i, 0]) * 69
            lon_error = abs(weighted_lon - test_labels[i, 1]) * 52

            error = np.sqrt(lat_error**2 + lon_error**2)
            errors.append(error)

        mean_error = np.mean(errors)
        mean_errors.append(mean_error)
        if verbose:
            print(f'{k}-NN weighted mean displacement error (miles): {mean_error:.2f}')

    # Plotting the results
    if verbose:
        plt.plot(ks, mean_errors, marker='o')
        plt.xlabel('Number of Neighbors k')
        plt.ylabel('Mean Displacement Error (miles)')
        plt.title('Weighted Mean Displacement Error vs. Number of Neighbors k')
        plt.grid(True)
        plt.show()

    return min(mean_errors)



def main():
    print("Predicting GPS from CLIP image features\n")

    # Import Data
    print("Loading Data")
    data = np.load('/Users/christian/Documents/GitHub/CS189/hw7/im2spain/im2spain_data.npz')

    # Split tran and test set
    train_features = data['train_features']  # [N_train, dim] array
    test_features = data['test_features']    # [N_test, dim] array
    train_labels = data['train_labels']      # [N_train, 2] array of (lat, lon) coords
    test_labels = data['test_labels']        # [N_test, 2] array of (lat, lon) coords
    train_files = data['train_files']        # [N_train] array of strings
    test_files = data['test_files']          # [N_test] array of strings

    # Data Information
    print('Train Data Count:', train_features.shape[0])

    # Part A: Feature and label visualization (modify plot_data method)
    #plot_data(train_features, train_labels)

    # Find the 3 nearest neighbors of test image 53633239060.jpg
    
    # Fit the model with the training data
    knn = NearestNeighbors(n_neighbors=3).fit(train_features)

    # Find image data in test set
    image = '53633239060.jpg'
    image_data = test_features[np.where(test_files == image)]
    
    # Find the 3 nearest neighbors of the `image`
    nearestNeighbors = knn.kneighbors(image_data, 3, return_distance=False)[0]

    # Extract the data of 3 nearest images to the test image
    imageOneIdx = nearestNeighbors[0]
    imageTwoIdx = nearestNeighbors[1]
    imageThreeIdx = nearestNeighbors[2]

    print(train_files[imageOneIdx])
    print(train_files[imageTwoIdx])
    print(train_files[imageThreeIdx])

    # (c): establish a naive baseline of predicting the mean of the training set
    meanLongitude = np.mean(train_labels[:, 0]) # mean longitude
    meanLatitude = np.mean(train_labels[:, 1]) # mean latitude

    baselinePredictions = np.full(test_labels.shape, (meanLongitude, meanLatitude))

    displacement = test_labels - baselinePredictions

    meanLongitudeDisplacement = (np.mean(displacement[:, 0])) * 52
    meanLatitudeDisplacement = (np.mean(displacement[:, 1])) * 69

    print(f"MDE Longitude in Miles: {meanLongitudeDisplacement}")
    print(f"MDE Latitude in Miles: {meanLatitudeDisplacement}")


    #grid_search(train_features, train_labels, test_features, test_labels)
    
    # Parts G: rerun grid search after modifications to find the best value of k
    weighted_grid_search(train_features, train_labels, test_features, test_labels)
    
    #compare to linear regression for different # of training points
    mean_errors_lin = []
    mean_errors_nn = []

    ratios = np.arange(0.1, 1.1, 0.1)

    for r in ratios:
        # determine split size
        num_samples = int(r * len(train_features))
        print('samples:', num_samples)
        
        # Define regression model
        regression = LinearRegression()

        # split training data
        trainingData = train_features[:num_samples]
        trainingLabels = train_labels[:num_samples]
        
        regression.fit(trainingData, trainingLabels)
        linRegPred = regression.predict(test_features)
        
        # Calculate displacement error in miles
        lat_error = abs(linRegPred[:, 1] - test_labels[:, 1]) * 69  
        lon_error = abs(linRegPred[:, 0] - test_labels[:, 0]) * 52 

        # Euclidean distance in miles as the mean error for this test point
        e_lin = np.mean(np.sqrt(lat_error**2 + lon_error**2))
        mean_errors_lin.append(e_lin)

        e_nn = grid_search(trainingData, trainingLabels, test_features, test_labels, verbose=False)
        mean_errors_nn.append(e_nn)

        print(f'\nTraining set ratio: {r} ({num_samples})')
        print(f'Linear Regression mean displacement error (miles): {e_lin:.1f}')
        print(f'kNN mean displacement error (miles): {e_nn:.1f}')


    print(len(mean_errors_lin))
    print(len(mean_errors_nn))

    # Plot error vs training set size
    plt.plot(ratios, mean_errors_lin, label='lin. reg.')
    plt.plot(ratios, mean_errors_nn, label='kNN')
    plt.xlabel('Training Set Ratio')
    plt.ylabel('Mean Displacement Error (miles)')
    plt.title('Mean Displacement Error (miles) vs. Training Set Ratio')
    plt.legend()
    plt.show()
       

if __name__ == '__main__':
    main()