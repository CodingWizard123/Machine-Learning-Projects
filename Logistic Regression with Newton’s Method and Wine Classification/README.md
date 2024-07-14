# Logistic Regression with Newton’s Method and Wine Classification

## Project Explanation

This project aims to implement logistic regression using Newton’s method for optimization and apply it to classify a wine dataset. The project involves deriving mathematical expressions for the gradient and Hessian, performing logistic regression with Newton’s method, and implementing batch and stochastic gradient descent for comparison. Additionally, the project explores feature normalization, PCA for dimensionality reduction, and regularization techniques.

## Mathematical Concepts

### Cost Function for Logistic Regression
The cost function \( J(w) \) for logistic regression is given by:

$\[ J(w) = -\sum_{i=1}^{n} \left( y_i \log(s_i) + (1 - y_i) \log(1 - s_i) \right) \]$

where $\( s_i = \sigma(x_i \cdot w) \)$ is the sigmoid function.

### Gradient of the Cost Function
The gradient of the cost function \( \nabla_w J(w) \) is derived as:

$\[ \nabla_w J(w) = X^T \cdot (s - y) \]$

where $\( X \)$ is the design matrix, $\( s \)$ is the vector of predictions, and $\( y \)$ is the vector of actual labels.

### Hessian of the Cost Function
The Hessian matrix $\( H \)$ of the cost function is given by:

$\[ H = X^T \cdot S \cdot (I - S) \cdot X \]$

where $\( S \)$ is a diagonal matrix with elements $\( s_i (1 - s_i) \)$.

### Newton's Method
Newton’s method for optimization updates the weight vector \( w \) as follows:

$\[ w \leftarrow w - H^{-1} \cdot \nabla_w J(w) \]$

## Machine Learning Concepts

### Logistic Regression
Logistic regression is a classification algorithm used to predict the probability of a binary outcome. It uses the logistic function to model the probability of the default class (0 or 1).

### Batch Gradient Descent
Batch gradient descent updates the weights using the entire dataset in each iteration. The update rule with \( L2 \) regularization is:

$\[ w \leftarrow w + \epsilon \left( X^T (y - s(X \cdot w)) + \lambda w \right) \]$

### Stochastic Gradient Descent (SGD)
SGD updates the weights using one training example at a time. The update rule with $\( L2 \)$ regularization is:

$\[ w \leftarrow w + \epsilon (y_i - s(x_i \cdot w)) x_i + \lambda w \]$

### Principal Component Analysis (PCA)
PCA is a dimensionality reduction technique that projects the data onto the directions of maximum variance. It is used to reduce the number of features while retaining most of the variance.

## Python Concepts

### Libraries Used
- **NumPy**: For numerical operations.
- **SciPy**: For specialized functions and optimizations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Plotly**: For data visualization.
- **Scikit-learn**: For preprocessing and evaluation metrics.

### Key Functions
- **Normalization**: Standardizing features by removing the mean and scaling to unit variance.
- **Batch Gradient Descent Implementation**: Custom implementation for logistic regression.
- **Stochastic Gradient Descent Implementation**: Custom implementation for logistic regression.
- **PCA Implementation**: Custom implementation for reducing the dataset’s dimensionality.

## Importance

This project showcases the practical implementation of logistic regression using Newton’s method, batch gradient descent, and stochastic gradient descent. The application of PCA highlights the importance of dimensionality reduction in improving model performance. The techniques and methods used are fundamental in machine learning and can be applied to various real-world problems, including classification tasks in finance, healthcare, and marketing.

## Outcomes

### Results
The logistic regression model was trained and evaluated on the wine dataset. Key results include:
- **Initial weights and predictions**: Demonstrated the convergence of Newton’s method.
- **Batch Gradient Descent**: Converged to a minimum cost of 0.1224.
- **Stochastic Gradient Descent**: Converged faster but to a slightly higher cost of 0.2323.
- **PCA Transformation**: Reduced the data dimensionality from 12 to 6 while retaining 82.57% of the variance.
- **Kaggle Competition**: Achieved a best score of 0.99 by fitting the logistic model on the PCA-transformed data.

### Visualizations
- **Cost Function vs. Iterations**: Plots showing the convergence of batch and stochastic gradient descent.
- **PCA Components**: Visualizations of the explained variance and the transformed data.

The results demonstrate the efficiency of Newton’s method and the importance
feature engineering and dimensionality reduction in improving model performance.
The project's approach and findings provide valuable insights for anyone interested
in logistic regression and optimization techniques in machine learning.

