# Gaussian Classification and Support Vector Machines for Digit and Spam Detection

## Project Explanation
This project focuses on applying Gaussian classification and support vector machines (SVM) to build classifiers for digit recognition and spam detection. The primary goal is to demonstrate the effectiveness of these models in classification tasks using Gaussian distributions to model the data and SVMs to create robust classifiers.

### Methodology and Workflow
1. **Data Preprocessing**: The MNIST dataset for digit recognition and a custom dataset for spam detection were used. Data preprocessing steps included normalization and feature extraction.
2. **Gaussian Classification**: Gaussian distributions were fitted to the data of each class, and maximum likelihood estimation (MLE) was used to estimate the parameters.
3. **Support Vector Machines**: SVM classifiers were trained on the processed data, and hyperparameters were optimized using grid search.
4. **Model Evaluation**: The performance of the classifiers was evaluated using metrics such as accuracy, confusion matrices, and ROC curves.

## Mathematical Concepts
### Gaussian Distributions
Gaussian distributions are defined by their mean (μ) and covariance matrix (Σ). The probability density function (PDF) of a Gaussian distribution in d dimensions is given by:
$\[ f(x|\mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right) \]$

### Maximum Likelihood Estimation (MLE)
MLE is used to estimate the parameters (mean and covariance) of the Gaussian distributions. For a set of observations $\(X = \{x_1, x_2, \ldots, x_n\}\)$, the log-likelihood function is:
$\[ \log L(\mu, \Sigma) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log|\Sigma| - \frac{1}{2} \sum_{i=1}^{n} (x_i-\mu)^T \Sigma^{-1}(x_i-\mu) \]$

### Bayesian Decision Theory
Bayesian decision theory involves finding the optimal decision boundary that minimizes the probability of misclassification. The decision boundary is determined by equating the posterior probabilities of the classes.

## Machine Learning Concepts
### Linear Discriminant Analysis (LDA)
LDA assumes that the class-conditional probabilities are Gaussian with class-specific means but a common covariance matrix. The decision rule is based on the posterior probabilities:
$\[ \delta_k(x) = x^T\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \log\pi_k \]$
where \(\pi_k\) is the prior probability of class k.

### Quadratic Discriminant Analysis (QDA)
QDA models each class with its own covariance matrix. The decision rule is:
$\[ \delta_k(x) = -\frac{1}{2}\log|\Sigma_k| - \frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k) + \log\pi_k \]$

### Support Vector Machines (SVM)
SVMs are supervised learning models used for classification and regression. They work by finding the hyperplane that best separates the classes in the feature space. The objective function is:
$\[ \min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \]$
subject to the constraints:
$\[ y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \]$

## Python Concepts
### Libraries Used
- **NumPy**: For numerical operations and handling arrays.
- **SciPy**: For statistical functions and operations.
- **Matplotlib**: For plotting and visualizing the results.
- **Pandas**: For data manipulation and handling.
- **scikit-learn**: For implementing machine learning algorithms such as SVM.

### Code Techniques
- **Vectorization**: Efficient computation using NumPy's vectorized operations.
- **Matrix Operations**: Inversion, multiplication, and eigen decomposition for covariance matrices.
- **Log-Likelihood Calculation**: Using log-likelihood for parameter estimation.

## Importance
### Significance in Machine Learning
Gaussian classifiers and SVMs are fundamental in statistical learning and provide robust approaches to classification. They are essential for understanding more complex models and are widely used in various applications, from image recognition to spam detection.

### Real-World Applications
- **Digit Recognition**: Used in automated mail sorting, banking, and other document processing tasks.
- **Spam Detection**: Critical for email filtering, improving communication efficiency, and cybersecurity.

## Outcomes
### Results Analysis
The project achieved high accuracy in digit recognition and spam detection tasks. Key metrics include:
- **Accuracy**: Percentage of correctly classified instances.
- **Confusion Matrix**: To visualize the performance of the classifier.
- **ROC Curve**: To evaluate the trade-off between true positive rate and false positive rate.

### Visualizations
- **Isocontours of Gaussian Distributions**: Plots showing the density function of Gaussian distributions.
- **Eigenvectors and Eigenvalues**: Visualizing the principal components of the data.
- **Decision Boundaries**: Illustrating the boundaries determined by LDA, QDA, and SVM classifiers.

## Conclusion
This project demonstrates the power and applicability of Gaussian classifiers and support vector machines in machine learning. By providing a probabilistic framework and robust classification techniques, these models offer reliable solutions for various classification problems.
