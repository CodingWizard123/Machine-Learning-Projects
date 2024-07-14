# Decision Tree and Random Forest Classifiers for Titanic and Iris Datasets

## Project Explanation

This project implements and evaluates Decision Tree and Random Forest classifiers on the Titanic and Iris datasets. The primary goal is to compare the performance of these classifiers and understand their strengths and weaknesses. The workflow involves data preprocessing, model training, evaluation, and result visualization.

## Mathematical Concepts

### Entropy and Information Gain
Entropy is a measure of the impurity in a dataset. It is calculated as:

$\[ H(y) = - \sum_{c} p(c) \log_2 p(c) \]$

where \( p(c) \) is the probability of class \( c \). Information gain measures the reduction in entropy after a dataset is split on a feature. It is given by:

$\[ IG(X, y) = H(y) - \sum_{i} \frac{|y_i|}{|y|} H(y_i) \]$

### Gini Impurity
Gini impurity is another measure of impurity, calculated as:

$\[ G(y) = 1 - \sum_{c} p(c)^2 \]$

### Decision Tree Splitting
A decision tree splits the data at each node based on the feature that provides the highest information gain or the lowest Gini impurity.

## Machine Learning Concepts

### Decision Tree Classifier
A Decision Tree is a non-parametric supervised learning method used for classification. It builds the tree by splitting the dataset into subsets based on the most significant feature at each node.

### Random Forest Classifier
A Random Forest is an ensemble method that constructs multiple decision trees and merges them to obtain a more accurate and stable prediction. Each tree is trained on a bootstrap sample of the data, and the final prediction is made by averaging the predictions of all trees.

## Python Concepts

### Libraries Used
- **NumPy**: For numerical operations.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For machine learning algorithms and evaluation metrics.
- **Matplotlib & Seaborn**: For data visualization.

### Key Functions and Methods
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling.
- **DecisionTree Class**: Custom implementation of the decision tree classifier.
- **RandomForest Class**: Custom implementation of the random forest classifier.
- **Model Evaluation**: Calculating accuracy and visualizing decision boundaries.

## Importance

This project demonstrates the implementation and application of fundamental machine learning techniques like decision trees and random forests. These classifiers are widely used in various fields such as finance, healthcare, and marketing for tasks like risk assessment, diagnosis, and customer segmentation. Understanding these models' working principles and performance helps build robust predictive systems.

## Outcomes

### Results
The classifiers were trained and evaluated on the Titanic and Iris datasets. Key results include:
- **Decision Tree Accuracy**: Demonstrated on both datasets with varying depths and splitting criteria.
- **Random Forest Accuracy**: Showed improved performance over individual decision trees due to the ensemble approach.
- **Feature Importance**: Analyzed to understand which features contribute most to the model's predictions.

### Visualizations
- **Decision Boundaries**: Plots showing how the classifiers split the feature space.
- **Accuracy Comparison**: Bar charts comparing the accuracy of different models.

### Conclusion
The project highlights the strengths of ensemble methods like random forests in providing more accurate and stable predictions compared to single decision trees. The analysis and visualizations provide valuable insights into model performance and feature importance, making this project a significant learning resource for anyone interested in machine learning classifiers.
