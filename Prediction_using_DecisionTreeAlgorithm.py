# PREDICTION USING DECISION TREE ALGORITHM

import sklearn.datasets as datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the iris dataset
iris = datasets.load_iris()

# Create a DataFrame for the iris dataset
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head(5))

# Assigning the target variable
y = iris.target
print(y)

# Import the Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree Classifier instance
dtree = DecisionTreeClassifier()

# Train the Decision Tree Classifier on the dataset
dtree.fit(df, y)

print('Decision Tree Classifier Created')

# Convert target names to a list
class_names = list(iris.target_names)

# Customize the appearance of the Decision Tree plot
plt.figure(figsize=(20, 12))
plot_tree(dtree, feature_names=iris.feature_names, class_names=class_names,
          filled=True, rounded=True, fontsize=12, impurity=False, precision=2,
          proportion=True, node_ids=True, ax=plt.gca())
plt.title("Decision Tree Visualization", fontsize=18)
plt.show()

# Now you can use the trained classifier to make predictions on new data
# For example:
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Replace with your own data
prediction = dtree.predict(new_data)
print("Predicted class:", prediction)