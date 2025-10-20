from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST dataset
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X / 255.0 # Normalize pixel values to [0,1]

# Split into training (60K) and test (10K) sets
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

reports = {}

# DT  ['entropy', 20, 'random', 0.9834, 0.9835153922542205]
clf = DecisionTreeClassifier(criterion="entropy", max_depth=20, splitter="random", random_state=42)
clf.fit(X_train, y_train)
final_predictions = clf.predict(X_test)
final_accuracy = accuracy_score(y_test, final_predictions)
final_f1 = f1_score(y_test, final_predictions, average="weighted")
reports["dt"] = (final_accuracy, final_f1)

# Bagging DT ['entropy', 20, 'random', 50, 1.0, 0.9977, 0.9976979281353218]
dtree = DecisionTreeClassifier(criterion="entropy", max_depth=20, splitter="random", random_state=42)
bagging_dtree = BaggingClassifier(
    estimator=dtree,
    n_estimators=50,
    max_samples=1.0,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
bagging_dtree.fit(X_train, y_train)
final_predictions = bagging_dtree.predict(X_test)
final_accuracy = accuracy_score(y_test, final_predictions)
final_f1 = f1_score(y_test, final_predictions, average="weighted")
reports["bagging_dt"] = (final_accuracy, final_f1)

# Random Forest ['entropy', 10, 20, 'log2', 0.9999, 0.9998999899989999]
rand_forest = RandomForestClassifier(
    criterion="entropy", 
    max_depth=10,
    n_estimators=20,
    max_features="log2",
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
rand_forest.fit(X_train, y_train)
final_predictions = rand_forest.predict(X_test)
final_accuracy = accuracy_score(y_test, final_predictions)
final_f1 = f1_score(y_test, final_predictions, average="weighted")
reports["random_forest"] = (final_accuracy, final_f1)

# Gradient Boosting ['friedman_mse', 5, 50, 0.1, 0.5, 0.9998, 0.9998]
gradient = GradientBoostingClassifier(
    criterion="friedman_mse", 
    max_depth=5,
    n_estimators=50,
    learning_rate=0.1,
    subsample=0.5,
    random_state=42
)
gradient.fit(X_train, y_train)
final_predictions = gradient.predict(X_test)
final_accuracy = accuracy_score(y_test, final_predictions)
final_f1 = f1_score(y_test, final_predictions, average="weighted")
reports["gradient_boosting"] = (final_accuracy, final_f1)

print(reports)

# {
# 'dt': (0.8834, 0.8833323376721032), 
# 'bagging_dt': (0.9684, 0.9683572194213158), 
# 'random_forest': (0.9339, 0.9337530089415542), 
# 'gradient_boosting': (0.9511, 0.9510907272201082)
# }