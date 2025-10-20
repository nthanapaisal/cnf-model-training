from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST dataset
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X / 255.0 # Normalize pixel values to [0,1]

# Split into training (60K) and test (10K) sets
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# reports = {}

# # DT  ['entropy', 20, 'random', 0.9834, 0.9835153922542205]
# clf = DecisionTreeClassifier(criterion="entropy", max_depth=20, splitter="random", random_state=42)
# clf.fit(X_train, y_train)
# final_predictions = clf.predict(X_test)
# final_accuracy = accuracy_score(y_test, final_predictions)
# final_f1 = f1_score(y_test, final_predictions, average="weighted")
# reports["dt"] = (final_accuracy, final_f1)

# # Bagging DT ['entropy', 20, 'random', 50, 1.0, 0.9977, 0.9976979281353218]
# dtree = DecisionTreeClassifier(criterion="entropy", max_depth=20, splitter="random", random_state=42)
# bagging_dtree = BaggingClassifier(
#     estimator=dtree,
#     n_estimators=50,
#     max_samples=1.0,
#     bootstrap=True,
#     n_jobs=-1,
#     random_state=42
# )
# bagging_dtree.fit(X_train, y_train)
# final_predictions = bagging_dtree.predict(X_test)
# final_accuracy = accuracy_score(y_test, final_predictions)
# final_f1 = f1_score(y_test, final_predictions, average="weighted")
# reports["bagging_dt"] = (final_accuracy, final_f1)

# # Random Forest ['entropy', 10, 20, 'log2', 0.9999, 0.9998999899989999]
# rand_forest = RandomForestClassifier(
#     criterion="entropy", 
#     max_depth=10,
#     n_estimators=20,
#     max_features="log2",
#     bootstrap=True,
#     n_jobs=-1,
#     random_state=42
# )
# rand_forest.fit(X_train, y_train)
# final_predictions = rand_forest.predict(X_test)
# final_accuracy = accuracy_score(y_test, final_predictions)
# final_f1 = f1_score(y_test, final_predictions, average="weighted")
# reports["random_forest"] = (final_accuracy, final_f1)

# # Gradient Boosting ['friedman_mse', 5, 50, 0.1, 0.5, 0.9998, 0.9998]
# gradient = GradientBoostingClassifier(
#     criterion="friedman_mse", 
#     max_depth=5,
#     n_estimators=50,
#     learning_rate=0.1,
#     subsample=0.5,
#     random_state=42
# )
# gradient.fit(X_train, y_train)
# final_predictions = gradient.predict(X_test)
# final_accuracy = accuracy_score(y_test, final_predictions)
# final_f1 = f1_score(y_test, final_predictions, average="weighted")
# reports["gradient_boosting"] = (final_accuracy, final_f1)

# print(reports)

# {
# 'dt': (0.8834, 0.8833323376721032), 
# 'bagging_dt': (0.9684, 0.9683572194213158), 
# 'random_forest': (0.9339, 0.9337530089415542), 
# 'gradient_boosting': (0.9511, 0.9510907272201082)
# }
############################################################################################################################
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)
# tune only with 10,000
X_small, _, y_small, _ = train_test_split(
    X_train, y_train, train_size=10000, stratify=y_train, random_state=42
)
X_train = np.array(X_train)
X_valid = np.array(X_valid)
X_test = np.array(X_test)
X_small = np.array(X_small)
############################################################################################################################
# hyper tune DT
# DT  ['entropy', 20, 'random', 0.9834, 0.9835153922542205]
reports_tuned = {}
best_depth = None
max_acc = float('-inf')
for depth in [10,20,30]:
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=depth, splitter="random", random_state=42)
    clf.fit(X_small, y_small)
    predictions = clf.predict(X_valid)

    accuracy = accuracy_score(y_valid, predictions)

    # update best hyper params
    if accuracy > max_acc:
        max_acc = accuracy
        best_crit, best_depth, best_split = "entropy", depth, "random"

# Combine train and validate and retrain
print(f"Best Params: criterion: {best_crit}, depth: {best_depth}, split: {best_split}, accuracy: {max_acc}")
train_valid_x = np.concatenate([X_train, X_valid] , axis=0)
train_valid_y = np.concatenate([y_train, y_valid] , axis=0)
clf = DecisionTreeClassifier(criterion=best_crit, max_depth=best_depth, splitter=best_split, random_state=42)
clf.fit(train_valid_x, train_valid_y)
final_predictions = clf.predict(X_test)

final_accuracy = accuracy_score(y_test, final_predictions)
final_f1 = f1_score(y_test, final_predictions, average="weighted")

reports_tuned["dt"] = [best_crit,best_depth,best_split,final_accuracy,final_f1]

############################################################################################################################
# hyper tune BDT: 
# Bagging DT ['entropy', 20, 'random', 50, 1.0, 0.9977, 0.9976979281353218]
best_depth, best_n_estimator, best_max_sample = None, None, None
max_acc = float('-inf')
for depth in [10,20,30]:
    for number_trees in [50,80]:
        for sample_frac in [0.1,0.5]:
            dtree = DecisionTreeClassifier(criterion="entropy", max_depth=depth, splitter="random", random_state=42)
            bagging_dtree = BaggingClassifier(
                estimator=dtree,
                n_estimators=number_trees,
                max_samples=sample_frac,
                bootstrap=True,
                n_jobs=-1,
                random_state=42
            )
            bagging_dtree.fit(X_small, y_small)
            predictions = bagging_dtree.predict(X_valid)

            accuracy = accuracy_score(y_valid, predictions)

            # update best hyper params
            if accuracy > max_acc:
                max_acc = accuracy
                best_crit, best_depth, best_split, best_n_estimator, best_max_sample = "entropy", depth, "random", number_trees, sample_frac

# Combine train and validate and retrain
print(f"Best Params: criterion: {best_crit}, depth: {best_depth}, split: {best_split}, n_estimator: {best_n_estimator}, max_samples: {best_max_sample}, accuracy: {max_acc}")
train_valid_x = np.concatenate([X_train, X_valid] , axis=0)
train_valid_y = np.concatenate([y_train, y_valid] , axis=0)
dtree = DecisionTreeClassifier(criterion=best_crit, max_depth=best_depth, splitter=best_split, random_state=42)
bagging_dtree = BaggingClassifier(
                        estimator=dtree,
                        n_estimators=best_n_estimator,
                        max_samples=best_max_sample,
                        bootstrap=True,
                        n_jobs=-1,
                        random_state=42
                    )
bagging_dtree.fit(train_valid_x, train_valid_y)
final_predictions = bagging_dtree.predict(X_test)

final_accuracy = accuracy_score(y_test, final_predictions)
final_f1 = f1_score(y_test, final_predictions, average="weighted")

reports_tuned["bagging_dt"] = [best_crit,best_depth,best_split,best_n_estimator,best_max_sample,final_accuracy,final_f1]

############################################################################################################################
# hyper tune randomforest
# Random Forest ['entropy', 10, 20, 'log2', 0.9999, 0.9998999899989999]
best_crit, best_depth, best_n_estimator, best_feat = None, None, None, None
max_acc = float('-inf')
for depth in [10,20]:
    for number_trees in [50,100]:
        rand_forest = RandomForestClassifier(
            criterion="entropy", 
            max_depth=depth,
            n_estimators=number_trees,
            max_features="log2",
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        )
        rand_forest.fit(X_small, y_small)
        predictions = rand_forest.predict(X_valid)

        accuracy = accuracy_score(y_valid, predictions)

        # update best hyper params
        if accuracy > max_acc:
            max_acc = accuracy
            best_crit, best_depth, best_n_estimator, best_feat = "entropy", depth, number_trees, "log2"

# Combine train and validate and retrain
print(f"Best Params: criterion: {best_crit}, depth: {best_depth}, n_estimator: {best_n_estimator}, max_features: {best_feat}, accuracy: {max_acc}")
train_valid_x = np.concatenate([X_train, X_valid] , axis=0)
train_valid_y = np.concatenate([y_train, y_valid] , axis=0)
rand_forest = RandomForestClassifier(
    criterion=best_crit, 
    max_depth=best_depth,
    n_estimators=best_n_estimator,
    max_features=best_feat,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
rand_forest.fit(train_valid_x, train_valid_y)
final_predictions = rand_forest.predict(X_test)

final_accuracy = accuracy_score(y_test, final_predictions)
final_f1 = f1_score(y_test, final_predictions, average="weighted")

reports_tuned["random_forest"] = [best_crit,best_depth,best_n_estimator,best_feat,final_accuracy,final_f1]

############################################################################################################################
# hyper tune gboost
# Gradient Boosting ['friedman_mse', 5, 50, 0.1, 0.5, 0.9998, 0.9998]

best_crit, best_depth, best_n_estimator, best_lr, best_subsample = None, None, None, None, None
max_acc = float('-inf')
for depth in [3,5]:
    for number_trees in [50,100]:
        for lr in [0.1, 0.2]:
            for subsample in [0.5]:
                gradient = GradientBoostingClassifier(
                    criterion="friedman_mse", 
                    max_depth=depth,
                    n_estimators=number_trees,
                    learning_rate=lr,
                    subsample=subsample,
                    random_state=42
                )
                gradient.fit(X_small, y_small)
                predictions = gradient.predict(X_valid)

                accuracy = accuracy_score(y_valid, predictions)

                # update best hyper params
                if accuracy > max_acc:
                    max_acc = accuracy
                    best_crit, best_depth, best_n_estimator, best_lr, best_subsample = "friedman_mse", depth, number_trees,  lr, subsample

# Combine train and validate and retrain
print(f"Best Params: criterion: {best_crit}, depth: {best_depth}, n_estimator: {best_n_estimator}, best_lr: {best_lr}, best_subsample: {best_subsample}, accuracy: {max_acc}")
train_valid_x = np.concatenate([X_train, X_valid] , axis=0)
train_valid_y = np.concatenate([y_train, y_valid] , axis=0)
gradient = GradientBoostingClassifier(
    criterion=best_crit, 
    max_depth=best_depth,
    n_estimators=best_n_estimator,
    learning_rate=best_lr,
    subsample=best_subsample,
    random_state=42
)
gradient.fit(train_valid_x, train_valid_y)
final_predictions = gradient.predict(X_test)

final_accuracy = accuracy_score(y_test, final_predictions)
final_f1 = f1_score(y_test, final_predictions, average="weighted")

reports_tuned["gradient_boosting"] = [best_crit,best_depth,best_n_estimator,best_lr, best_subsample,final_accuracy,final_f1]

print(reports_tuned)