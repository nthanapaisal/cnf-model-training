import os 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_bagging_dt():
    clauses = ["300","500","1000","1500","1800"]
    examples = ["100","1000","5000"]

    #hyperparams
    criterions = ["entropy", "gini"]
    splitters = ["best","random"]
    max_depths = {
        "100": [2,4,8,10],
        "1000": [5,10,15,20],
        "5000": [5,10,20,30],
    }
    n_estimators = [10,20,50]
    max_samples = [0.1,0.5,0.75,1.0]
    report = {}

    def read_data(file_path):
        if os.path.isfile(file_path):
            pdf = pd.read_csv(file_path, header=None)
            pdf_x = pdf.iloc[:,:-1]
            pdf_y = pdf.iloc[:,-1]
            return pdf_x, pdf_y

    def generate_report(c,e,best_crit,best_depth,best_split,best_n_estimator,best_max_sample,final_accuracy,final_f1):
        report[(c,e)] = [best_crit,best_depth,best_split,best_n_estimator,best_max_sample,final_accuracy,final_f1]


    for c in clauses:
        for e in examples:
            train_path = f"./all_data/train_c{c}_d{e}.csv"
            validate_path = f"./all_data/valid_c{c}_d{e}.csv"
            test_path = f"./all_data/test_c{c}_d{e}.csv"

            # read data
            train_data_x, train_data_y = read_data(train_path)
            validate_data_x, validate_data_y = read_data(validate_path)
            test_data_x, test_data_y = read_data(test_path)

            # train with train data and use validate to tune
            print("======================")
            print(f"Training Dataset: clauses: {c}, examples: {e}")
            best_crit, best_depth, best_split, best_n_estimator, best_max_sample = None, None, None, None, None
            max_acc = float('-inf')
            for crit in criterions:
                for depth in max_depths[e]:
                    for split in splitters:
                        for number_trees in n_estimators:
                            for sample_frac in max_samples:
                                print(f"Hypertuning DT for criterion: {crit}, depth: {depth}, split: {split}, n_estimator: {number_trees}, max_samples: {sample_frac}")
                                dtree = DecisionTreeClassifier(criterion=crit, max_depth=depth, splitter=split, random_state=42)
                                bagging_dtree = BaggingClassifier(
                                    estimator=dtree,
                                    n_estimators=number_trees,
                                    max_samples=sample_frac,
                                    bootstrap=True,
                                    n_jobs=-1,
                                    random_state=42
                                )
                                bagging_dtree.fit(train_data_x, train_data_y)
                                predictions = bagging_dtree.predict(validate_data_x)

                                accuracy = accuracy_score(validate_data_y, predictions)

                                # update best hyper params
                                if accuracy > max_acc:
                                    max_acc = accuracy
                                    best_crit, best_depth, best_split, best_n_estimator, best_max_sample = crit, depth, split, number_trees, sample_frac

            # Combine train and validate and retrain
            print(f"Best Params: criterion: {best_crit}, depth: {best_depth}, split: {best_split}, n_estimator: {best_n_estimator}, max_samples: {best_max_sample}, accuracy: {max_acc}")
            train_valid_x = pd.concat([train_data_x, validate_data_x] ,ignore_index=True)
            train_valid_y = pd.concat([train_data_y, validate_data_y] ,ignore_index=True)
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
            final_predictions = bagging_dtree.predict(test_data_x)
            
            # Eval Metrics
            final_accuracy = accuracy_score(test_data_y, final_predictions)
            final_f1 = f1_score(test_data_y, final_predictions)

            print(f"Final Evaluations: accuracy:{final_accuracy}, f1: {final_f1}")
            generate_report(c,e,best_crit,best_depth,best_split,best_n_estimator,best_max_sample,final_accuracy,final_f1)
    return report

# {
#     ('300', '100'): ['entropy', 10, 'best', 50, 0.5, 0.755, 0.7487179487179487], 
#     ('300', '1000'): ['entropy', 10, 'best', 50, 1.0, 0.8605, 0.8628992628992629], 
#     ('300', '5000'): ['gini', 30, 'random', 50, 1.0, 0.9134, 0.9175080967803391], 
#     ('500', '100'): ['entropy', 10, 'random', 50, 1.0, 0.82, 0.82], 
#     ('500', '1000'): ['gini', 10, 'random', 50, 1.0, 0.8795, 0.8802781917536016], 
#     ('500', '5000'): ['gini', 30, 'best', 50, 1.0, 0.9381, 0.9381680151832984], 
#     ('1000', '100'): ['gini', 4, 'best', 50, 0.1, 0.89, 0.8921568627450981], 
#     ('1000', '1000'): ['entropy', 5, 'random', 50, 0.1, 0.919, 0.9227836034318398], 
#     ('1000', '5000'): ['entropy', 30, 'best', 50, 0.5, 0.9578, 0.9576645264847512], 
#     ('1500', '100'): ['entropy', 2, 'best', 50, 0.1, 0.995, 0.9950248756218906], 
#     ('1500', '1000'): ['entropy', 10, 'best', 50, 0.1, 0.983, 0.9829488465396189], 
#     ('1500', '5000'): ['gini', 20, 'random', 50, 0.5, 0.9902, 0.9901803607214429], 
#     ('1800', '100'): ['entropy', 2, 'random', 50, 0.1, 0.995, 0.9950248756218906], 
#     ('1800', '1000'): ['entropy', 20, 'random', 50, 0.1, 0.9945, 0.9944972486243121], 
#     ('1800', '5000'): ['entropy', 30, 'random', 20, 1.0, 0.9973, 0.9972997299729973]
# }

# max_depths = {
#     "100": [2,4,8,10],
#     "1000": [5,10,15,20],
#     "5000": [5,10,20,30],
# }
# n_estimators = [10,20,50]
# max_samples = [0.1,0.5,0.75,1.0]


# max_depths = {
#     "100": [2,4,10],
#     "1000": [5,10,20],
#     "5000": [20,30],
# }
# n_estimators = [20,50]
# max_samples = [0.1,0.5,1.0]