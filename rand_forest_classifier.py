import os 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_random_forest():
    clauses = ["300","500","1000","1500","1800"]
    examples = ["100","1000","5000"]

    #hyperparams
    criterions = ["entropy", "gini"]
    max_depths = {
        "100": [2,4,8,10],
        "1000": [5,10,15,20],
        "5000": [5,10,20,30],
    }
    n_estimators = [10,20,50]
    max_features = ["sqrt", "log2"]
    report = {}

    def read_data(file_path):
        if os.path.isfile(file_path):
            pdf = pd.read_csv(file_path, header=None)
            pdf_x = pdf.iloc[:,:-1]
            pdf_y = pdf.iloc[:,-1]
            return pdf_x, pdf_y

    def generate_report(c,e,best_crit,best_depth,best_n_estimator,best_feat,final_accuracy,final_f1):
        report[(c,e)] = [best_crit,best_depth,best_n_estimator,best_feat,final_accuracy,final_f1]


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
            best_crit, best_depth, best_n_estimator, best_feat = None, None, None, None
            max_acc = float('-inf')
            for crit in criterions:
                for depth in max_depths[e]:
                        for number_trees in n_estimators:
                            for max_feat in max_features:
                                print(f"Hypertuning DT for criterion: {crit}, depth: {depth}, n_estimator: {number_trees}, max_features: {max_feat}")
                                rand_forest = RandomForestClassifier(
                                    criterion=crit, 
                                    max_depth=depth,
                                    n_estimators=number_trees,
                                    max_features=max_feat,
                                    bootstrap=True,
                                    n_jobs=-1,
                                    random_state=42
                                )
                                rand_forest.fit(train_data_x, train_data_y)
                                predictions = rand_forest.predict(validate_data_x)

                                accuracy = accuracy_score(validate_data_y, predictions)

                                # update best hyper params
                                if accuracy > max_acc:
                                    max_acc = accuracy
                                    best_crit, best_depth, best_n_estimator, best_feat = crit, depth, number_trees, max_feat

            # Combine train and validate and retrain
            print(f"Best Params: criterion: {best_crit}, depth: {best_depth}, n_estimator: {best_n_estimator}, max_features: {best_feat}, accuracy: {max_acc}")
            train_valid_x = pd.concat([train_data_x, validate_data_x] ,ignore_index=True)
            train_valid_y = pd.concat([train_data_y, validate_data_y] ,ignore_index=True)
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
            final_predictions = rand_forest.predict(test_data_x)
            
            # Eval Metrics
            final_accuracy = accuracy_score(test_data_y, final_predictions)
            final_f1 = f1_score(test_data_y, final_predictions)

            print(f"Final Evaluations: accuracy:{final_accuracy}, f1: {final_f1}")
            generate_report(c,e,best_crit,best_depth,best_n_estimator,best_feat,final_accuracy,final_f1)
    return report

# {
#     ('300', '100'): ['entropy', 2, 50, 'sqrt', 0.755, 0.743455497382199], 
#     ('300', '1000'): ['entropy', 5, 50, 'sqrt', 0.8465, 0.8473396320238687], 
#     ('300', '5000'): ['gini', 5, 50, 'log2', 0.8709, 0.8726196349284657], 
#     ('500', '100'): ['entropy', 4, 50, 'sqrt', 0.86, 0.8556701030927835], 
#     ('500', '1000'): ['entropy', 5, 50, 'sqrt', 0.9095, 0.9100844510680576], 
#     ('500', '5000'): ['gini', 10, 50, 'sqrt', 0.9351, 0.9363538295577131], 
#     ('1000', '100'): ['entropy', 4, 50, 'log2', 0.99, 0.9900990099009901], 
#     ('1000', '1000'): ['entropy', 15, 50, 'sqrt', 0.98, 0.9799599198396793], 
#     ('1000', '5000'): ['gini', 10, 50, 'log2', 0.9936, 0.9936114993012577], 
#     ('1500', '100'): ['entropy', 2, 50, 'log2', 1.0, 1.0], 
#     ('1500', '1000'): ['entropy', 10, 50, 'sqrt', 1.0, 1.0], 
#     ('1500', '5000'): ['gini', 20, 50, 'log2', 1.0, 1.0], 
#     ('1800', '100'): ['entropy', 2, 50, 'sqrt', 1.0, 1.0], 
#     ('1800', '1000'): ['entropy', 5, 50, 'sqrt', 1.0, 1.0], 
#     ('1800', '5000'): ['entropy', 10, 20, 'log2', 0.9999, 0.9998999899989999]
# }

# criterions = ["entropy", "gini"]
# max_depths = {
#     "100": [2,4,8,10],
#     "1000": [5,10,15,20],
#     "5000": [5,10,20,30],
# }
# n_estimators = [10,20,50]
# max_features = ["sqrt", "log2"]


# criterions = ["entropy", "gini"]
# max_depths = {
#     "100": [2,4],
#     "1000": [5,10,15],
#     "5000": [5,10,20],
# }
# n_estimators = [20,50]
# max_features = ["sqrt", "log2"]