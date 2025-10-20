import os 
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_gradient_boost():
    clauses = ["300","500","1000","1500","1800"]
    examples = ["100","1000","5000"]

    #hyperparams
    criterions = ["friedman_mse", "squared_error"]
    max_depths = {
        "100": [2,4,8,10],
        "1000": [5,10,15,20],
        "5000": [5,10,20,30],
    }
    n_estimators = [10,20,50]
    learning_rates = [0.01, 0.05, 0.1]
    subsamples = [0.5, 1.0]
    report = {}

    def read_data(file_path):
        if os.path.isfile(file_path):
            pdf = pd.read_csv(file_path, header=None)
            pdf_x = pdf.iloc[:,:-1]
            pdf_y = pdf.iloc[:,-1]
            return pdf_x, pdf_y

    def generate_report(c,e,best_crit,best_depth,best_n_estimator,best_lr, best_subsample,final_accuracy,final_f1):
        report[(c,e)] = [best_crit,best_depth,best_n_estimator,best_lr, best_subsample,final_accuracy,final_f1]


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
            best_crit, best_depth, best_n_estimator, best_lr, best_subsample = None, None, None, None, None
            max_acc = float('-inf')
            for crit in criterions:
                for depth in max_depths[e]:
                    for number_trees in n_estimators:
                        for lr in learning_rates:
                            for subsample in subsamples:
                                print(f"Hypertuning for criterion: {crit}, depth: {depth}, n_estimator: {number_trees}, learning_rate: {lr}, subsample: {subsample}")
                                gradient = GradientBoostingClassifier(
                                    criterion=crit, 
                                    max_depth=depth,
                                    n_estimators=number_trees,
                                    learning_rate=lr,
                                    subsample=subsample,
                                    random_state=42
                                )
                                gradient.fit(train_data_x, train_data_y)
                                predictions = gradient.predict(validate_data_x)

                                accuracy = accuracy_score(validate_data_y, predictions)

                                # update best hyper params
                                if accuracy > max_acc:
                                    max_acc = accuracy
                                    best_crit, best_depth, best_n_estimator, best_lr, best_subsample = crit, depth, number_trees,  lr, subsample

            # Combine train and validate and retrain
            print(f"Best Params: criterion: {best_crit}, depth: {best_depth}, n_estimator: {best_n_estimator}, best_lr: {best_lr}, best_subsample: {best_subsample}, accuracy: {max_acc}")
            train_valid_x = pd.concat([train_data_x, validate_data_x] ,ignore_index=True)
            train_valid_y = pd.concat([train_data_y, validate_data_y] ,ignore_index=True)
            gradient = GradientBoostingClassifier(
                criterion=best_crit, 
                max_depth=best_depth,
                n_estimators=best_n_estimator,
                learning_rate=best_lr,
                subsample=best_subsample,
                random_state=42
            )
            gradient.fit(train_valid_x, train_valid_y)
            final_predictions = gradient.predict(test_data_x)
            
            # Eval Metrics
            final_accuracy = accuracy_score(test_data_y, final_predictions)
            final_f1 = f1_score(test_data_y, final_predictions)

            print(f"Final Evaluations: accuracy:{final_accuracy}, f1: {final_f1}")
            generate_report(c,e,best_crit,best_depth,best_n_estimator,best_lr, best_subsample,final_accuracy,final_f1)
    return report
