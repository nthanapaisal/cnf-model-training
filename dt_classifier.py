import os 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_dt():
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

    report = {}

    def read_data(file_path):
        if os.path.isfile(file_path):
            pdf = pd.read_csv(file_path, header=None)
            pdf_x = pdf.iloc[:,:-1]
            pdf_y = pdf.iloc[:,-1]
            return pdf_x, pdf_y

    def generate_report(c,e,best_crit,best_depth,best_split,final_accuracy,final_f1):
        report[(c,e)] = [best_crit,best_depth,best_split,final_accuracy,final_f1]


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
            best_crit, best_depth, best_split = None, None, None
            max_acc = float('-inf')
            for crit in criterions:
                for depth in max_depths[e]:
                    for split in splitters:
                        print(f"Hypertuning DT for criterion: {crit}, depth: {depth}, split: {split}")
                        clf = DecisionTreeClassifier(criterion=crit, max_depth=depth, splitter=split, random_state=42)
                        clf.fit(train_data_x, train_data_y)
                        predictions = clf.predict(validate_data_x)

                        accuracy = accuracy_score(validate_data_y, predictions)

                        # update best hyper params
                        if accuracy > max_acc:
                            max_acc = accuracy
                            best_crit, best_depth, best_split = crit, depth, split

            # Combine train and validate and retrain
            print(f"Best Params: criterion: {best_crit}, depth: {best_depth}, split: {best_split}, accuracy: {max_acc}")
            train_valid_x = pd.concat([train_data_x, validate_data_x] ,ignore_index=True)
            train_valid_y = pd.concat([train_data_y, validate_data_y] ,ignore_index=True)
            clf = DecisionTreeClassifier(criterion=best_crit, max_depth=best_depth, splitter=best_split, random_state=42)
            clf.fit(train_valid_x, train_valid_y)
            final_predictions = clf.predict(test_data_x)
            
            # Eval Metrics
            final_accuracy = accuracy_score(test_data_y, final_predictions)
            final_f1 = f1_score(test_data_y, final_predictions)

            print(f"Final Evaluations: accuracy:{final_accuracy}, f1: {final_f1}")
            generate_report(c,e,best_crit,best_depth,best_split,final_accuracy,final_f1)
    return report

# {
#     ('300', '100'): ['gini', 4, 'best', 0.685, 0.7123287671232876], 
#     ('300', '1000'): ['entropy', 5, 'best', 0.6755, 0.7106553722692822], 
#     ('300', '5000'): ['gini', 10, 'random', 0.7827, 0.7903116858052688], 
#     ('500', '100'): ['entropy', 4, 'random', 0.695, 0.6965174129353234], 
#     ('500', '1000'): ['entropy', 5, 'best', 0.682, 0.687007874015748], 
#     ('500', '5000'): ['entropy', 10, 'best', 0.789, 0.79889439573008], 
#     ('1000', '100'): ['gini', 4, 'random', 0.73, 0.7476635514018691], 
#     ('1000', '1000'): ['entropy', 10, 'best', 0.798, 0.811214953271028], 
#     ('1000', '5000'): ['entropy', 10, 'best', 0.8557, 0.8616888718489408], 
#     ('1500', '100'): ['entropy', 2, 'best', 0.84, 0.8545454545454545], 
#     ('1500', '1000'): ['gini', 10, 'random', 0.912, 0.9143135345666992], 
#     ('1500', '5000'): ['entropy', 10, 'random', 0.9559, 0.9562976910117927], 
#     ('1800', '100'): ['entropy', 8, 'best', 0.92, 0.9238095238095239], 
#     ('1800', '1000'): ['entropy', 20, 'random', 0.9755, 0.9757545769421079], 
#     ('1800', '5000'): ['entropy', 30, 'random', 0.9843, 0.9844137794103047]
# }

# criterions = ["entropy", "gini"]
# splitters = ["best","random"]
# max_depths = {
# "100": [2,4,8,10],
# "1000": [5,10,15,20],
# "5000": [5,10,20,30],
# }

# criterions = ["entropy", "gini"]
# splitters = ["best","random"]
# max_depths = {
# "100": [2,4,8]
# "1000": [5,10,20]
# "5000": [10,30]
# }