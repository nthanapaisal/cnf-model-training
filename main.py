import json
from dt_classifier import train_dt 
from bagging_dt_classifier import train_bagging_dt 
from rand_forest_classifier import train_random_forest
from gradient_boost_classifier import train_gradient_boost
import pandas as pd 

# Part 1: DT
dt_report = train_dt()

# Part 2: Bagging DT
bagging_dt_report = train_bagging_dt()

# Part 3: Random Forest
random_forest_report = train_random_forest()

# Part 4: Gradient Boost
gradient_boost_report = train_gradient_boost()

print(dt_report)
print(bagging_dt_report)
print(random_forest_report)
print(gradient_boost_report)

# Combine Reports
final_report_full = {}
final_accuracy_report = {}
final_f1score_report = {}

models = ["dt", "bagging_dt", "random_forest", "gradient_boosting"]
reports = [dt_report, bagging_dt_report, random_forest_report, gradient_boost_report]

for i in range(len(models)):
    final_report_full[models[i]] = {f"c{k[0]}_d{k[1]}" : v for k,v in reports[i].items()}

    for k,v in reports[i].items():
        key = f"c{k[0]}_d{k[1]}"

        if key not in final_accuracy_report:
            final_accuracy_report[key] = []
        final_accuracy_report[key].append(v[-2])

        if key not in final_f1score_report:
            final_f1score_report[key] = []
        final_f1score_report[key].append(v[-1])

with open("./final_report_full.json", "w") as f:
    json.dump(final_report_full, f, indent=4)

acc_pdf = pd.DataFrame([[keyy] + vals for keyy, vals in final_accuracy_report.items()], columns=["Dataset", *models])
acc_pdf.to_csv("./final_accuracy_report.csv")
f1_pdf = pd.DataFrame([[keyy] + vals for keyy, vals in final_f1score_report.items()], columns=["Dataset", *models])
f1_pdf.to_csv("./final_f1score_report.csv")

print(final_report_full)
print(final_accuracy_report)
print(final_f1score_report)
