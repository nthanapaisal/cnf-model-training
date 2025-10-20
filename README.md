## Installation and set up: 
source ~/miniconda3/bin/activate &&
conda create -n ml python=3.11 &&
conda activate ml &&
conda install pandas &&
conda install numpy &&
conda install scikit-learn

## Run part 1-4 (DT,BDT,RF,GBoost) for CNF data
python3 main.py

## Run part 6 (model train for MNIST data)
python3 mnist.py

## Datasets:
15 CNF datasets with labels under ./all_data

clauses = ["300","500","1000","1500","1800"]

examples = ["100","1000","5000"]