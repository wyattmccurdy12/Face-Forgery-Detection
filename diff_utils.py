'''Utils for data manipulation'''
import csv

def load_csv(root_path, sub_path, from_subset_dir=True):
    '''Load a modified csv into a data loader as a list of rows'''
    numk = "10k" if root_path == "train" else "2k"
    if from_subset_dir:
        csv_path = f"subset_data/{root_path}/{sub_path}_{root_path}_{numk}_subset_update.csv"
    else:
        csv_path = f"{root_path}/{sub_path}_{root_path}_table_mod2.csv"

    print(f"Found {csv_path}. Loading data.")
    with open(csv_path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ")
        rows = list(csvreader)
    return rows        