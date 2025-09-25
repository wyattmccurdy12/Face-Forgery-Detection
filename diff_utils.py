'''Utils for data manipulation'''
import csv

def load_csv(root_path, sub_path):
    '''Load a modified csv into a data loader as a list of rows'''
    csv_path = f"{root_path}/{sub_path}_{root_path}_table_mod2.csv"

    print(f"Found {csv_path}. Loading data.")
    with open(csv_path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ")
        rows = list(csvreader)
    return rows        