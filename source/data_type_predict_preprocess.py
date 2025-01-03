'''
peprocress training data for cell type predict

<Single-cell Reactive Oxygen Species Regulome 
Profiling Reveals Dynamic Redox Regulation in Immune Cells>

Academia Sinica IBMS SYC`LAB
whuang022@gmail.com
'''
import itertools
import os
import pandas as pd

output_path = "../data_type_predict/train_data"
input_path = "../data_type_predict/raw_data_norm"


def read_csv_concate(id_list):
    '''
    concate csv files
    '''
    df_list = []
    for id in id_list:
        df = pd.read_csv(f"{input_path}/donor_{id}_norm.csv",
                         index_col=0,
                         low_memory=False)
        df.dropna(inplace=True)
        df.set_index("donor_" + str(id) + "_" + df.index.astype(str),
                     inplace=True)
        df_list.append(df)
    df_all = pd.concat(df_list)
    return df_all


os.mkdir(output_path)
combines = list(itertools.combinations(range(10), 2))

for combine in combines:
    test_ids = set(combine)
    train_ids = set(range(10)) - test_ids
    FNAME = '_'.join(str(s) for s in test_ids)
    df_train = read_csv_concate(train_ids)
    df_test = read_csv_concate(test_ids)
    os.mkdir(f"{output_path}/{FNAME}")
    df_train.to_csv(f"{output_path}/{FNAME}/train_{FNAME}.csv")
    df_test.to_csv(f"{output_path}/{FNAME}/test_{FNAME}.csv")
