'''
count mean/std of train
<Single-cell Reactive Oxygen Species Regulome 
Profiling Reveals Dynamic Redox Regulation in Immune Cells>

Academia Sinica IBMS SYC`LAB
whuang022@gmail.com
'''
import os
import pandas as pd

pd.set_option("display.precision", 4)
output_dir = '../data_type_predict/ml_models/catboost'
df_list=[]
for folder_name in os.listdir(output_dir):
    if folder_name.startswith("train"):
        folder_path = os.path.join(output_dir, folder_name)
        cat3_accu_report_path = os.path.join(folder_path, "./CatBoostClassifier/CatBoostClassifier_accu.csv")
        if os.path.exists(cat3_accu_report_path):
            df = pd.read_csv(cat3_accu_report_path,index_col=0)
            df_list.append(df)

merged_df = pd.concat(df_list, axis=0) 
grouped = merged_df.groupby(merged_df.index)
mean_values = grouped.mean()
std_values = grouped.std() 

print("\nmean:")
print(mean_values)
print("\nstd:")
print(std_values)
