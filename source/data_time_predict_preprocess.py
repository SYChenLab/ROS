'''
peprocress training data for data time predict

<Single-cell Reactive Oxygen Species Regulome 
Profiling Reveals Dynamic Redox Regulation in Immune Cells>

Academia Sinica IBMS SYC`LAB
whuang022@gmail.com
'''
import os
import pandas as pd


def main():
    '''
    main function 
    '''
    input_path = '../data_time_predict/raw_data_norm'
    output_path = '../data_time_predict/train_data'
    os.mkdir(output_path)
    # read three In vitro dataset
    in_vitro_df1 = pd.read_csv(f"{input_path}/In_vitro_1st.csv", index_col=0)
    in_vitro_df2 = pd.read_csv(f"{input_path}/In_vitro_2nd.csv", index_col=0)
    in_vitro_df3 = pd.read_csv(f"{input_path}/In_vitro_3rd.csv", index_col=0)
    # combine three dataset
    df = pd.concat([in_vitro_df1, in_vitro_df2, in_vitro_df3],
                   axis=0).reset_index().drop('index', axis=1)
    df.to_csv(f"{output_path}/MC38_In_vitro.csv", index=False)


if __name__ == '__main__':
    main()
