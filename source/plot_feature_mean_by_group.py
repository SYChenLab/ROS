'''
Plot mean value of selected feature with groups

<Single-cell Reactive Oxygen Species Regulome 
Profiling Reveals Dynamic Redox Regulation in Immune Cells>

Academia Sinica IBMS SYC`LAB
whuang022@gmail.com
'''
import os
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("-i",
                        "--input",
                        dest="input_csv",
                        required=True,
                        help="dataset (csv) file with prediction.")
    parser.add_argument("-o",
                        "--output",
                        dest="output_folder",
                        required=True,
                        help="output folder path.")
    
    parser.add_argument("-g",
                        "--group",
                        dest="group",
                        required=True,
                        help="feature for group to plot , must be str")
    parser.add_argument("-f",
                        "--features",
                        dest="features",
                        required=True,
                        type=str,
                        nargs='+',
                        help="A list of features for observe")
    args = parser.parse_args()
    # check output path
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    f_name = os.path.basename(args.input_csv)
    name = f_name.split(".")[0]
    print(f"input file: {name}.csv")
    df_with_predict=pd.read_csv(args.input_csv,index_col=0)
    predict_key = args.group
    df_grouped_list=df_with_predict.groupby(predict_key)
    out={}
    for gname, group in df_grouped_list:
        df_focus=df_grouped_list.get_group(gname)
        df_focus=df_focus[args.features]
        av_column = df_focus.mean(axis=0)
        out[gname]=av_column
        print (av_column)
    count=pd.DataFrame(out)

    print(count)
    count.to_csv(args.output_folder+'/'+f_name+"_add_predict_ex.csv")
    count.plot(kind="bar")
    plt.tight_layout()
    plt.savefig(args.output_folder+'/'+f_name+"_add_predict_ex.png")
