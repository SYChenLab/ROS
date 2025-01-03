'''
Plot distribution of feature group x prediction group
s

<Single-cell Reactive Oxygen Species Regulome 
Profiling Reveals Dynamic Redox Regulation in Immune Cells>

Academia Sinica IBMS SYC`LAB
whuang022@gmail.com
'''
import os
from argparse import ArgumentParser
from itertools import count
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
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
    parser.add_argument("-pre_out_col",
                        "--output_predict_col_name",
                        dest="output_predict_col_name",
                        required=True,
                        help="output_predict_col_name")
    parser.add_argument("-g",
                        "--group",
                        dest="group",
                        required=True,
                        help="feature for group to plot , must be str")
    args = parser.parse_args()
    # check output path
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    predict_key = args.output_predict_col_name
    f_name = os.path.basename(args.input_csv)
    name = f_name.split(".")[0]
    print(f"input file: {name}.csv")
    df_with_predict = pd.read_csv(args.input_csv, index_col=0)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))  #
    if args.group:
        df_grouped_list = df_with_predict.groupby(args.group)
        # compute counts
        out = {}
        for gname, group in df_grouped_list:
            bar_val = group[predict_key].value_counts()
            out[gname] = bar_val
            # plot KDE ans Bar plot
            kde_bar = group[predict_key].value_counts().sort_index()
            df_kde = pd.DataFrame(kde_bar)
            sns.kdeplot(data=df_kde,
                        x=df_kde.index,
                        weights=predict_key,
                        shade=True,
                        ax=ax1,
                        label=gname,
                        clip=(0, 5))
            sns.histplot(data=df_kde,
                         x=df_kde.index,
                         weights=predict_key,
                         kde=False,
                         discrete=True,
                         ax=ax2,
                         stat='probability',
                         label=gname + "frequence",
                         alpha=0.3,
                         bins=6)
        # plot bar plot
        ax1.legend()
        ax1.set_xlabel(f'{predict_key}')
        ax2.legend()
        ax2.set_xlabel(f'{predict_key}')
        plt.tight_layout()
        plt.savefig(f'{args.output_folder}/{name}{predict_key}.png')
        count = pd.DataFrame(out)
        print("counts:", count)
        count.to_csv(f"{args.output_folder}/{name}_count.csv")
        # plot bar plot
        count.plot(kind="bar")
        plt.title(name)
        plt.savefig(f"{args.output_folder}/{name}_count.png")
        # compute as rate
        for c in count.columns:
            count[c + "_rate"] = count[c] / count[c].sum()
            count.drop(c, axis=1, inplace=True)
        count.to_csv(f"{args.output_folder}/{name}_count_rate.csv")
        print("rates:", count)
        # plot bar plot
        count.plot(kind="bar")
        plt.title(name)
        plt.savefig(f"{args.output_folder}/{name}_count_rate.png")
