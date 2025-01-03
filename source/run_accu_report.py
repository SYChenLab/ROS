'''
concate accuracy report and sort with f1 score

Single-cell Reactive Oxygen Species Regulome 
Profiling Reveals Dynamic Redox Regulation in Immune Cells

Academia Sinica IBMS SYC`LAB
whuang022@gmail.com
'''
import os
from argparse import ArgumentParser
import pandas as pd

pd.set_option('display.precision', 2)


def make_report_summary(accu_fnames, report_name):
    '''
    make reports
    '''
    f1_all = []
    for accu_fname in accu_fnames:
        name = os.path.basename(accu_fname).split('_accu.csv')[0]
        try:
            df_accu = pd.read_csv(accu_fname, index_col=0)
            f1 = df_accu['f1-score'].rename(name + "_f1-score")
            f1_all.append(f1)
        except Exception as e:
            print(f"Error occurred: {e}")
    f1_df = pd.DataFrame(f1_all).sort_values(by="macro avg", ascending=False)
    f1_df.to_csv(report_name, float_format='%.4f')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="path to ml models accuracy (end with ..._accu.csv)")
    parser.add_argument("-o",
                        "--output",
                        dest="output",
                        required=True,
                        help="path file name to save result")
    args = parser.parse_args()

    accu_files = []
    for dirpath, dirnames, filenames in os.walk(args.input_path):
        for filename in filenames:
            if filename.endswith('_accu.csv'):
                accu_files.append(os.path.join(dirpath, filename))

    make_report_summary(accu_files, args.output)
