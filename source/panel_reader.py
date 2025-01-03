'''

panel_reader.py

This is a util tool for read csv with panel gene list

for example panel.csv is like:

panelA , Gene1 , Gene2 , Gene3
panelB , Gene4 , Gene5 ,

-----------------------------------------------------

which means  Gene1 , Gene2 , Gene3 was for panelA
and  Gene4 , Gene5 was for panelB

the functions as following can be used:

get_all_panels(): will return [ panelA , panelB]
get_panel(A): will return [Gene1 , Gene2 , Gene3]
get_panel(B): will return [Gene4 , Gene5]

<Single-cell Reactive Oxygen Species Regulome 
Profiling Reveals Dynamic Redox Regulation in Immune Cells>

Academia Sinica IBMS SYC`LAB
whuang022@gmail.com

'''
from argparse import ArgumentParser
import numpy as np
import pandas as pd


class PanelReader:
    '''
    padel reader from csv to list
    '''

    def __init__(self, path):
        self.__path = path
        self.__df = pd.read_csv(self.__path, header=None, index_col=0)

    def __remove_empty_from_list(self, list_input: list):
        '''
        drop np.nan from list
        '''
        return [i for i in list_input if i is not np.nan]

    def get_panel(self, panel_name):
        '''
        retrieve panel from CSV.
        '''
        if self.__df is not None:
            panel = self.__df
        else:
            raise FileNotFoundError
        try:
            panel = list(panel.loc[panel_name].values)
            return self.__remove_empty_from_list(panel)
        except KeyError as key_e:
            raise KeyError(f"Panel '{panel_name}' not found") from key_e
        except Exception as e:
            raise ValueError(
                f"An error occurred while retrieving panel '{panel_name}': {e}"
            ) from e

    def get_all_panels(self):
        '''
        return panel names
        '''
        return self.__df.index.to_list()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i",
                        "--input",
                        dest="input_csv",
                        required=True,
                        help="dataset (csv) file path.")
    parser.add_argument("-s",
                        "--select_panel",
                        dest="select_panel",
                        required=True,
                        nargs="+",
                        help="select panels")
    args = parser.parse_args()
    if args.input_csv:
        pr = PanelReader(args.input_csv)
        output = []
        if len(args.select_panel) > 0:
            for p in args.select_panel:
                output.extend(pr.get_panel(p))
            LIST_STR = ' '.join(str(x) for x in output)
            print(LIST_STR)
