'''
A Wrapper for train ml classifier

<Single-cell Reactive Oxygen Species Regulome 
Profiling Reveals Dynamic Redox Regulation in Immune Cells>

Academia Sinica IBMS SYC`LAB
whuang022@gmail.com
'''
import os
import sys
import time
import datetime
import importlib
import multiprocessing
import json
import warnings
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import shap
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

warnings.simplefilter(action='ignore', category=FutureWarning)
model_roc_curve_postfix = "_roc.png"
model_accurancy_postfix = "_accu.csv"
model_joblib_postfix = "_model_joblib.data"
model_fr = "_feature_importance"


class Classifier:
    '''
    Classifier model class
    '''

    def __init__(self,
                 name=None,
                 model_name=None,
                 module_name=None,
                 model_params=None,
                 model_file=None):
        '''
        initialize the classifier
        '''
        if model_file:
            # load the model and set properties dynamically
            self.model = joblib.load(model_file)

            # dynamically set properties
            self.__name = getattr(self.model, 'name',
                                  None) or f"{os.path.basename(model_file)}"
            self.__model_name = type(self.model).__name__
            self.__module_name = type(self.model).__module__
            self.__model_params = self.model.get_params()
        else:
            # initialize
            self.__name = name
            self.__model_name = model_name
            self.__module_name = module_name
            self.__model_params = model_params
            self.model = self.__new_ml_model(self.__model_name,
                                             self.__module_name,
                                             self.__model_params)

    def __new_ml_model(self, ml_model_name, ml_module_name, params):
        '''
        Dynamically import the machine learning class and instantiate it with params.
        '''
        try:
            module = importlib.import_module(ml_module_name)
            model_class = getattr(module, ml_model_name)
            return model_class(**params)
        except ImportError as e:
            raise ImportError(
                f"Error importing module '{ml_module_name}': {e}") from e
        except AttributeError as e:
            raise AttributeError(
                f"Model '{ml_model_name}' not found in '{ml_module_name}'."
            ) from e

    def fit(self, *args, **kwargs):
        '''
        Wrapper for model fit
        '''
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        '''
        Wrapper for model predict
        '''
        return self.model.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        '''
        Wrapper for model predict_proba
        '''
        return self.model.predict_proba(*args, **kwargs)

    def __repr__(self):
        '''
        Return a string representation of the classifier
        '''
        return f"{self.__name}"

    @property
    def class_names(self):
        '''
        Return class names
        '''
        if hasattr(self.model, 'classes_'):
            return self.model.classes_
        elif hasattr(self.model, 'get_booster'):
            # If model is XGBoost
            return self.model.get_booster().get_dump()[0]
        return []

    @property
    def name(self):
        '''
        Return the name of the classifier
        '''
        return self.__name

    @property
    def model_name(self):
        '''
        Return the model name
        '''
        return self.__model_name

    @property
    def module_name(self):
        '''
        Return the module name
        '''
        return self.__module_name

    @property
    def model_params(self):
        '''
        Return model parameters
        '''
        return self.__model_params


def get_core_numbers(parallel, core_used):
    '''
    return the correct cpu number to use
    '''
    core_to_use = 1
    if core_used:
        try:
            if parallel:
                core_to_use = int(core_used)
            else:
                core_to_use = 1
        except ValueError:
            if parallel:
                core_to_use = multiprocessing.cpu_count()
            else:
                core_to_use = 1
    else:
        if parallel:
            core_to_use = multiprocessing.cpu_count()
        else:
            core_to_use = 1
    if core_to_use <= 0:
        if parallel:
            core_to_use = multiprocessing.cpu_count()
        else:
            core_to_use = 1
    return core_to_use


def save_args_to_file(run_args, run_json, output_folder, file_name=None):
    '''
    save log run args to file
    '''
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if file_name is None:
        file_name = f"{output_folder}/run_{timestamp}_log.txt"
    with open(file_name, "a", encoding='utf-8') as f:
        f.write(f"run time : {datetime.datetime.now()}\n")
        for arg_tmp, val in vars(run_args).items():
            f.write(f"{arg_tmp}: {val}\n")
        json_str = json.dumps(run_json, indent=4)
        f.write("-------------input JSON setting-------------\n")
        f.write(json_str)


def create_ml_models(json_input):
    '''
    create ml classifiers from json config
    json_input : json filepath or json string
    '''
    classifiers = []
    # if json_input is json file path
    if os.path.isfile(json_input):
        with open(json_input, 'r', encoding='utf-8') as f:
            json_conf = json.load(f)
    else:
        # maybe if json_input is a json string
        try:
            json_conf = json.loads(json_input)
        except ValueError as e:
            print(f"Error with {e}")
            raise
    # if json obj, then wrap to json array
    if isinstance(json_conf, dict):
        json_conf = [json_conf]
    # new all classifier and return
    for classifier in json_conf:
        try:
            classifier = Classifier(classifier['name'], classifier['model'],
                                    classifier['module'], classifier['params'])
            classifiers.append(classifier)
        except:
            print(f'{classifier} not load ')
            raise
    return classifiers, json_conf


def get_feature_importance(classifier: Classifier,
                           feature_importance,
                           output_folder,
                           x_train,
                           x_test,
                           y_train,
                           y_test,
                           random_seed=666):
    '''
    get SHAP value or Permutation
    '''
    _ = y_train
    # use SHAP value to count feature importance
    if feature_importance == 'SHAP':
        shap_val_lst = []
        shap_val_cls = []
        sub_title = ""
        # model TreeExplainer
        tree_models = [
            "DecisionTreeClassifier", "RandomForestClassifier",
            "XGBClassifier", "CatBoostClassifier"
        ]
        if classifier.model_name in tree_models:
            explainer = shap.TreeExplainer(classifier.model)
            sub_title = "Tree Explainer"
        # others use KernelExplainer
        else:
            explainer = shap.KernelExplainer(classifier.model.predict_proba,
                                             x_train)
            sub_title = "Kernel Explainer"
        # compute SHAP value
        shap_values = explainer.shap_values(x_test)
        # collect SHAP
        for i, class_name in enumerate(classifier.class_names):
            shap_val = np.take(shap_values, i, axis=2)
            shap_val_lst.append(shap_val)
            shap_val_cls.append(class_name)
        # plot bar plot
        shap.summary_plot(shap_val_lst,
                          x_test,
                          plot_type="bar",
                          show=False,
                          class_names=shap_val_cls)
        plt.suptitle(f"SHAP Importance for {classifier.name}",
                     fontsize=24,
                     y=1)
        plt.title(sub_title, fontsize=16)
        plt.tight_layout()
        fig_path = f'{output_folder}/{ classifier.name}/{ classifier.name}{model_fr}.png'
        plt.savefig(fig_path, dpi=600)
        plt.close()
        return fig_path

    # use Permutation to count feature importance
    if feature_importance == 'Permutation':
        # compute permutation importance
        result = permutation_importance(classifier.model,
                                        x_test,
                                        y_test,
                                        n_repeats=2,
                                        random_state=random_seed,
                                        n_jobs=-1)
        importance = result.importances_mean
        sorted_idx = np.argsort(importance)[::-1]
        # plot bar plot
        plt.barh(range(len(importance)),
                 importance[sorted_idx],
                 align='center',
                 color='skyblue')
        plt.yticks(range(len(importance)),
                   np.array(x_train.columns)[sorted_idx],
                   fontsize=5)
        plt.xlabel("Permutation Importance")
        plt.title(f"Permutation Importance for {classifier.name}")
        plt.tight_layout()
        fig_path = f'{output_folder}/{classifier.name}/{classifier.name}{model_fr}.png'
        plt.savefig(fig_path, dpi=600)
        plt.close()
        return fig_path

    return f"Not Support {feature_importance}"


def test_eval_model(model_file_path,
                    test_file,
                    features,
                    input_index_col,
                    output_predict_col_name,
                    output_folder,
                    y_name=None):
    '''
    test the model, generate predictions, and save evaluation results.
    '''
    if y_name:
        cols = features + [y_name]
    else:
        cols = features
    test_df = pd.read_csv(test_file, index_col=input_index_col)
    org = pd.read_csv(test_file, index_col=input_index_col)
    # select features using the config file and panel
    if len(cols) > 0:
        test_df = test_df[cols]
    # select column without y_name for features
    if y_name:
        x = test_df.drop(columns=[y_name])
        y_ground_truth = test_df[y_name].values
    else:
        x = test_df.values
    # load the pre-trained  model
    classifier = Classifier(model_file=model_file_path)
    # make predictions
    y_predict = classifier.predict(x)
    #print(y_predict)
    y_probas = classifier.predict_proba(x)
    if y_name:
        # output classification report
        print(classification_report(y_ground_truth, y_predict))
        report = classification_report(y_ground_truth,
                                       y_predict,
                                       output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        # save accuracy report
        report_df.to_csv(
            f"{output_folder}/accu_report_test_{classifier.name}.csv", sep=",")
        # plot ROC curve
        skplt.metrics.plot_roc_curve(y_ground_truth, y_probas)
        plt.savefig(f"{output_folder}/roc_test_{classifier.name}.png")
    fname = os.path.basename(test_file)
    fname = fname.split(".")[0]
    org[output_predict_col_name] = y_predict
    org.to_csv(output_folder + "/" + fname + "_add_predict.csv")


def train_eval_model(classifier: Classifier,
                     output_folder,
                     x_train,
                     x_test,
                     y_train,
                     y_test,
                     feature_importance=None,
                     random_seed=666,
                     catboot_val_size=0.2):
    '''
    train the model, generate predictions, and save evaluation results.
    '''
    os.makedirs(f"{output_folder}/{classifier.name}", exist_ok=True)
    print(f"Training and evaluating model: {classifier.name}")
    # model fit
    if classifier.model_name == "CatBoostClassifier":
        # catboost model selection and fit
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=catboot_val_size,
            random_state=random_seed)
        classifier.fit(x_train, y_train, eval_set=(x_val, y_val))
    else:
        # other fit
        classifier.fit(x_train, y_train)
    # model predict
    y_predict = classifier.predict(x_test)
    y_probas = classifier.predict_proba(x_test)
    # evaluation report
    print(classification_report(y_test, y_predict))
    report = classification_report(y_test, y_predict, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        f'{output_folder}/{ classifier.name}/{ classifier.name}{model_accurancy_postfix}'
    )
    # plot roc curve
    skplt.metrics.plot_roc_curve(y_test, y_probas)
    plt.savefig(
        f'{output_folder}/{ classifier.name}/{ classifier.name}{model_roc_curve_postfix}'
    )
    plt.close()
    # joblib save model
    joblib.dump(
        classifier.model,
        f'{output_folder}/{ classifier.name}/{ classifier.name}{model_joblib_postfix}'
    )
    # run analysis feature importance
    if feature_importance:
        get_feature_importance(classifier,
                               feature_importance,
                               output_folder,
                               x_train,
                               x_test,
                               y_train,
                               y_test,
                               random_seed=random_seed)


def wrapper_train_eval_model(args):
    '''
    wrapper for train_eval_model
    '''
    return train_eval_model(*args)


def main():
    '''
    main function
    '''
    # parse command-line arguments
    parser = ArgumentParser()
    ## common args
    # input file
    parser.add_argument("-i",
                        "--input",
                        dest="input_csv",
                        required=True,
                        help="dataset (csv) file path.")
    parser.add_argument("-ix",
                        "--input_index_col",
                        required=False,
                        dest="input_index_col",
                        help="input dataset (csv) index col number.")
    # output path
    parser.add_argument("-o",
                        "--output",
                        dest="output_folder",
                        required=True,
                        help="output folder path where results will be saved.")
    # features (X) name
    parser.add_argument("-f",
                        "--features",
                        dest="features",
                        required=True,
                        type=str,
                        nargs='+',
                        help="A list of features for train and prediction")
    # target (Y) name
    parser.add_argument("-y",
                        "--y_name",
                        dest="y_name",
                        required=False,
                        help="column to predict (defult is 'None') ")
    parser.add_argument(
        "-mod",
        "--run_mode",
        dest="run_mode",
        choices=['prediction', 'training'],
        required=True,
        help=
        "run with training mode or prediction mode . defult is training mode")
    ## training mode args
    # model build config
    parser.add_argument(
        "-m",
        "--model",
        dest="model_json",
        required=False,
        help=
        "machine learning model config (json) file path or json array in training mode."
    )
    parser.add_argument(
        "-s",
        "--random_seed",
        default=666,
        required=False,
        dest="random_seed",
        help="Random seed for reproducibility in training default is 666 " +
        "(use 'auto' for randomly set 'random seeds' by system time).")
    parser.add_argument(
        "-no_encode",
        "--no_label_encoding",
        dest="no_label_encoding",
        action="store_true",
        required=False,
        help=
        "set this flag to training model without label-encoding in training mode."
    )
    parser.add_argument(
        "-p",
        "--parallel",
        dest="parallel",
        default=False,
        required=False,
        action="store_true",
        help="parallel run train and evaluation mmachine models")
    parser.add_argument("-c",
                        "--core",
                        dest="core_used",
                        required=False,
                        help="parallel procress number in training mode.")
    parser.add_argument(
        "-ts",
        "--test_size",
        dest="test_size",
        type=float,
        required=False,
        default=0.2,
        help=
        "size (percentage) of test size (validation data size) in training mode. defult=0.2"
    )
    parser.add_argument(
        "-fi",
        "--feature_importance",
        dest="feature_importance",
        choices=['SHAP', 'Permutation'],
        required=False,
        help="use SHAP or Permutation to explain feature importance")
    ## prediction mode args
    parser.add_argument(
        "-pre_m",
        "--pretrained_model_path",
        dest="pretrained_model_path",
        required=False,
        help="load pretrained model from path only for prediction mode.")
    parser.add_argument(
        "-pre_out_col",
        "--output_predict_col_name",
        dest="output_predict_col_name",
        required=False,
        help="output predict col name to save for prediction mode.")
    # load args parser
    args = parser.parse_args()
    # print all arguments of args
    print("Input arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    # parse index_col
    if args.input_index_col:
        args.input_index_col = int(args.input_index_col)
    # checkout output path
    os.makedirs(args.output_folder, exist_ok=True)
    if args.run_mode == "prediction":
        test_eval_model(args.pretrained_model_path,
                        args.input_csv,
                        args.features,
                        args.input_index_col,
                        args.output_predict_col_name,
                        args.output_folder,
                        y_name=args.y_name)
    if args.run_mode == "training":
        ml_models, conf = create_ml_models(args.model_json)
        # checkout parallel procress number
        cores = get_core_numbers(args.parallel, args.core_used)
        print(f"core to used : {cores}")
        if args.random_seed == 'auto':
            random_seed_setting = int(time.time())
        else:
            random_seed_setting = int(args.random_seed)
        np.random.seed(random_seed_setting)
        # save log info to file
        save_args_to_file(args, conf, args.output_folder)
        # load training data
        df = pd.read_csv(args.input_csv, index_col=args.input_index_col)
        df_cols = args.features + [args.y_name]
        df = df[df_cols]
        # select column without y_name for features
        X = df.drop(columns=[args.y_name])
        # encode y
        if args.no_label_encoding:
            y = df[args.y_name].values
            class_names = list(set(y))
        else:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(df[args.y_name].values)
            class_names = list(range(len(label_encoder.classes_)))
        # train test split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=random_seed_setting)
        # feature importance
        FEATURE_IM = None
        if args.feature_importance in ['SHAP', 'Permutation']:
            FEATURE_IM = args.feature_importance
        # run train model and eval
        if args.parallel:
            p = [(clf, args.output_folder, X_train, X_test, Y_train, Y_test,
                  FEATURE_IM, random_seed_setting) for clf in ml_models]
            process_map(wrapper_train_eval_model,
                        p,
                        max_workers=cores,
                        desc="Training Models Progress",
                        ncols=100,
                        dynamic_ncols=True)
        else:
            for clf in tqdm(ml_models,
                            desc="Training Models Progress",
                            ncols=100,
                            dynamic_ncols=True,
                            file=sys.stdout):
                train_eval_model(clf, args.output_folder, X_train, X_test,
                                 Y_train, Y_test, FEATURE_IM,
                                 random_seed_setting)

    return 0


if __name__ == '__main__':
    main()
    sys.exit(0)
