'''
support ml algorithm list for server

<Single-cell Reactive Oxygen Species Regulome 
Profiling Reveals Dynamic Redox Regulation in Immune Cells>

Academia Sinica IBMS SYC`LAB
whuang022@gmail.com
'''
import inspect
import importlib

support_ml_alog = {
    "KNeighborsClassifier": ("sklearn.neighbors", "KNeighborsClassifier"),
    "DecisionTreeClassifier": ("sklearn.tree", "DecisionTreeClassifier"),
    "RandomForestClassifier": ("sklearn.ensemble", "RandomForestClassifier"),
    "MLPClassifier": ("sklearn.neural_network", "MLPClassifier"),
    "AdaBoostClassifier": ("sklearn.ensemble", "AdaBoostClassifier"),
    "GaussianNB": ("sklearn.naive_bayes", "GaussianNB"),
    "QuadraticDiscriminantAnalysis":
    ("sklearn.discriminant_analysis", "QuadraticDiscriminantAnalysis"),
    "CatBoostClassifier": ("catboost", "CatBoostClassifier"),
    "XGBClassifier": ("xgboost", "XGBClassifier")
}


def get_model_params(module_name, class_name):
    '''
    auto get class params
    '''
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        signature = inspect.signature(model_class)
        params = [
            param.name for param in signature.parameters.values()
            if param.name != 'self'
        ]
        return params
    except ImportError as e:
        raise ImportError(
            f"Error importing module '{module_name}': {e}") from e
    except AttributeError as e:
        raise AttributeError(
            f"Model '{class_name}' not found in module '{module_name}'."
        ) from e


def get_ml_model_params(ml_model_name):
    '''
    auto get ml model paramss
    '''
    module_name, class_name = support_ml_alog[ml_model_name]
    params = get_model_params(module_name, class_name)
    return params
