[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Reactive Oxygen Species Machine Learning Project

This is the machine learning pipeline and web service of the research: 

> Single-cell Reactive Oxygen Species Regulome Profiling Reveals Dynamic Redox Regulation in Immune Cells
Yi-Chuan Wang; Ping-Hsun Wu; Wen-Chieh Ting; Yi-Fu Wang; Ming-Han Yang; Tung-Hung Su; Jia-Ying Su; Hsun-I Sun; Wei-Min Huang; Pei-Ling Tsai; Gerlinde Wernig; Shih-Lei Lai; Chia-Wei Li; Tai-Ming Ko; Kai-Chien Yang; Ya-Jen Chang; Yijuang Chern; Yao-Ming Chang; Mei-Chuan Kuo; Yen-Tsung Huang; Yi-Shiuan Tzeng; Shih-Yu Chen; Jih-Luh Tang.

to use the pipeline, please follow the steps below.

## Installation

### 1. Create conda environment

```bash
conda create -n ros_ml_env python=3.9.5
conda activate ros_ml_env
```

### 2. Install dependencies

#### 2.1 Install dependencies for ml_classifier.py and utils programs
```bash
pip install numpy==1.26.3
pip install pandas==1.5.3
pip install scipy==1.11.4
pip install scikit-learn==1.2.1
pip install xgboost==1.6.2
pip install Catboost==1.1
pip install traitlets==5.9.0
pip install ipywidgets==8.0.2
pip install scikit-plot
pip install mljar-scikit-plot
pip install ipython
pip install shap
pip install tqdm
pip install seaborn
```
#### 2.2 Install for web-service of ml classifier

Our tool 'ml_classifier.py' was a command-line tool. If you want to use its web service, please install sections 2.1 and 2.2. 

```bash
pip install flask==3.1.0
pip install flask-cors==5.0.0
pip install celery==5.4.0
pip install redis==5.2.1
pip install amqp==5.1.0
pip install flask-socketio==5.5.0
pip install sqlitedict
```

## Usage

## 1. Prepare Data

Follow the commands to download data or vist [here](https://www.dropbox.com/scl/fo/r2clk9s0n8ne6obieisff/AB5nVbldJUrnR9ytYifXkwA?rlkey=1v2y88t0j51p2i8oi4cpzjyyc&st=g5wj5gyl&dl=0) or [here](https://drive.google.com/drive/folders/1Qlo76aIos0IYL7z1Ir03wSY1OCCmE5ki?usp=sharing) .

### 1.1. data_type_predict_preprocess.py

To prepare data of the type predict ,use the following commandands :

```bash
mkdir data_type_predict

cd data_type_predict

curl -L -o raw_data_norm.zip "https://www.dropbox.com/scl/fi/d4fhf4r7ilc69tt4vd92b/data_type_predict_raw_data_norm.zip?rlkey=og54o827spe8d8ik9pq2t8xya&st=iveyauxg&dl=1" && echo "Download Complete"

unzip raw_data_norm.zip
```
or simply 
```bash
chmod +x make_data_type_predict.sh
```
```bash
./make_data_type_predict.sh
```
The path of './data_type_predict' should be like:

```bash
./data_type_predict
├── raw_data_norm
│   ├── donor_0_norm.csv
│   ├── donor_1_norm.csv
│   ├── donor_2_norm.csv
│   ├── donor_3_norm.csv
│   ├── donor_4_norm.csv
│   ├── donor_5_norm.csv
│   ├── donor_6_norm.csv
│   ├── donor_7_norm.csv
│   ├── donor_8_norm.csv
│   └── donor_9_norm.csv
```

than create all combinations of 8 donors for training and 2 donors for testing from a 10 donors dataset by run :

```bash
cd source

python data_type_predict_preprocess.py
```
then will get:

```bash
./data_type_predict
│
├── raw_data_norm
│   ├── donor_0_norm.csv
│   ├── donor_1_norm.csv
│   ├── donor_2_norm.csv
│   ├── donor_3_norm.csv
│   ├── donor_4_norm.csv
│   ├── donor_5_norm.csv
│   ├── donor_6_norm.csv
│   ├── donor_7_norm.csv
│   ├── donor_8_norm.csv
│   └── donor_9_norm.csv
└── train_data
    ├── 0_1
    │   ├── test_0_1.csv
    │   └── train_0_1.csv
    ├── 0_2
    ...

```
### 1.2. data_time_predict_preprocess.py

To prepare data of the time predict ,use the following commandands :

```bash
mkdir data_time_predict

cd data_time_predict

curl -L -o raw_data_norm.zip "https://www.dropbox.com/scl/fi/1nd992lyalrxadopg3dhw/data_time_predict_raw_data_norm.zip?rlkey=l6cydbfgyvwidgj277nqfsxfk&st=71b1h5h4&dl=1" && echo "Download Complete"

unzip raw_data_norm.zip

curl -L -o test_data.zip "https://www.dropbox.com/scl/fi/vu56ntoi4dku1gzk5r4kw/data_time_predict_test_data.zip?rlkey=iqiv10s64ejw9q6rim6edvz4t&e=1&st=3mtt67rd&dl=1" && echo "Download Complete"

unzip test_data.zip
```

or simply 
```bash
chmod +x make_data_time_predict.sh
```
```bash
./make_data_time_predict.sh
```

The path of './data_time_predict' should be like:

```bash
./data_time_predict
├── raw_data_norm
│   ├── In_vitro_1st.csv
│   ├── In_vitro_2nd.csv
│   └── In_vitro_3rd.csv
├── test_data
│   ├── 2023_01_09_MC38_1st_norm.csv
│   ├── 2023_01_09_MC38_2nd_norm.csv
│   └── 2023_01_09_MC38_3rd_norm.csv
```

We need to combine 3 Invitro OT-1 CD8 T cells ROS markers dataset as training data by run :

```bash
cd source

python data_time_predict_preprocess.py
```
then will get:

```bash
./data_time_predict
├── raw_data_norm
│   ├── In_vitro_1st.csv
│   ├── In_vitro_2nd.csv
│   └── In_vitro_3rd.csv
├── test_data
│   ├── 2023_01_09_MC38_1st_norm.csv
│   ├── 2023_01_09_MC38_2nd_norm.csv
│   └── 2023_01_09_MC38_3rd_norm.csv
└── train_data
    └── MC38_In_vitro.csv

```


## 1.3 panel_reader.py

read panel table (csv) and output selected panel genes.

```bash
usage: panel_reader.py [-h] -i INPUT_CSV -s SELECT_PANEL [SELECT_PANEL ...]

options:
  -h, --help            show this help message and exit
  -i INPUT_CSV, --input INPUT_CSV
                        dataset (csv) file path.
  -s SELECT_PANEL [SELECT_PANEL ...], --select_panel SELECT_PANEL [SELECT_PANEL ...]

```

for example 
```bash
python panel_reader.py -i MC38_panel.csv -s ROS signaling
```

will output :

```bash
NNT KEAP1 HSP70 PCYXL PRDX4 SOD2 53bp1 GPX4 NRF2 PDI HO1 MTH1 OLR1 CD36 catalase ACOX3 AQP8 Ref_APE ERO1B QSOX1 oxPTP oxDJ1 p53 CD163 mTOR pS6 HIF1a p38MAPK pNFkB pERK c_Jun
```

## 2.Train and Test Machine Learning Model

### 2.1 ml_classifier.py

This program is a wrapper of Python's common machine learning classifier, which trains models and predicts. We used it to work on our ROS prediction task.

```bash

usage: ml_classifier.py [-h] -i INPUT_CSV [-ix INPUT_INDEX_COL] -o
                        OUTPUT_FOLDER -f FEATURES [FEATURES ...] [-y Y_NAME]
                        -mod {prediction,training} [-m MODEL_JSON]
                        [-s RANDOM_SEED] [-no_encode] [-p] [-c CORE_USED]
                        [-ts TEST_SIZE] [-fi {SHAP,Permutation}]
                        [-pre_m PRETRAINED_MODEL_PATH]
                        [-pre_out_col OUTPUT_PREDICT_COL_NAME]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_CSV, --input INPUT_CSV
                        dataset (csv) file path.
  -ix INPUT_INDEX_COL, --input_index_col INPUT_INDEX_COL
                        input dataset (csv) index col number.
  -o OUTPUT_FOLDER, --output OUTPUT_FOLDER
                        output folder path where results will be saved.
  -f FEATURES [FEATURES ...], --features FEATURES [FEATURES ...]
                        A list of features for train and prediction
  -y Y_NAME, --y_name Y_NAME
                        column to predict (defult is 'None')
  -mod {prediction,training}, --run_mode {prediction,training}
                        run with training mode or prediction mode . defult is
                        training mode
  -m MODEL_JSON, --model MODEL_JSON
                        machine learning model config (json) file path or json
                        array in training mode.
  -s RANDOM_SEED, --random_seed RANDOM_SEED
                        Random seed for reproducibility in training default is
                        666 (use 'auto' for randomly set 'random seeds' by
                        system time).
  -no_encode, --no_label_encoding
                        set this flag to training model without label-encoding
                        in training mode.
  -p, --parallel        parallel run train and evaluation mmachine models
  -c CORE_USED, --core CORE_USED
                        parallel procress number in training mode.
  -ts TEST_SIZE, --test_size TEST_SIZE
                        size (percentage) of test size (validation data size)
                        in training mode. defult=0.2
  -fi {SHAP,Permutation}, --feature_importance {SHAP,Permutation}
                        use SHAP or Permutation to explain feature importance
  -pre_m PRETRAINED_MODEL_PATH, --pretrained_model_path PRETRAINED_MODEL_PATH
                        load pretrained model from path only for prediction
                        mode.
  -pre_out_col OUTPUT_PREDICT_COL_NAME, --output_predict_col_name OUTPUT_PREDICT_COL_NAME
                        output predict col name to save for prediction mode.

```
And the -m should input a JSON format file include with the following key-values paire:

| KEY    | VALUES                 |
| ------ | ---------------------- |
| name   | classifier name        |
| model  | classifier class Name  |
| module | classifier module Name |
| params | classifier params      |


```bash
[
  {
    "name": <classifier name>,
    "model": <classifier class Name>,
    "module": <classifier module Name>,
    "params": {
      <classifier parms key > : <classifier parms value>,
      ...
    }
  }
  ,...
]
```
example 'ml_models_catboost.json':

```bash
[
  {
    "name": "CatBoostClassifier",
    "model": "CatBoostClassifier",
    "module": "catboost",
    "params": {
      "loss_function": "MultiClass",
      "custom_metric": ["Accuracy"],
      "random_seed": 666,
      "logging_level": "Verbose",
      "eval_metric": "TotalF1",
      "use_best_model": true,
      "iterations": 1500,
      "depth": 8
    }
  }
]
```
for example in command line type the following commandss to train a model of iris data set :

```bash
python ml_classifier.py -i ../test/iris.csv -o ../test/output -f sepal_length sepal_width petal_length petal_width -y species -mod training -no_encode -m '[ { "name": "Catboost", "model": "CatBoostClassifier", "module": "catboost", "params": { "loss_function": "MultiClass", "custom_metric": ["Accuracy"], "random_seed": 666, "logging_level": "Verbose", "eval_metric": "TotalF1", "use_best_model": true, "iterations": 100, "depth": 10 } } ]'
```

or for our project

```bash
python ml_classifier.py -mod training -o ../test/output -i ../data_time_predict/train_data/MC38_In_vitro.csv -y day -m '[ { "name": "Catboost", "model": "CatBoostClassifier", "module": "catboost", "params": { "loss_function": "MultiClass", "custom_metric": ["Accuracy"], "random_seed": 666, "logging_level": "Verbose", "eval_metric": "TotalF1", "use_best_model": true, "iterations": 1500, "depth": 8 } } ]' -f NNT KEAP1 HSP70 PCYXL PRDX4 SOD2 53bp1 GPX4 NRF2 PDI HO1 MTH1 OLR1 CD36 catalase ACOX3 AQP8 Ref_APE ERO1B QSOX1 oxPTP oxDJ1 p53 CD163 mTOR pS6 HIF1a p38MAPK pNFkB pERK c_Jun
```

or use JSON config file as model setting:

```bash
python ml_classifier.py -mod training -o ../test/output -i ../data_time_predict/train_data/MC38_In_vitro.csv -y day -m ml_models_catboost.json -f NNT KEAP1 HSP70 PCYXL PRDX4 SOD2 53bp1 GPX4 NRF2 PDI HO1 MTH1 OLR1 CD36 catalase ACOX3 AQP8 Ref_APE ERO1B QSOX1 oxPTP oxDJ1 p53 CD163 mTOR pS6 HIF1a p38MAPK pNFkB pERK c_Jun
```

### 2.2 run_day_predict_ml.sh (time prediction)

This script will automatic run ros day prediction procress and  output to ../data_time_predict/ml_models

set the execute bit :

```bash
chmod +x run_day_predict_ml.sh
```
and run :

```bash
./run_day_predict_ml.sh 
```

### 2.3 run_type_predict_ml.sh (cell-type prediction)

This script will automatic run ros cell-type prediction procress and  output to ../data_type_predict/ml_models

set the execute bit :


```bash
chmod +x run_type_predict_ml.sh
```
and run :

```bash
./run_type_predict_ml.sh 
```

## 3.Analysis accurancy of the machine learning models

### 3.1 run_accu_report.py

```bash

python run_accu_report.py -i ../data_time_predict/ml_models -o ../data_time_predict/ml_models/ml_performance.csv
```
or run with script:
```bash
chmod +x run_data_time_predict_run_accu_report.sh
```

```bash
./run_data_time_predict_run_accu_report.sh
```

In our project of time prediction , we got the best model by ued CatBoostClassifier :

(ml_performance.csv) 

|                     |0                 |1                 |2                 |3                 |4                 |5                 |accuracy          |macro avg         |weighted avg      |
|---------------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|
|CatBoostClassifier_f1-score|0.9805            |0.8055            |0.6882            |0.6744            |0.6961            |0.6008            |0.7327            |0.7409            |0.7320            |
|Xgboost_f1-score           |0.9773            |0.7854            |0.6806            |0.6660            |0.6922            |0.5935            |0.7256            |0.7325            |0.7248            |
|Random Forest_f1-score     |0.9666            |0.7023            |0.6246            |0.6227            |0.6752            |0.5347            |0.6862            |0.6877            |0.6842            |
|Neural Net_f1-score        |0.9602            |0.7230            |0.6299            |0.5865            |0.6454            |0.5322            |0.6728            |0.6795            |0.6704            |
|QDA_f1-score               |0.9409            |0.6749            |0.6139            |0.6218            |0.6277            |0.5171            |0.6668            |0.6661            |0.6659            |
|AdaBoost_f1-score          |0.9585            |0.6429            |0.6003            |0.5496            |0.6169            |0.4976            |0.6418            |0.6443            |0.6398            |
|Naive Bayes_f1-score       |0.9332            |0.5693            |0.5522            |0.5565            |0.5539            |0.4901            |0.6121            |0.6092            |0.6101            |
|Decision Tree_f1-score     |0.9473            |0.5583            |0.5783            |0.5370            |0.5869            |0.4091            |0.6136            |0.6028            |0.6108            |
|KNN_f1-score               |0.9562            |0.6450            |0.5449            |0.5213            |0.5409            |0.3584            |0.5992            |0.5944            |0.5929            |

### 3.2 compute the mean and std. of models

To compute the accurancy statistics of the cell type prediction can run as :

```bash
python model_stat_train.py
```

```bash
python model_stat_test.py
```

or run by script:

```bash
  ./run_model_stat.sh
```
and will get 


```bash
Train Accuracy Stats:
mean:
              precision  recall  f1-score      support
Bcell            0.6503  0.3391    0.4454    4095.5556
NKcell           0.7005  0.6465    0.6723    9104.8889
Tcell            0.8194  0.9011    0.8583   29158.4444
accuracy         0.9355  0.9355    0.9355       0.9355
basophil         0.7596  0.5848    0.6607    1516.2889
macro avg        0.8156  0.7392    0.7668  164895.5778
monocyte         0.9729  0.9717    0.9723   12660.6222
neutrophil       0.9912  0.9920    0.9916  108359.7778
weighted avg     0.9328  0.9355    0.9325  164895.5778

std:
              precision  recall  f1-score    support
Bcell            0.0103  0.0230    0.0214   407.7512
NKcell           0.0082  0.0217    0.0144  1145.2200
Tcell            0.0076  0.0092    0.0079  2027.9351
accuracy         0.0036  0.0036    0.0036     0.0036
basophil         0.0110  0.0239    0.0177   135.8217
macro avg        0.0026  0.0068    0.0052  8263.2131
monocyte         0.0017  0.0031    0.0021  1275.2812
neutrophil       0.0008  0.0005    0.0006  5168.4164
weighted avg     0.0036  0.0036    0.0036  8263.2131

Test Accuracy Stats:
mean:
              precision  recall  f1-score      support
Bcell            0.6112  0.3240    0.4185    5088.4000
NKcell           0.6592  0.6297    0.6338   11388.0000
Tcell            0.8149  0.8977    0.8528   36405.2000
accuracy         0.9326  0.9326    0.9326       0.9326
basophil         0.7162  0.5408    0.6075    1903.0000
macro avg        0.7932  0.7252    0.7452  206119.0000
monocyte         0.9675  0.9677    0.9676   15819.4000
neutrophil       0.9904  0.9912    0.9908  135515.0000
weighted avg     0.9319  0.9326    0.9297  206119.0000

std:
              precision  recall  f1-score     support
Bcell            0.0893  0.0549    0.0530   2008.1602
NKcell           0.1169  0.0610    0.0600   5690.6970
Tcell            0.0560  0.0223    0.0277  10142.7066
accuracy         0.0143  0.0143    0.0143      0.0143
basophil         0.1004  0.0913    0.0638    651.5431
macro avg        0.0315  0.0159    0.0178  41316.0854
monocyte         0.0141  0.0090    0.0101   6355.2965
neutrophil       0.0018  0.0035    0.0019  25640.8262
weighted avg     0.0150  0.0143    0.0157  41316.0854
```
The low standard deviation showed the machine learning model's generalization ability between different donors (persons) when we used the ROS markers to train and predict cell type. And can Ignore differences between individuals.

Fig. 1i and Fig. 2i was from train_8_9_output and test_8_9_output .

## 4. Visualize distribution and Validation
## 4.1 plot_distribution.py

To validate our prediction we use the following program to plot the distribution with ground truth labeling and the predicted labeling.

```bash
usage: plot_distribution.py [-h] -i INPUT_CSV -o OUTPUT_FOLDER -pre_out_col
                            OUTPUT_PREDICT_COL_NAME -g GROUP

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_CSV, --input INPUT_CSV
                        dataset (csv) file with prediction.
  -o OUTPUT_FOLDER, --output OUTPUT_FOLDER
                        output folder path.
  -pre_out_col OUTPUT_PREDICT_COL_NAME, --output_predict_col_name OUTPUT_PREDICT_COL_NAME
                        output_predict_col_name
  -g GROUP, --group GROUP
                        feature for group to plot , must be str
```
We plot the bar plot and the KDE plot of date time prediction as the following commands:

```bash
python plot_distribution.py -i ../data_time_predict/ml_predict/2023_01_09_MC38_1st_norm_add_predict.csv -o ../data_time_predict/ml_predict/2023_01_09_MC38_1st_norm -pre_out_col ros_predict_day -g group

python plot_distribution.py -i ../data_time_predict/ml_predict/2023_01_09_MC38_2nd_norm_add_predict.csv -o ../data_time_predict/ml_predict/2023_01_09_MC38_2nd_norm -pre_out_col ros_predict_day -g group

python plot_distribution.py -i ../data_time_predict/ml_predict/2023_01_09_MC38_3rd_norm_add_predict.csv -o ../data_time_predict/ml_predict/2023_01_09_MC38_3rd_norm -pre_out_col ros_predict_day -g group
```
or run by :

```bash
chmod +x run_data_time_predict_plot.sh
```

```bash
./run_data_time_predict_plot.sh
```
Fig. 2i was the result of combining those three results.

## 4.2 plot_feature_mean_by_group.py

We also validate our prediction by plotting the mean values of the exhaustion markers from each predicted date that were from the ROS marker.

To make the plot from features , we used this program: 

```bash
usage: plot_feature_mean_by_group.py [-h] -i INPUT_CSV -o OUTPUT_FOLDER -g
                                     GROUP -f FEATURES [FEATURES ...]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_CSV, --input INPUT_CSV
                        dataset (csv) file with prediction.
  -o OUTPUT_FOLDER, --output OUTPUT_FOLDER
                        output folder path.
  -g GROUP, --group GROUP
                        feature for group to plot , must be str
  -f FEATURES [FEATURES ...], --features FEATURES [FEATURES ...]
                        A list of features for observe
```
we can plot the figure with:

```bash
python plot_feature_mean_by_group.py -i ../data_time_predict/ml_predict/2023_01_09_MC38_1st_norm_add_predict.csv -o ../data_time_predict/ml_predict/2023_01_09_MC38_1st_norm -g ros_predict_day -f TCF1_7 TOX PD1 LAG3 TIM3

python plot_feature_mean_by_group.py -i ../data_time_predict/ml_predict/2023_01_09_MC38_2nd_norm_add_predict.csv -o ../data_time_predict/ml_predict/2023_01_09_MC38_2nd_norm -g ros_predict_day -f TCF1_7 TOX PD1 LAG3 TIM3

python plot_feature_mean_by_group.py -i ../data_time_predict/ml_predict/2023_01_09_MC38_3rd_norm_add_predict.csv -o ../data_time_predict/ml_predict/2023_01_09_MC38_3rd_norm -g ros_predict_day -f TCF1_7 TOX PD1 LAG3 TIM3

```

or run with the script:


```bash
chmod +x run_plot_exh.sh
```

```bash
./run_plot_exh.sh
```

Fig. 2j was the result of combining those three results.

## 5. Result

The whole training dataset and the machine learning models result can be download from here:

### 5.1 cell type prediction [data_type_predict](https://drive.google.com/file/d/1gahIpnIpDcpZETXB0d-hwvPWWdDTE0rL/view?usp=sharing) (7GB)

### 5.2 date prediction [data_time_predict](https://drive.google.com/file/d/15Rom75-S36tyBRuRe-yct3aWJhBa55SF/view?usp=sharing) (220 MB)

## 6. Run Web Application

We also created a web application as GUI for training machine learning models for classification. To use it

step1. cd to path : 

```bash
cd ./web_app
```

step2. run the Celery queue:

```bash
celery -A tasks.celery worker --loglevel=info --concurrency=4
```

step3. run run Flask api server :

```bash
python server.py
```
than will start the server at [http://127.0.0.1:5000](http://127.0.0.1:5000)

step4. and vist  [index.html](./source/web_app/index.html) to use it.


The detail of web_app usage can be downlad at [here](./web_app/20241225_ML_Platform.pdf)



