#!/bin/bash
# ROS ML for day time prediction
#
# <Single-cell Reactive Oxygen Species Regulome 
# Profiling Reveals Dynamic Redox Regulation in Immune Cells>
#
# Academia Sinica IBMS SYC`LAB
# whuang022@gmail.com
#
eval "$(conda shell.bash hook)"
# use conda ros_ml_env
conda activate ros_ml_env
echo $CONDA_DEFAULT_ENV
classifier="ml_classifier.py"
# get training 
features=$(python panel_reader.py -i MC38_panel.csv -s ROS signaling)
#run baselines models training on Invitro
python $classifier -o ../data_time_predict/ml_models/out_baselines -i ../data_time_predict/train_data/MC38_In_vitro.csv -y day -m ml_models_baselines.json -mod training -f $features -p
# run powerful models training on Invitro
python $classifier -o ../data_time_predict/ml_models/out_powerful -i ../data_time_predict/train_data/MC38_In_vitro.csv -y day -m ml_models_powerful.json -mod training -f $features -fi SHAP
python $classifier -o ../data_time_predict/ml_models/out_powerful -i ../data_time_predict/train_data/MC38_In_vitro.csv -y day -m ml_models_powerful_2.json -mod training -f $features -fi Permutation
# run prediction on Invivo
python $classifier -i ../data_time_predict/test_data/2023_01_09_MC38_1st_norm.csv -ix 0 -f $features -o ../data_time_predict/ml_predict  -mod prediction -pre_out_col ros_predict_day -pre_m ../data_time_predict/ml_models/out_powerful/CatBoostClassifier/CatBoostClassifier_model_joblib.data
python $classifier -i ../data_time_predict/test_data/2023_01_09_MC38_2nd_norm.csv -ix 0 -f $features -o ../data_time_predict/ml_predict  -mod prediction -pre_out_col ros_predict_day -pre_m ../data_time_predict/ml_models/out_powerful/CatBoostClassifier/CatBoostClassifier_model_joblib.data
python $classifier -i ../data_time_predict/test_data/2023_01_09_MC38_3rd_norm.csv -ix 0 -f $features -o ../data_time_predict/ml_predict  -mod prediction -pre_out_col ros_predict_day -pre_m ../data_time_predict/ml_models/out_powerful/CatBoostClassifier/CatBoostClassifier_model_joblib.data
