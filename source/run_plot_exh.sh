#!/bin/bash
# Run plot_feature_mean_by_group.py to plot each exhaustion marker`s mean value of each group. 
# The group was based on the prediction exhaustion date of each ROS marker.
#
# <Single-cell Reactive Oxygen Species Regulome 
# Profiling Reveals Dynamic Redox Regulation in Immune Cells>
#
# Academia Sinica IBMS SYC`LAB
# whuang022@gmail.com

eval "$(conda shell.bash hook)"
conda activate ros_ml_env
echo $CONDA_DEFAULT_ENV
features=$(python panel_reader.py -i MC38_panel.csv -s exhaustion2)
python plot_feature_mean_by_group.py -i ../data_time_predict/ml_predict/2023_01_09_MC38_1st_norm_add_predict.csv -o ../data_time_predict/ml_predict/2023_01_09_MC38_1st_norm -g ros_predict_day -f $features
python plot_feature_mean_by_group.py -i ../data_time_predict/ml_predict/2023_01_09_MC38_2nd_norm_add_predict.csv -o ../data_time_predict/ml_predict/2023_01_09_MC38_2nd_norm -g ros_predict_day -f $features
python plot_feature_mean_by_group.py -i ../data_time_predict/ml_predict/2023_01_09_MC38_3rd_norm_add_predict.csv -o ../data_time_predict/ml_predict/2023_01_09_MC38_3rd_norm -g ros_predict_day -f $features