#!/bin/bash
# Run plot_distribution.py for three MC38 dataset
#
# <Single-cell Reactive Oxygen Species Regulome 
# Profiling Reveals Dynamic Redox Regulation in Immune Cells>
#
# Academia Sinica IBMS SYC`LAB
# whuang022@gmail.com
eval "$(conda shell.bash hook)"
conda activate ros_ml_env
echo $CONDA_DEFAULT_ENV
# plot for each data
python plot_distribution.py -i ../data_time_predict/ml_predict/2023_01_09_MC38_1st_norm_add_predict.csv -o ../data_time_predict/ml_predict/2023_01_09_MC38_1st_norm -pre_out_col ros_predict_day -g group
python plot_distribution.py -i ../data_time_predict/ml_predict/2023_01_09_MC38_2nd_norm_add_predict.csv -o ../data_time_predict/ml_predict/2023_01_09_MC38_2nd_norm -pre_out_col ros_predict_day -g group
python plot_distribution.py -i ../data_time_predict/ml_predict/2023_01_09_MC38_3rd_norm_add_predict.csv -o ../data_time_predict/ml_predict/2023_01_09_MC38_3rd_norm -pre_out_col ros_predict_day -g group
