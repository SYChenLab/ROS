#!/bin/bash
# Run accuracy report for data time predict
#
# <Single-cell Reactive Oxygen Species Regulome 
# Profiling Reveals Dynamic Redox Regulation in Immune Cells>
#
# Academia Sinica IBMS SYC`LAB
# whuang022@gmail.com
eval "$(conda shell.bash hook)"
# use conda ros_ml_env
conda activate ros_ml_env
echo $CONDA_DEFAULT_ENV
python run_accu_report.py -i ../data_time_predict/ml_models -o ../data_time_predict/ml_models/ml_performance.csv