#!/bin/bash
# ROS ML for day time prediction
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
printf '\nTrain Accuracy Stats:'
python model_stat_train.py
printf '\nTest Accuracy Stats:'
python model_stat_test.py