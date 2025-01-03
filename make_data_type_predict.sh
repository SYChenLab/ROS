#!/bin/bash
#
# script for download and unzip type predict
#
# <Single-cell Reactive Oxygen Species Regulome 
# Profiling Reveals Dynamic Redox Regulation in Immune Cells>
#
# Academia Sinica IBMS SYC`LAB
# whuang022@gmail.com

mkdir data_type_predict

cd data_type_predict

curl -L -o raw_data_norm.zip "https://www.dropbox.com/scl/fi/d4fhf4r7ilc69tt4vd92b/data_type_predict_raw_data_norm.zip?rlkey=og54o827spe8d8ik9pq2t8xya&st=iveyauxg&dl=1" && echo "Download Complete"

unzip raw_data_norm.zip

rm raw_data_norm.zip