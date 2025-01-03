#!/bin/bash
#
# script for download and unzip data_time_predict
#
# <Single-cell Reactive Oxygen Species Regulome 
# Profiling Reveals Dynamic Redox Regulation in Immune Cells>
#
# Academia Sinica IBMS SYC`LAB
# whuang022@gmail.com

mkdir data_time_predict

cd data_time_predict

curl -L -o raw_data_norm.zip "https://www.dropbox.com/scl/fi/1nd992lyalrxadopg3dhw/data_time_predict_raw_data_norm.zip?rlkey=l6cydbfgyvwidgj277nqfsxfk&st=71b1h5h4&dl=1" && echo "Download Complete"

unzip raw_data_norm.zip

rm raw_data_norm.zip

curl -L -o test_data.zip "https://www.dropbox.com/scl/fi/vu56ntoi4dku1gzk5r4kw/data_time_predict_test_data.zip?rlkey=iqiv10s64ejw9q6rim6edvz4t&e=1&st=3mtt67rd&dl=1" && echo "Download Complete"

unzip test_data.zip

rm test_data.zip

