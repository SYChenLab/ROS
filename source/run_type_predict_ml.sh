#!/bin/bash
# run all dataset
#
# <Single-cell Reactive Oxygen Species Regulome 
# Profiling Reveals Dynamic Redox Regulation in Immune Cells>
#
# Academia Sinica IBMS SYC`LAB
# whuang022@gmail.com
eval "$(conda shell.bash hook)"
conda activate ros_ml_env
echo $CONDA_DEFAULT_ENV
classifier="ml_classifier.py"
output_dir="../data_type_predict/ml_models/catboost"
for dir in ../data_type_predict/train_data/*; do
    if [ -d "$dir" ]; then
        dir_name=$(basename "$dir")
        train_file="$dir/train_${dir_name}.csv"
        train_output="$output_dir/train_${dir_name}_output"
        test_file="$dir/test_${dir_name}.csv"
        test_output="$output_dir/test_${dir_name}_output"
        # train on 8 donors 
        features=$(python panel_reader.py -i MC38_panel.csv -s ROS2 signaling2)
	    python $classifier -i $train_file -o $train_output -y type -mod training -s 666 -m ml_models_catboost.json -f $features -ix 0 -no_encode
	    mv ./catboost_info $output_dir/${dir_name}_catboost_info
        #cp -r /media/whuang022/DATA/ros_curve $train_output/ros_curve
        # test  on other 2 donors 
        python $classifier -i $test_file -f $features -o $test_output -y type -mod prediction -pre_m $train_output/CatBoostClassifier/CatBoostClassifier_model_joblib.data -ix 0 -pre_out_col ros_predict_celltype
        #cp -r /media/whuang022/DATA/ros_curve $test_output/ros_curve
    fi
done
