'''
Flask API server for ml_classifier.py 

<Single-cell Reactive Oxygen Species Regulome 
Profiling Reveals Dynamic Redox Regulation in Immune Cells>

Academia Sinica IBMS SYC`LAB
whuang022@gmail.com
'''
import os
import datetime
import shutil
import tempfile
import time
import uuid
import base64
import csv
import io
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from celery.result import AsyncResult
from celery import Celery
from sqlitedict import SqliteDict
from model_parms import get_ml_model_params
from tasks import execute_run_ml

app = Flask(__name__)
CORS(app)

OUTPUT_FOLDER = 'celery_task_output'
UPLOAD_FOLDER = 'uploads'

model_roc_curve_postfix = "_roc.png"
model_accurancy_postfix = "_accu.csv"
model_joblib_postfix = "_model_joblib.data"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # allow max szie = 1GB

capp = Celery('tasks',
              broker='redis://localhost:6379/0',
              backend='redis://localhost:6379/0',
              task_track_started=True)

result_paths = SqliteDict('result_paths.db', outer_stack=False)
task_meta = SqliteDict('task_meta.db', outer_stack=False)
task_list = SqliteDict('task_list.db', outer_stack=False)

status_updates = []

ui_color = {
    'PENDING': 'secondary',
    'SUCCESS': 'success',
    'STARTED': 'warning',
    'FAILURE': 'danger'
}


def create_directory_with_uuid():
    '''
    create output path & uuid
    '''
    result_id_key = str(uuid.uuid4())
    result_id_value = str(uuid.uuid4())
    output_root = os.path.join(os.getcwd(), OUTPUT_FOLDER)
    task_outpath = os.path.join(output_root, result_id_value)
    os.makedirs(task_outpath)
    result_paths[result_id_key] = result_id_value
    result_paths.commit()
    return result_id_key, result_id_value, task_outpath


def list_subfolders(directory):
    '''
    list subfolder of a folder
    '''
    try:
        items = os.listdir(directory)
        subfolders = [
            item for item in items
            if os.path.isdir(os.path.join(directory, item))
        ]
        return subfolders
    except FileNotFoundError:
        print(f"folder {directory} not exist")
        return []
    except PermissionError:
        print(f"no  permission to access {directory}")
        return []


def read_csv_to_str(csv_file_path):
    '''
    csv to single str
    '''
    output = io.StringIO()
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_writer = csv.writer(output)
        for row in csv_reader:
            csv_writer.writerow(row)
    return output.getvalue()


def get_file_details(filepath):
    '''
    retrun filetime and md5hash
    '''
    modified_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                  time.localtime(os.path.getmtime(filepath)))
    return modified_time


@app.route('/upload', methods=['POST'])
def upload():
    '''
    upload csv file to upload path
    '''
    if 'file' not in request.files:
        return jsonify({'message': 'file not upload.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'no file has selected.'}), 400

    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return jsonify({'message':
                        f'file {file.filename} upload successed.'}), 200
    return jsonify({'message': 'can only upload csv file.'}), 400


@app.route('/get_ml_model_file/<result_id>/<model_name>', methods=['GET'])
def get_ml_model_file(result_id, model_name):
    '''
    get ml model from result_id (result_id_key)
    '''
    output_root = os.path.join(os.getcwd(), OUTPUT_FOLDER)
    task_outpath = os.path.join(output_root, result_paths[result_id])
    models = list_subfolders(task_outpath)
    if model_name in models:
        model_path = os.path.join(task_outpath, model_name)
        model_joblib_fname = model_name + model_joblib_postfix
        model_joblib_fpath = os.path.join(model_path, model_joblib_fname)
        with tempfile.TemporaryDirectory() as tmpdir_name:
            model_joblib_fpath_tmp = os.path.join(tmpdir_name,
                                                  model_joblib_fname)
            shutil.copyfile(model_joblib_fpath, model_joblib_fpath_tmp)
            return send_from_directory(tmpdir_name,
                                       model_joblib_fname,
                                       as_attachment=True)
    else:
        return jsonify({'message': f'model {model_name} not found .'}), 400


@app.route('/get_ml_results/<result_id>', methods=['GET'])
def get_ml_results(result_id):
    '''
    get ml results from result_id (result_id_key)
    '''
    output_root = os.path.join(os.getcwd(), OUTPUT_FOLDER)
    task_outpath = os.path.join(output_root, result_paths[result_id])
    models = list_subfolders(task_outpath)
    ml_results = []
    meta = task_meta[result_id]
    for model_name in models:
        model_path = os.path.join(task_outpath, model_name)
        model_roc_curve_fpath = os.path.join(
            model_path, model_name + model_roc_curve_postfix)
        model_accurancy_fpath = os.path.join(
            model_path, model_name + model_accurancy_postfix)
        model_roc_curve_img_base64 = None
        with open(model_roc_curve_fpath, 'rb') as img_file:
            model_roc_curve_img_base64 = base64.b64encode(
                img_file.read()).decode('utf-8')
        model_accurancy = read_csv_to_str(model_accurancy_fpath)
        ml_result = {
            "project_name": meta["task_args"]["project_name"],
            "task_id": meta["task_record"]["task_id"],
            "task_time": meta["task_record"]["task_time"],
            "result_id": result_id,
            "model_name": model_name,
            "model_url":
            f'{request.url_root}get_ml_model_file/{result_id}/{model_name}',
            "model_roc_curve_fname": model_name + model_roc_curve_postfix,
            "model_roc_curve": model_roc_curve_img_base64,
            "model_accurancy_fname": model_name + model_accurancy_postfix,
            "model_accurancy": model_accurancy,
            "input_csv": meta["task_args"]["input_csv"],
            "input_index_col": meta["task_args"]["input_index_col"],
            "random_seed": meta["task_args"]["random_seed"],
            "features": meta["task_args"]["features"],
            "target": meta["task_args"]["y_name"],
            "algorithm_params": meta["task_args"]["algorithm_params"]
        }
        ml_results.append(ml_result)

    return jsonify(ml_results)


@app.route('/delete_file', methods=['POST'])
def delete_file():
    '''
    del file
    '''
    filename = request.form['filename']
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    return '', 200


@app.route('/rename_file', methods=['POST'])
def rename_file():
    '''
    rename file name
    '''
    old_filename = request.form['old_filename']
    new_filename = request.form['new_filename']
    old_filepath = os.path.join(UPLOAD_FOLDER, old_filename)
    new_filepath = os.path.join(UPLOAD_FOLDER, new_filename)

    if os.path.exists(old_filepath):
        os.rename(old_filepath, new_filepath)
    return '', 200


@app.route('/get_csv_files', methods=['GET'])
def get_csv_files():
    '''
    list csv files
    '''
    files = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith('.csv'):  # Filter for CSV files
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            modified_time = get_file_details(filepath)
            files.append({
                'name': filename,
                'modified_time': modified_time,
            })
    return jsonify({'files': files})


@app.route('/get_columns', methods=['POST'])
def get_columns():
    '''
    show columns of file
    '''
    file = request.values.get('file')
    input_index_col = request.values.get('input_index_col')
    if input_index_col != "":
        input_index_col = int(input_index_col)
    else:
        input_index_col = None
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        columns = pd.read_csv(file_path,
                              index_col=input_index_col).columns.tolist()
        return jsonify({'columns': columns})

    except (OSError, ValueError) as e:
        return jsonify({'error': str(e)}), 400


@app.route('/submit_project', methods=['POST'])
def submit_project():
    ''''
    submit & execute run ml project
    '''
    data = request.get_json()
    project_name = data.get('project_name')
    input_csv = data.get('input_csv')
    features = data.get('features')
    y_name = str(data.get('target'))
    algorithm = data.get('algorithm')
    input_index_col = data.get('input_index_col')
    algorithm_params = data.get('algorithm_params')
    result_id_key, _, result_outpath = create_directory_with_uuid()
    args = {
        "project_name": project_name,
        "input_csv": input_csv,
        "input_index_col": input_index_col,
        "use_encode": data.get('use_encode'),
        "random_seed": data.get('random_seed'),
        "features": features,
        "y_name": y_name,
        "algorithm": algorithm,
        "algorithm_params": algorithm_params,
        "result_id": result_id_key,
        "result_outpath": result_outpath
    }
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    task = execute_run_ml.apply_async(args=[args])
    task_record = {
        "task_name": project_name,
        "task_id": task.id,
        "task_time": current_time
    }
    n = len(task_list)
    task_list[str(n + 1)] = task_record
    task_list.commit()
    task_meta[result_id_key] = {"task_args": args, "task_record": task_record}
    task_meta.commit()
    return jsonify({
        'message':
        f'Project setup submitted successfully! \nTask ID = {task.id}'
    })


@app.route('/status/<task_id>')
def task_status(task_id):
    '''
    get task status
    '''
    task = AsyncResult(task_id, app=capp)
    if task.state == 'PENDING':
        return jsonify({'status': 'Task is pending'})
    if task.state == 'SUCCESS':
        return jsonify({'status': 'Task completed', 'result': task.result})
    if task.state == 'STARTED':
        return jsonify({'status': 'Task has started running.'})
    if task.state == 'FAILURE':
        return jsonify({'status': 'Task failed', 'error': task.result})
    return jsonify({'status': task.state})


@app.route('/tasks')
def get_tasks_status():
    '''
    list submited tasks
    '''
    task_statuses = []
    for _, task_rec in task_list.items():
        task_name = task_rec["task_name"]
        task_id = task_rec["task_id"]
        task_time = task_rec["task_time"]
        task = AsyncResult(task_id, app=capp)
        if task.state == 'SUCCESS':
            if task.result['result_state']:
                color = 'success'
                state = 'SUCCESS'
            else:
                color = 'danger'
                state = 'FAILURE'
            task_statuses.append({
                'task_name': task_name,
                'task_time': task_time,
                'task_id': task_id,
                'status': state,
                'color': color,
                'result': task.result
            })
        else:

            task_statuses.append({
                'task_name': task_name,
                'task_time': task_time,
                'task_id': task_id,
                'status': task.state,
                'color': ui_color[task.state],
                'result': task.result
            })

    return jsonify(task_statuses)


@app.route('/get_algorithm_params/<model_name>', methods=['GET'])
def get_algorithm_params(model_name):
    '''
    list available params of ml model
    '''
    params = get_ml_model_params(model_name)
    return jsonify({'parameters': params})


if __name__ == '__main__':
    app.run(debug=True)
