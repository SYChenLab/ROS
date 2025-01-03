'''
celery task for server

<Single-cell Reactive Oxygen Species Regulome 
Profiling Reveals Dynamic Redox Regulation in Immune Cells>

Academia Sinica IBMS SYC`LAB
whuang022@gmail.com

'''
import os
import json
import subprocess
from celery import Celery
from model_parms import support_ml_alog

celery = Celery('tasks',
                broker='redis://localhost:6379/0',
                backend='redis://localhost:6379/0',
                task_track_started=True)

OUTPUT_FOLDER = 'celery_task_output'
UPLOAD_FOLDER = 'uploads'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def convert_value(value):
    '''
    convert str to data-type
    '''
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def convert_dict(input_dict):
    ''''
    convert dict val (str) to data-type
    '''
    return {key: convert_value(value) for key, value in input_dict.items()}


@celery.task
def execute_run_ml(args):
    '''
    execute ml_classifier.py with celery.task
    '''
    input_csv = os.path.join(UPLOAD_FOLDER, args["input_csv"])
    features = args["features"]
    algorithm_params = convert_dict(args["algorithm_params"])
    module, model = support_ml_alog[args["algorithm"]]
    result_id = args["result_id"]
    m = [{
        "name": args["algorithm"],
        "model": model,
        "module": module,
        "params": algorithm_params
    }]
    m = json.dumps(m)
    command = [
        "python", "ml_classifier.py", "-o", args["result_outpath"], "-i", input_csv,
        "-y", args["y_name"],"-mod","training", "-m", m, "-s", args['random_seed'], "-f"
    ]
    command.extend(features)
    if args["input_index_col"] != "":
        command.extend(["-ix", str(args["input_index_col"])])
    if args["use_encode"] == "False":
        command.extend(["-no_encode"])
    try:
        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)

        stdout, stderr = process.communicate()
        process.stdout.close()
        process.stderr.close()
        process.wait()

        result_state = False
        if process.returncode == 0:
            result_state = True
        results = {
            "result_id": result_id,
            "result_stdout": stdout,
            "result_stderr": stderr,
            "result_state": result_state
        }

        return results

    except OSError as e:
        raise f"OS error: {str(e)}"
    finally:
        if process.poll() is None:
            process.terminate()
            process.stdout.close()
            process.stderr.close()
