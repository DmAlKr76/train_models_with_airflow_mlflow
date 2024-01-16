import io
import json
import logging
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import NoReturn, Literal, Dict, Any

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago


DEFAULT_ARGS = {
    "owner": "Dmitry Kruglov",
    "email": "example@gmail.com",
    "email_on_failure": True,
    "email_on_retry": False,
    "retry": 3,
    "retry_delay": timedelta(minutes=1)
}


_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = 'ml-ops-start'
DATA_PATH = 'datasets/california_housing.pkl'
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", 
            "Population", "AveOccup", "Latitude", "Longitude"]
TARGET = "MedHouseVal"


models = dict(zip(['rf', 'lr', 'hgb'], [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()]))

    
def init() -> Dict[str, Any]:
    metrics = {}
    for model_name in models.keys():
        metrics[model_name] = dict()
        metrics[model_name]['start_tiemstamp'] = datetime.now().strftime("%Y%m%d %H:%M")
    _LOG.info("Train pipeline started")
    return metrics


def train_model(**kwargs) -> Dict[str, Any]:
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='init')
    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook("s3_connector")
    data = {}
    # Загрузить готовые данные с S3
    for name in ['X_train', 'X_test', 'y_train', 'y_test']:
        file = s3_hook.download_file(key=f"dataset/{name}.pkl", bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)
    # Обучить модели
    for m_name in metrics.keys():
        model = models[m_name]
        metrics[m_name]['train_start'] = datetime.now().strftime("%Y%m%d %H:%M")
        model.fit(data['X_train'], data['y_train'])
        prediction = model.predict(data['X_test'])
        metrics[m_name]['train_end'] = datetime.now().strftime("%Y%m%d %H:%M")
        # Посчитать метрики
        metrics[m_name]['r2_score'] = r2_score(data['y_test'], prediction)
        metrics[m_name]['rmse'] = mean_squared_error(data['y_test'], prediction)**0.5
        metrics[m_name]['mae'] = mean_absolute_error(data['y_test'], prediction)
    return metrics

def save_results(**kwargs) -> None:
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='train_model')
    # Сохранить результат на S3
    date = datetime.now().strftime("%Y_%m_%d_%H")
    s3_hook = S3Hook("s3_connector")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")
    for model_name in models.keys():
        metrics[model_name]['end_tiemstamp'] = datetime.now().strftime("%Y%m%d %H:%M")
        json_byte_object = json.dumps(metrics[model_name])
        resource.Object(BUCKET, f"results/{model_name}_{date}.json").put(Body=json_byte_object)
    _LOG.info("Success")


dag = DAG(
    dag_id = 'mlops_dag_4',
    schedule_interval = '0 1 * * *',
    start_date = days_ago(2),
    catchup = False,
    tags = ['mlops'],
    default_args = DEFAULT_ARGS
)


task_init = PythonOperator(task_id='init', 
                            python_callable=init, 
                            dag=dag,
                            provide_context=True)

task_train_model = PythonOperator(task_id='train_model', 
                                    python_callable=train_model, 
                                    dag=dag,
                                    provide_context=True)

task_save_results = PythonOperator(task_id='save_results', 
                                    python_callable=save_results, 
                                    dag=dag,
                                    provide_context=True)


task_init >>  task_train_model >> task_save_results