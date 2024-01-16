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


def create_dag(dag_id:str, m_name: Literal['rf', 'lr', 'hgb']):
    def init() -> Dict[str, Any]:
        metrics = {}
        metrics['model'] = m_name
        metrics['start_tiemstamp'] = datetime.now().strftime("%Y%m%d %H:%M")
        _LOG.info("Train pipeline started")
        return metrics
    
    
    def get_data_from_postgres(**kwargs) -> Dict[str, Any]:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='init')

        return metrics
        

    def prepare_data(**kwargs) -> Dict[str, Any]:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='get_data')
        return metrics


    def train_model(**kwargs) -> Dict[str, Any]:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='prepare_data')
        m_name = metrics['model']
        # Использовать созданный ранее S3 connection
        s3_hook = S3Hook("s3_connector")
        data = {}
        # Загрузить готовые данные с S3
        for name in ['X_train', 'X_test', 'y_train', 'y_test']:
            file = s3_hook.download_file(key=f"dataset/{name}.pkl", bucket_name=BUCKET)
            data[name] = pd.read_pickle(file)
        # Обучить модель    
        model = models[m_name]
        metrics['train_start'] = datetime.now().strftime("%Y%m%d %H:%M")
        model.fit(data['X_train'], data['y_train'])
        prediction = model.predict(data['X_test'])
        metrics['train_end'] = datetime.now().strftime("%Y%m%d %H:%M")
        # Посчитать метрики
        metrics['r2_score'] = r2_score(data['y_test'], prediction)
        metrics['rmse'] = mean_squared_error(data['y_test'], prediction)**0.5
        metrics['mae'] = mean_absolute_error(data['y_test'], prediction)

        return metrics

    def save_results(**kwargs) -> None:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='train_model')
        
        metrics['end_tiemstamp'] = datetime.now().strftime("%Y%m%d %H:%M")
        # Сохранить результат на S3
        date = datetime.now().strftime("%Y_%m_%d_%H")
        s3_hook = S3Hook("s3_connector")
        session = s3_hook.get_session("ru-central1")
        resource = session.resource("s3")
        json_byte_object = json.dumps(metrics)
        resource.Object(BUCKET, f"results/{metrics['model']}_{date}.json").put(Body=json_byte_object)
        _LOG.info("Success")


    dag = DAG(
        dag_id = dag_id,
        schedule_interval = '0 1 * * *',
        start_date = days_ago(2),
        catchup = False,
        tags = ['mlops'],
        default_args = DEFAULT_ARGS
    )
    
    with dag:

        task_init = PythonOperator(task_id='init', 
                                   python_callable=init, 
                                   dag=dag,
                                   provide_context=True)

        task_get_data = PythonOperator(task_id='get_data', 
                                       python_callable=get_data_from_postgres, 
                                       dag=dag,
                                       provide_context=True)

        task_prepare_data = PythonOperator(task_id='prepare_data', 
                                           python_callable=prepare_data, 
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


        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results
        
        
for model_name in models.keys():
    create_dag(f'{model_name}_train', model_name)