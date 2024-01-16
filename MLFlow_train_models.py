import json
import logging
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Literal
from sqlalchemy import create_engine

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

NAME = "Albertovich7" 
BUCKET = 'ml-ops-start'
FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup",
    "Latitude", "Longitude"
]
TARGET = "MedHouseVal"
EXPERIMENT_NAME = NAME
DAG_ID = NAME
DATA_PATH = f"{NAME}/datasets/california_housing.pkl"

models =  dict(zip(['rf', 'lr', 'hgb'], [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()]))
default_args = {
    "owner": "Dmitry Kruglov",
    "email": "example@gmail.com",
    "email_on_failure": True,
    "email_on_retry": False,
    "retry": 3,
    "retry_delay": timedelta(minutes=1)
}


dag = DAG(dag_id=DAG_ID,
          default_args=default_args,
          schedule_interval = '0 1 * * *',
          start_date = days_ago(2),
          catchup = False,
          tags = ['mlops'],
          )


def init() -> Dict[str, Any]:
    _LOG.info("Train pipeline started")
    metrics = {}
    metrics['start_tiemstamp'] = datetime.now().strftime("%Y%m%d %H:%M")
    
    exp_name = NAME
    experiment_id = mlflow.create_experiment(exp_name, artifact_location=f"s3://kda-mlflow-artifacts/{exp_name}")
    e_id = mlflow.get_experiment(experiment_id).experiment_id
    mlflow.set_experiment(exp_name)
    
    with mlflow.start_run(
            run_name="parent_run", 
            experiment_id = e_id, 
            description = "parent"
        ) as parent_run:
        metrics['experiment_name'] = exp_name
        metrics['experiment_id'] = e_id
        metrics['run_id'] = mlflow.active_run().info.run_id
        
    _LOG.info("Init task finished")
    return metrics


def get_data_from_postgres(**kwargs) -> Dict[str, Any]:
    _LOG.info("Data download started")
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='init')
    metrics['data_download_start'] = datetime.now().strftime("%Y%m%d %H:%M")
    
    pg_hook = PostgresHook('pg_connection')
    con = pg_hook.get_conn()
    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")
    
    data = pd.read_sql_query("SELECT * FROM california_housing", con)

    pickle_byte_obj = pickle.dumps(data)
    resource.Object(BUCKET, DATA_PATH).put(Body=pickle_byte_obj)
    
    metrics['data_download_end'] = datetime.now().strftime("%Y%m%d %H:%M")
    _LOG.info("Data download finished")
    
    return metrics


def prepare_data(**kwargs) -> Dict[str, Any]:
    _LOG.info("Data preparation started")
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='get_data')
    metrics['data_preparation_start'] = datetime.now().strftime("%Y%m%d %H:%M")
    
    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")
    
    file = s3_hook.download_file(key=DATA_PATH, bucket_name=BUCKET)
    data = pd.read_pickle(file)
    
    X, y = data[FEATURES], data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.fit_transform(X_test)

    for name, data in zip(['X_train', 'X_test', 'y_train', 'y_test'],
                          [X_train_fitted, X_test_fitted, y_train, y_test]):
        pickle_byte_obj = pickle.dumps(data)
        resource.Object(BUCKET, f"{NAME}/datasets/{name}.pkl").put(Body=pickle_byte_obj)
        
    metrics['data_preparation_end'] = datetime.now().strftime("%Y%m%d %H:%M")
    _LOG.info("Data preparation finished")
    
    return metrics

def train_mlflow_model(model: Any, name: str, X_train: np.array,
                       X_test: np.array, y_train: pd.Series,
                       y_test: pd.Series) -> None:

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    
    signature = infer_signature(X_test, prediction)
    model_info = mlflow.sklearn.log_model(model, name, signature=signature)
    mlflow.evaluate(
        model_info.model_uri,
        data=X_test,
        targets=y_test.values,
        model_type="regressor",
        evaluators=["default"],
    )


def train_model(**kwargs) -> Dict[str, Any]:
    _LOG.info("Train models started")
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='prepare_data')
    m_name = kwargs["model_name"]

    s3_hook = S3Hook("s3_connection")
    data = {}
    for name in ['X_train', 'X_test', 'y_train', 'y_test']:
        file = s3_hook.download_file(key=f"{NAME}/datasets/{name}.pkl", bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)

    with mlflow.start_run(
            run_name=m_name, 
            experiment_id = metrics['experiment_id'], 
            description = "child",
            nested=True
        ) as child_run:
            model = models[m_name]
            metrics[f"train_start_{m_name}"] = datetime.now().strftime("%Y%m%d %H:%M")
            train_mlflow_model(model, data['X_train_fitted'], data['X_test_fitted'], data['y_train'], data['y_test'])
            metrics[f"train_end_{m_name}"] = datetime.now().strftime("%Y%m%d %H:%M")

    _LOG.info("Train models started")
    return metrics


def save_results(**kwargs) -> None:
    _LOG.info("Save results started")
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids=["train_rf", "train_lr", "train_hgb"])[0]
    metrics["end_timestamp"] = datetime.now().strftime("%Y%m%d %H:%M")
    
    date = datetime.now().strftime("%Y%m%d %H:%M")
    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")
    json_byte_object = json.dumps(metrics)
    resource.Object(BUCKET, f"{NAME}/results/{metrics['model']}_{date}.json").put(Body=json_byte_object)



#################################### INIT DAG ####################################


task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)

task_get_data = PythonOperator(task_id="get_data",
                               python_callable=get_data_from_postgres,
                               dag=dag,
                               provide_context=True)

task_prepare_data = PythonOperator(task_id="prepare_data",
                                   python_callable=prepare_data,
                                   dag=dag,
                                   provide_context=True)

task_train_models = [
    PythonOperator(task_id=f"train_{model_name}",
                   python_callable=train_model,
                   dag=dag,
                   provide_context=True,
                   op_kwargs={"model_name": model_name})
    for model_name in models.keys()
]

task_save_results = PythonOperator(task_id="save_results",
                                   python_callable=save_results,
                                   dag=dag,
                                   provide_context=True)

task_init >> task_get_data >> task_prepare_data >> task_train_models >> task_save_results
