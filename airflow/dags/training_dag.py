from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    'owner': 'nlp_team',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'sentiment_model_training',
    default_args=default_args,
    description='A daily training pipeline for Vietnamese Sentiment Analysis',
    schedule_interval=timedelta(days=1),            
    start_date=days_ago(1),
    tags=['nlp', 'sentiment', 'training'],
) as dag:

    start_task = BashOperator(
        task_id='start_training',
        bash_command='echo "Starting model training pipeline..."',
    )

    training_script_path = "train_pipeline.py" 
    
    train_model = BashOperator(
        task_id='train_model',
        bash_command=f'cd /opt/airflow/project && python {training_script_path}',
    )

    notify_success = BashOperator(
        task_id='notify_success',
        bash_command='echo "Training completed successfully. Model saved."',
    )

    start_task >> train_model >> notify_success
