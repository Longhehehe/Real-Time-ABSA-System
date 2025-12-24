from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

# Default args for the DAG
default_args = {
    'owner': 'nlp_team',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'sentiment_model_training',
    default_args=default_args,
    description='A daily training pipeline for Vietnamese Sentiment Analysis',
    schedule_interval=timedelta(days=1), # Run daily
    start_date=days_ago(1),
    tags=['nlp', 'sentiment', 'training'],
) as dag:

    # Task 1: Print Start Message
    start_task = BashOperator(
        task_id='start_training',
        bash_command='echo "Starting model training pipeline..."',
    )

    # Task 2: Execute the Training Script
    # IMPORTANT: Adjust the path to where your train_pipeline.py is located
    # In Docker (via docker-compose), we mounted the root dir to /opt/airflow/project
    training_script_path = "train_pipeline.py" 
    
    # Note: In a real Airflow setup (e.g. Docker), the path would be mapped differently (e.g. /opt/airflow/dags/...).
    # Here we assume a local or compatible path for demonstration.
    train_model = BashOperator(
        task_id='train_model',
        bash_command=f'cd /opt/airflow/project && python {training_script_path}',
    )

    # Task 3: Notification (Simulated)
    notify_success = BashOperator(
        task_id='notify_success',
        bash_command='echo "Training completed successfully. Model saved."',
    )

    # Workflow Dependency
    start_task >> train_model >> notify_success
