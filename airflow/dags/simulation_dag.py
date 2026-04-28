from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    'owner': 'nlp_team',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'retries': 0,
}

with DAG(
    'simulation_pipeline',
    default_args=default_args,
    description='Trigger Data Enrichment and Review Streaming Simulation',
    schedule_interval=None,                      
    start_date=days_ago(1),
    tags=['simulation', 'kafka'],
) as dag:

    enrich_task = BashOperator(
        task_id='enrich_data',
        bash_command='cd /opt/airflow/project && python kafka/data_enricher.py',
    )

    simulate_task = BashOperator(
        task_id='trigger_simulation',
        bash_command='cd /opt/airflow/project && python kafka/producer.py',
    )

    enrich_task >> simulate_task
