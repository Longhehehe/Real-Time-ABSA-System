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
    schedule_interval=None, # Manual trigger only
    start_date=days_ago(1),
    tags=['simulation', 'kafka'],
) as dag:

    # Task 1: enrich_data
    # Runs kafka/data_enricher.py to generate new random product IDs
    enrich_task = BashOperator(
        task_id='enrich_data',
        bash_command='cd /opt/airflow/project && python kafka/data_enricher.py',
    )

    # Task 2: trigger_simulation
    # Runs kafka/producer.py to stream 50 reviews to Kafka
    simulate_task = BashOperator(
        task_id='trigger_simulation',
        bash_command='cd /opt/airflow/project && python kafka/producer.py',
    )

    enrich_task >> simulate_task
