"""
PhoBERT ABSA Training DAG with Multi-Polarity Support
Train PhoBERT model with multi-label sentiment classification.
"""
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os

# Default args for the DAG
default_args = {
    'owner': 'nlp_team',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
    'execution_timeout': timedelta(hours=3),
}

# Define the DAG
with DAG(
    'phobert_absa_training',
    default_args=default_args,
    description='Train PhoBERT Multi-Polarity ABSA model from labeled/ folder',
    schedule_interval=None,  # Manual trigger
    start_date=days_ago(1),
    tags=['nlp', 'sentiment', 'phobert', 'absa', 'training', 'multipolarity'],
    catchup=False,
) as dag:

    # Task 1: Check GPU availability
    check_gpu = BashOperator(
        task_id='check_gpu',
        bash_command='''
            echo "Checking GPU availability..."
            python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
        ''',
    )

    # Task 2: List labeled files
    check_data = BashOperator(
        task_id='check_labeled_data',
        bash_command='''
            echo "Checking labeled data files..."
            cd /opt/airflow/project
            ls -la labeled/ | head -20
            echo "Total files:"
            ls labeled/*.xlsx 2>/dev/null | wc -l
        ''',
    )

    # Task 3: Train Multi-Polarity Model (OPTIMIZED PARAMETERS)
    train_model = BashOperator(
        task_id='train_multipolarity_model',
        bash_command='''
            echo "Training PhoBERT Multi-Polarity ABSA model (FULL TRAINING)..."
            cd /opt/airflow/project
            python -c "
import sys
sys.path.insert(0, '/opt/airflow/project')
from phobert_trainer_multipolarity import train_model_multipolarity

# Use labeled/ folder directly (contains xlsx files)
DATA_PATH = '/opt/airflow/project/labeled'
MODEL_DIR = '/opt/airflow/project/models/phobert_absa_multipolarity'

# OPTIMIZED HYPERPARAMETERS FOR BEST RESULTS:
# - epochs=25: More epochs for better convergence
# - batch_size=8: Smaller batch = more updates per epoch (better generalization)
# - learning_rate=2e-5: Standard for PhoBERT fine-tuning
# - max_length=256: Full context for reviews

output_dir = train_model_multipolarity(
    data_path=DATA_PATH,
    output_dir=MODEL_DIR,
    epochs=25,           # More training epochs
    batch_size=8,        # Smaller batch for more gradient updates
    learning_rate=2e-5,  # Standard PhoBERT learning rate
    max_length=256       # Full context preservation
)

print(f'Training complete! Model saved to: {output_dir}')
"
        ''',
        execution_timeout=timedelta(hours=4),  # Increased timeout for full training
    )

    # Task 4: Verify model was saved
    verify_model = BashOperator(
        task_id='verify_model',
        bash_command='''
            echo "Verifying trained model..."
            cd /opt/airflow/project
            ls -la models/phobert_absa_multipolarity/
            if [ -f models/phobert_absa_multipolarity/phobert_absa_multipolarity.pt ]; then
                echo "Model file found!"
                cat models/phobert_absa_multipolarity/config.json
            else
                echo "Model file not found!"
                exit 1
            fi
        ''',
    )

    # Task 5: Test model inference
    test_inference = BashOperator(
        task_id='test_model_inference',
        bash_command='''
            echo "Testing model inference..."
            cd /opt/airflow/project
            python -c "
import sys
sys.path.insert(0, '/opt/airflow/project')
from phobert_trainer_multipolarity import predict_multipolarity

test_texts = [
    'San pham tot, giao hang nhanh, dong goi can than',
    'Chat luong kem, ship cham, that vong'
]

results = predict_multipolarity(
    test_texts,
    model_path='/opt/airflow/project/models/phobert_absa_multipolarity'
)

print('Inference test results:')
for i, res in enumerate(results):
    print(f'Text {i+1}:')
    for asp, info in list(res.items())[:3]:
        if info.get('mentioned'):
            print(f'  {asp}: {info.get(\"sentiments\", [])}')
print('Inference test passed!')
"
        ''',
    )

    # Workflow Dependency
    check_gpu >> check_data >> train_model >> verify_model >> test_inference
