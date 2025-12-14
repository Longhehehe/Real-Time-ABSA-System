"""
PhoBERT ABSA Training DAG with Auto-Retraining and Model Comparison
Train PhoBERT model, compare with old model, update only if better.
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
    description='Auto-retrain PhoBERT model with new data, compare and update if better',
    schedule_interval=None,  # Manual trigger
    start_date=days_ago(1),
    tags=['nlp', 'sentiment', 'phobert', 'absa', 'training', 'auto-retrain'],
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

    # Task 2: Check if new data exists, merge with old data
    merge_data = BashOperator(
        task_id='merge_training_data',
        bash_command='''
            echo "Checking and merging training data..."
            cd /opt/airflow/project
            python -c "
import os
import sys
sys.path.insert(0, '/opt/airflow/project')
from phobert_trainer import merge_datasets

OLD_DATA = '/opt/airflow/project/data/label/absa_grouped_vietnamese_test.xlsx'
NEW_DATA = '/opt/airflow/project/data/label/new_training_data.xlsx'
MERGED_DATA = '/opt/airflow/project/data/label/merged_training_data.xlsx'

# Check if new data exists
if os.path.exists(NEW_DATA):
    print('📊 New training data found!')
    merged_path = merge_datasets(OLD_DATA, NEW_DATA, MERGED_DATA)
    print(f'✅ Merged data saved to: {merged_path}')
else:
    print('ℹ️ No new data found, using existing data')
    # Copy old data as merged for consistency
    import shutil
    shutil.copy(OLD_DATA, MERGED_DATA)
    print(f'✅ Using original data: {OLD_DATA}')
"
        ''',
    )

    # Task 3: Train new model and compare with old
    train_and_compare = BashOperator(
        task_id='train_and_compare_models',
        bash_command='''
            echo "Training new model and comparing..."
            cd /opt/airflow/project
            python -c "
import sys
sys.path.insert(0, '/opt/airflow/project')
from phobert_trainer import train_and_compare

DATA_PATH = '/opt/airflow/project/data/label/merged_training_data.xlsx'
MODEL_DIR = '/opt/airflow/project/models/phobert_absa'

should_update, new_f1, old_f1 = train_and_compare(
    data_path=DATA_PATH,
    model_dir=MODEL_DIR,
    epochs=5,
    batch_size=16,
    min_improvement=0.01
)

# Write result to file for next task
with open('/tmp/model_comparison_result.txt', 'w') as f:
    f.write(f'{should_update},{new_f1:.4f},{old_f1:.4f}')

print(f'\\n📊 Final Result:')
print(f'   Should update: {should_update}')
print(f'   New F1: {new_f1:.4f}')
print(f'   Old F1: {old_f1:.4f}')
"
        ''',
        execution_timeout=timedelta(hours=2),
    )

    # Task 4: Check comparison result and notify
    notify_result = BashOperator(
        task_id='notify_training_result',
        bash_command='''
            echo "Reading comparison result..."
            if [ -f /tmp/model_comparison_result.txt ]; then
                RESULT=$(cat /tmp/model_comparison_result.txt)
                UPDATED=$(echo $RESULT | cut -d',' -f1)
                NEW_F1=$(echo $RESULT | cut -d',' -f2)
                OLD_F1=$(echo $RESULT | cut -d',' -f3)
                
                if [ "$UPDATED" = "True" ]; then
                    echo "🎉 SUCCESS: Model updated!"
                    echo "   New F1: $NEW_F1 (was: $OLD_F1)"
                else
                    echo "ℹ️ Model NOT updated (new model not significantly better)"
                    echo "   New F1: $NEW_F1, Old F1: $OLD_F1"
                fi
            else
                echo "⚠️ Comparison result file not found"
            fi
            echo "Training pipeline completed!"
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
sys.path.insert(0, 'app')
from absa_predictor import PhoBERTPredictor, SENTIMENT_MAP

predictor = PhoBERTPredictor()
if predictor.load_model():
    result = predictor.predict_single('Sản phẩm tốt, giao hàng nhanh')
    print('✅ Inference test passed!')
    for asp, val in list(result.items())[:3]:
        print(f'  {asp}: {SENTIMENT_MAP.get(val, val)}')
else:
    print('❌ Model loading failed!')
    exit(1)
"
        ''',
    )

    # Workflow Dependency
    check_gpu >> merge_data >> train_and_compare >> notify_result >> test_inference
