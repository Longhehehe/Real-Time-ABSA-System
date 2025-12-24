"""
Airflow API Client for Streamlit Integration
Triggers DAGs and monitors their status via Airflow REST API.
"""
import requests
import time
from typing import Dict, Optional, Tuple

# Airflow API Configuration
AIRFLOW_BASE_URL = "http://airflow-webserver:8080"
AIRFLOW_API_URL = f"{AIRFLOW_BASE_URL}/api/v1"
AIRFLOW_USERNAME = "admin"
AIRFLOW_PASSWORD = "admin"

# DAG Configuration
DAG_ID = "realtime_absa_pipeline"


def get_auth():
    """Get authentication tuple for requests."""
    return (AIRFLOW_USERNAME, AIRFLOW_PASSWORD)


def trigger_dag(product_id: str, product_url: str = "", max_reviews: int = 50, reviews: list = None) -> Tuple[bool, str]:
    """
    Trigger the realtime_absa_pipeline DAG with parameters.
    
    Args:
        product_id: Unique product identifier
        product_url: Optional Lazada URL (for crawling in DAG)
        max_reviews: Max reviews to crawl (if URL provided)
        reviews: Pre-crawled reviews list (from Streamlit)
    
    Returns:
        Tuple of (success, dag_run_id or error_message)
    """
    url = f"{AIRFLOW_API_URL}/dags/{DAG_ID}/dagRuns"
    
    payload = {
        "conf": {
            "product_id": product_id,
            "product_url": product_url,
            "max_reviews": max_reviews,
            "reviews": reviews or []  # Send pre-crawled reviews
        }
    }
    
    try:
        response = requests.post(
            url,
            json=payload,
            auth=get_auth(),
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code in [200, 201]:
            data = response.json()
            dag_run_id = data.get("dag_run_id", "")
            print(f"✅ DAG triggered successfully: {dag_run_id}")
            return True, dag_run_id
        else:
            error = response.text
            print(f"❌ Failed to trigger DAG: {error}")
            return False, error
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False, str(e)


def get_dag_run_status(dag_run_id: str) -> Dict:
    """
    Get the status of a DAG run.
    
    Returns:
        Dict with status info or empty dict on error
    """
    url = f"{AIRFLOW_API_URL}/dags/{DAG_ID}/dagRuns/{dag_run_id}"
    
    try:
        response = requests.get(
            url,
            auth=get_auth(),
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {}
            
    except requests.exceptions.RequestException:
        return {}


def wait_for_dag_completion(dag_run_id: str, timeout: int = 300, poll_interval: int = 5) -> Tuple[bool, str]:
    """
    Wait for DAG run to complete.
    
    Returns:
        Tuple of (success, final_state)
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status = get_dag_run_status(dag_run_id)
        
        if not status:
            time.sleep(poll_interval)
            continue
        
        state = status.get("state", "")
        
        if state == "success":
            return True, "success"
        elif state in ["failed", "upstream_failed"]:
            return False, state
        
        # Still running
        time.sleep(poll_interval)
    
    return False, "timeout"


def get_task_instances(dag_run_id: str) -> list:
    """
    Get task instances for a DAG run (for progress tracking).
    """
    url = f"{AIRFLOW_API_URL}/dags/{DAG_ID}/dagRuns/{dag_run_id}/taskInstances"
    
    try:
        response = requests.get(
            url,
            auth=get_auth(),
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("task_instances", [])
        return []
            
    except requests.exceptions.RequestException:
        return []
