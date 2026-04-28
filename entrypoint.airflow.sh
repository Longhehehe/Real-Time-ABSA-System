#!/bin/bash
set -e

# Fix ownership of mounted volumes when running as root.
AIRFLOW_UID=${AIRFLOW_UID:-50000}
if [ "$(id -u)" -eq 0 ]; then
  if [ -d /opt/airflow/logs ]; then
    chown -R "${AIRFLOW_UID}":0 /opt/airflow/logs
    chmod -R u+rwx /opt/airflow/logs
  fi

  if [ -d /opt/airflow/plugins ]; then
    chown -R "${AIRFLOW_UID}":0 /opt/airflow/plugins
  fi

  if [ -d /opt/airflow/dags ]; then
    chown -R "${AIRFLOW_UID}":0 /opt/airflow/dags
  fi
fi

# Execute Airflow using the upstream entrypoint when available.
if [ -x /entrypoint ]; then
  exec /entrypoint "$@"
fi

# Fallback for images without /entrypoint.
exec airflow "$@"
