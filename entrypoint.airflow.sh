#!/bin/bash
set -e

# Fix ownership of mounted volumes for Airflow user (UID 50000)
if [ -d /opt/airflow/logs ]; then
  chown -R 50000:0 /opt/airflow/logs
  chmod -R u+rwx /opt/airflow/logs
fi

if [ -d /opt/airflow/plugins ]; then
  chown -R 50000:0 /opt/airflow/plugins
fi

if [ -d /opt/airflow/dags ]; then
  chown -R 50000:0 /opt/airflow/dags
fi

# Execute the original Airflow command
exec "$@"
