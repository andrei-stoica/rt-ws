FROM apache/airflow:2.6.1

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install fastparquet kaggle scikit-learn catboost

COPY ./kaggle.json /home/airflow/.kaggle/kaggle.json

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends unzip
USER airflow
