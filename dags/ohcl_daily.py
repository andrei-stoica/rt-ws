from __future__ import annotations

import datetime

import pendulum
import logging
from os import path

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

data_dir = "/data"
tmp_dir = path.join(data_dir, "tmp")


def clean_data(**kargs):
    import pandas as pd
    from os import path
    from datetime import datetime

    task_instance = kargs.get("ti")

    # in case of a run spanning 2 days use xcom
    date = datetime.now().date()
    task_instance.xcom_push(key="date", value=date)

    symbol_file = path.join(tmp_dir, "symbols_valid_meta.csv")
    etfs_dir = path.join(tmp_dir, "etfs")
    stocks_dir = path.join(tmp_dir, "stocks")
    out_path = path.join(data_dir, f"ohcl_raw_{date}")

    symbols = pd.read_csv(symbol_file)
    etf_mapping = {"Y": True, "N": False}
    symbols.ETF = symbols.ETF.map(etf_mapping)

    dfs = []
    for i, symbol, etf, name in symbols[
        ["NASDAQ Symbol", "ETF", "Security Name"]
    ].itertuples():
        directory = etfs_dir if etf else stocks_dir
        filename = f"{symbol}.csv"

        df = pd.read_csv(path.join(directory, filename))
        df["Symbol"], df["Security Name"] = symbol, name
        dfs.append(
            df[
                [
                    "Symbol",
                    "Security Name",
                    "Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Volume",
                ]
            ]
        )
        if i % 1000 == 0:
            logging.info(f"Loaded {i}/{len(symbols)}")
    logging.info(f"Loaded all data")

    data = pd.concat(dfs)
    data.to_parquet(
        path=out_path,
        partition_cols=["Date"],
        index=False,
        engine="fastparquet",
    )
    logging.info(f"Written cleaned data to {out_path}")


def feature_engineering(**kargs):
    import pandas as pd
    from os import path

    task_instance = kargs.get("ti")
    date = task_instance.xcom_pull(key="date")

    in_path = path.join(data_dir, f"ohcl_raw_{date}")
    out_path = path.join(data_dir, f"ohcl_features_{date}")

    columns = ["Symbol", "Date", "Volume", "Close"]
    data = pd.read_parquet(in_path, columns=columns)
    logging.info(f"Read data from {in_path}")

    column_renaming = {
        "Volume": "vol_moving_avg",
        "Close": "adj_close_rolling_med",
    }
    rolling_avgs = (
        data.groupby("Symbol")
        .rolling(30, on="Date")[["Volume", "Close"]]
        .mean()
        .rename(columns=column_renaming)
        .reset_index()
        .dropna()
    )
    logging.info(f"Rolling averages computed for {list(column_renaming.keys())}")

    rolling_avgs.to_parquet(
        path=out_path,
        partition_cols=["Date"],
        index=False,
        engine="fastparquet",
    )
    logging.info(f"Written feature data to {out_path}")

def watcher():
    raise AirflowException("Failing run because of an upstream faliure")

##
# the RandomForestRegressor from the example code could not run on my
# machine with this dataset due to memory limitations. So the decision
# to switch was practical one.
#
# After some internet research, I discovered an article that compared
# multiple models at this exact task. CatBoost had the best perfomace.
# That being said, the perfomace it exhibited in that write-up was
# highly influenced by recent futures trading volumes. This dataset
# does not include volume for futures and options.
# It would be interesting to try some of the training methodologies from
# LLMs. I suspect adapting BERT's masked language model would be suitable
# here due to the reactive nature of the market. Knowledge of the future
# could help make sense of patterns. I would also like to add the
# derivatives market to the dataset since it showed to have great impact
# in the article stated earlier.
#
# SOURCES:
# [Trading volume prediction](https://medium.com/machine-learning-with-market-data/trading-volume-prediction-on-the-example-of-nasdaq-index-futures-6033de7ba716)
#
##
def model_training(**kargs):
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    from os import path, mkdir
    from catboost import CatBoostRegressor

    task_instance = kargs.get("ti")
    date = task_instance.xcom_pull(key="date")

    path_to_raw = path.join(data_dir, f"ohcl_raw_{date}")
    path_to_features = path.join(data_dir, f"ohcl_features_{date}")
    path_to_log = path.join(data_dir, f"ohcl_model_testing_losses.csv")
    path_to_models = path.join(data_dir, "models")
    path_to_new_model = path.join(path_to_models, f"{date}_catboost.cbm")

    volumes = pd.read_parquet(path_to_raw, columns=["Symbol", "Date", "Volume"])
    features = pd.read_parquet(path_to_features)
    data = pd.merge(features, volumes, on=["Date", "Symbol"])

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index("Date")
    logging.info(f"Loaded data from [{path_to_raw}, {path_to_features}]")

    # This follows the same data preperation as the example code.
    # However, if goal of this model is to predict the next days volume I
    # propose that the features should be from day *x* and the target should be
    # from day *x+1*.
    features = ["vol_moving_avg", "adj_close_rolling_med"]
    target = "Volume"

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = CatBoostRegressor(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_pred)
    train_mse = mean_squared_error(y_train, y_pred)

    y_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)

    logging.info(f"testing accuracy: (mae: {test_mae}, mse: {test_mse})")
    logging.info(f"training accuracy: (mae: {train_mae}, mse: {train_mse})")

    if not path.exists(path_to_models):
        mkdir(path_to_models)
    model.save_model(path_to_new_model)

    with open(path_to_log, "a") as f:
        if not path.exists(path_to_log):
            f.write("date, mae, mse")
        f.write(f"{date}, {test_mae}, {test_mse}")


## START DAG CREATION:
##
# This is set to run daily. I was thinking that a *real* pipeline such as this
# one would run only retrieve the latest day's data and update an existing 
# dataset. Then it would train or even update the weights of an existing model
# based on the new data. 
#
# Since this dataset is static, the current implementation retrieves the entire
# dataset every time and trains a new model.
##
with DAG(
    dag_id="ohcl_daily",
    schedule="0 0 * * *",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=60),
    tags=["daily", "ohcl", "model", "training"],
) as dag:
    ##
    # Task to download latest dataset. In a real environment this would update
    # existing dataset with the last days data.
    #
    # Ideally files would be hosted on S3 or similar instead of locally.
    #
    ##
    download = BashOperator(
        task_id="download_dataset",
        retries=3,
        bash_command=f"""mkdir -p {tmp_dir} \
        && kaggle datasets download -p {tmp_dir} -d jacksoncrow/stock-market-dataset \
        && unzip -o {tmp_dir}/stock-market-dataset.zip  -d {tmp_dir}
        """,
    )

    clean = PythonOperator(task_id="clean_data", python_callable=clean_data)
    features = PythonOperator(
        task_id="feature_engineering", python_callable=feature_engineering
    )
    training = PythonOperator(
        task_id="model_training", python_callable=model_training
    )

    cleanup_tmp = BashOperator(
        task_id="cleanup_tmp",
        bash_command=f"""[[ -d {tmp_dir} ]] && rm -r {tmp_dir}""",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    fail_watcher = PythonOperator(
        task_id="watcher",
        trigger_rule=TriggerRule.ONE_FAILED,
        retries=0,
        python_callable=watcher,
    )

    download >> clean >> features >> training
    clean >> cleanup_tmp

    [cleanup_tmp, training] >> fail_watcher



## END DAG CREATION:

if __name__ == "__main__":
    dag.test()
