import random
import yaml
import pandas as pd
from pyspark.sql import SparkSession
import typing


def return_or_load(object_or_path, object_type, load_func):
    if isinstance(object_or_path, object_type):
        return object_or_path
    return load_func(object_or_path)


def load_simple_dict_config(path_config: str) -> dict:
    with open(path_config) as f:
        config = yaml.safe_load(f)
    return config



class SparkInitializer:
    def __init__(self, app_name="MySparkApp", master="local[*]", config=None):
        self.app_name = app_name
        self.master = master
        self.config = config if config else {}
        self.config["spark.local.dir"] = f"/tmp/pyspark/{random.randint(0, 1000000)}"
        self.spark = self._init_spark()

    def _init_spark(self):
        spark = SparkSession.builder.appName(self.app_name).master(self.master)

        for key, value in self.config.items():
            spark = spark.config(key, value)

        return spark.getOrCreate()

    def stop(self):
        self.spark.catalog.clearCache()
        self.spark.stop()


def split_batches(x: typing.Any, batch_size: int) -> typing.List[typing.Any]:
    length = len(x)
    if batch_size == -1 or length < batch_size:
        return [x]
    else:
        return [x[i : i + batch_size] for i in range(0, length, batch_size)]


def anti_join(df1, df2, on_columns=[]):
    assert len(on_columns) > 0
    # perform outer join
    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        target_df = df1[on_columns]
        target_df["index"] = df1.index

        outer = target_df.merge(df2[on_columns], how="outer", indicator=True)
        # perform anti-join
        anti_join = outer[(outer._merge == "left_only")].drop("_merge", axis=1)
        anti_join["index"] = anti_join["index"].astype(int)
        anti_join = anti_join.set_index("index")
        return df1.loc[anti_join.index]
    else:
        return df1.join(df2[on_columns], how="leftanti")
