import sys
import warnings
import typer
from feature_manager.processing_data import processing_data
from feature_manager.feature_transform import FeatureTransforming
from models.DLRM import build_and_train_model
from utils.util import (
    SparkInitializer,
    return_or_load,
    load_simple_dict_config,
)

warnings.filterwarnings("ignore")
app = typer.Typer(pretty_exceptions_enable=False)

spark_initializer = SparkInitializer()
spark_connect = spark_initializer.spark
# spark_initializer.clear_cache()
feature_config = "/home/xuan-sang/total_venvs/DLRM_Amazon/configs/feature_manage.yaml"


@app.command()
def generating_features():
    config = return_or_load(feature_config, dict, load_simple_dict_config)
    print("*"*100)
    processing_data(
        spark=spark_connect,
        config=config
    )
    FeatureTransforming(config=config).run()
    print("DONE GENERATING FEATURES!")


@app.command()
def training_model():
    config = return_or_load(feature_config, dict, load_simple_dict_config)
    build_and_train_model(config=config)


def is_debugging() -> bool:
    return (gettrace := getattr(sys, "gettrace")) and gettrace()


if __name__ == "__main__":
    if not is_debugging():
        app()
    else:
        generating_features()
        training_model()
        spark_initializer.stop()
