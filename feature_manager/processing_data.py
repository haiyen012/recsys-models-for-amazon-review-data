from pathlib import Path
import shutil
import pyspark.sql.functions as F
from pyspark.sql.types import (
    BooleanType,
    ArrayType,
    LongType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)
from utils.util import (
    SparkInitializer,
    return_or_load,
    load_simple_dict_config,
)


digital_music_schema = StructType(
    [
        StructField("asin", StringType()),
        StructField("image", StringType()),
        StructField("overall", DoubleType()),
        StructField("reviewText", StringType()),
        StructField("reviewTime", StringType()),
        StructField("reviewerID", StringType()),
        StructField("reviewerName", StringType()),
        StructField(
            "style",
            StructType(
                [
                    StructField("Color:", StringType()),
                    StructField("Format:", StringType()),
                    StructField("Size:", StringType()),
                ]
            ),
        ),
        StructField("summary", StringType()),
        StructField("unixReviewTime", LongType()),
        StructField("verified", BooleanType()),
        StructField("vote", StringType()),
    ]
)


meta_data_schema = StructType(
    [
        StructField("also_buy", ArrayType(StringType())),
        StructField("also_view", ArrayType(StringType())),
        StructField("asin", StringType()),
        StructField("brand", StringType()),
        StructField("category", ArrayType(StringType())),
        StructField("date", StringType()),
        StructField("description", ArrayType(StringType())),
        StructField(
            "details",
            StructType(
                [
                    StructField("\n    Item Weight: \n    ", StringType()),
                    StructField("\n    Product Dimensions: \n    ", StringType()),
                    StructField("ASIN::", StringType()),
                    StructField("ASIN:", StringType()),
                    StructField("Apparel", StringType()),
                    StructField("Audio CD", StringType()),
                    StructField("Audio Cassette", StringType()),
                    StructField("Blu-ray Audio", StringType()),
                    StructField("DVD", StringType()),
                    StructField("DVD Audio", StringType()),
                    StructField("Label::", StringType()),
                    StructField("MP3 Music", StringType()),
                    StructField("Note on Boxed Sets::", StringType()),
                    StructField("Number of Discs::", StringType()),
                    StructField("Original Release Date::", StringType()),
                    StructField("Please Note::", StringType()),
                    StructField("Run Time::", StringType()),
                    StructField("SPARS Code::", StringType()),
                    StructField("Shipping Weight::", StringType()),
                    StructField("UPC::", StringType()),
                    StructField("Vinyl", StringType()),
                    StructField("Vinyl Bound", StringType()),
                ]
            ),
        ),
        StructField("feature", ArrayType(StringType())),
        StructField("fit", StringType()),
        StructField("imageURL", ArrayType(StringType())),
        StructField("imageURLHighRes", ArrayType(StringType())),
        StructField("main_cat", StringType()),
        StructField("price", StringType()),
        StructField("rank", StringType()),
        StructField("similar_item", StringType()),
        StructField("tech1", StringType()),
        StructField("tech2", StringType()),
        StructField("title", StringType()),
    ]
)


def process_review_data(spark, config, schema):
    raw_dir = config["raw_review_data_dir"]
    save_dir = config["review_data_dir"]
    if Path(save_dir).exists() and Path(save_dir).is_dir():
        shutil.rmtree(Path(save_dir))
    df_with_schema = spark.read.schema(schema).json(raw_dir)
    df_with_schema = df_with_schema.select(
        F.col("asin").alias("product_id"),
        "image",
        F.col("overall").alias("product_rating"),
        F.col("reviewText").alias("review_text"),
        F.col("reviewTime").alias("review_time"),
        F.col("reviewerID").alias("reviewer_id"),
        F.col("reviewerName").alias("reviewer_name"),
        F.col("style.Color:").alias("product_color"),
        F.col("style.Format:").alias("product_format"),
        F.col("Style.Size:").alias("product_size"),
        "summary",
        "verified",
        "vote",
    )
    df_with_schema.write.partitionBy("product_rating").parquet(
        save_dir, mode="overwrite"
    )


def process_meta_data(spark, config, schema):
    raw_dir = config["raw_product_data_dir"]
    save_dir = config["product_data_dir"]
    if Path(save_dir).exists() and Path(save_dir).is_dir():
        shutil.rmtree(Path(save_dir))
    df_with_schema = spark.read.schema(schema).json(raw_dir)
    df_with_schema = df_with_schema.select(
        "also_buy",
        "also_view",
        F.col("asin").alias("product_id"),
        F.col("brand").alias("brand_name"),
        "category",
        F.col("description").alias("product_description"),
        F.col("details.\n    Item Weight: \n    ").alias("item_weight"),
        F.col("details.\n    Product Dimensions: \n    ").alias("product_dimension"),
        F.col("details.Apparel").alias("Apparel"),
        F.col("details.Audio CD").alias("Audio CD"),
        F.col("details.Audio Cassette").alias("Audio Cassette"),
        F.col("details.Blu-ray Audio").alias("Blu-ray Audio"),
        F.col("details.DVD").alias("DVD"),
        F.col("details.DVD Audio").alias("DVD Audio"),
        F.col("details.MP3 Music").alias("MP3 Music"),
        F.col("details.Vinyl").alias("vinyl"),
        F.col("details.Vinyl Bound").alias("vinyl_bound"),
        "feature",
        "fit",
        F.col("imageURL").alias("image_url"),
        F.col("imageURLHighRes").alias("image_url_high_res"),
        "main_cat",
        "price",
        "rank",
        "title",
    )
    df_with_schema.write.mode("overwrite").parquet(save_dir)


def processing_data(spark, config):
    process_review_data(
        spark=spark,
        config=config,
        schema=digital_music_schema,
    )
    process_meta_data(
        spark=spark,
        config=config,
        schema=meta_data_schema,
    )


if __name__ == "__main__":
    config_path = "./configs/feature_manage.yaml"
    config = return_or_load(config_path, dict, load_simple_dict_config)

    spark_initializer = SparkInitializer(app_name="Processing Data")
    spark_connect = spark_initializer.spark
    processing_data(spark=spark_connect, config=config)
    spark_connect.stop()
