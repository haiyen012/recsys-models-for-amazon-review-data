import pandas as pd
import numpy as np
from utils.util import return_or_load, load_simple_dict_config


class ReviewFeatureTransforming():
    def __init__(self, config):
        self.config = config

    def review_feature_run(self):
        df = self._get_review_data()
        df = self._review_feature_transforming(df)
        return df

    def _get_review_data(self):
        return pd.read_parquet(self.config["review_data_dir"])

    def _review_feature_transforming(self, df):
        df = df.copy()
        df = df.drop_duplicates(
            subset=["reviewer_id", "product_id"], keep="last")
        df.drop(columns=['product_color', 'product_size'], inplace=True)
        df = self._vote_feature(df=df)
        df = self._review_time_feature(df=df)
        df = self._product_rating_feature(df=df)
        df = self._verified_feature(df=df)
        df = self._have_image_feature(df=df)
        df = self._review_text_len_feature_transforming(df=df)
        df = self._mean_product_rating_feature_transforming(df=df)
        df = self._deviation_rating_product_feature_transforming(df=df)
        df = self._mean_review_rating_feature_transforming(df=df)
        df = self._deviation_reviewer_rating_feature_transforming(df=df)
        df = self._total_products_rated_feature_transforming(df=df)
        df = self._total_rating_users_feature_transforming(df=df)
        df = self._review_day_to_last_reviewed_day_feature_transforming(df=df)
        df = self._time_diff_feature_transforming(df=df)
        df = self._avg_purchase_interval_feature_transforming(df=df)
        df = self._days_since_last_review_feature_transforming(df=df)
        df = df[['product_id',
                 'review_time',
                 'reviewer_id',
                 'product_format',
                 'verified',
                 'vote',
                 'product_rating',
                 'review_text_len',
                 'mean_product_rating',
                 'mean_reviewer_rating',
                 'deviation_rating_product',
                 'deviation_reviewer_rating',
                 'have_image',
                 'total_products_rated',
                 'total_rating_users',
                 'review_day_to_last_reviewed_day',
                 'days_since_last_review',
                 'times_diff',
                 'avg_purchase_interval']]
        return df

    def _vote_feature(self, df):
        df = df.copy()
        df["vote"] = df["vote"].str.replace(',', '').fillna(0).astype(int)
        return df

    def _review_time_feature(self, df):
        df = df.copy()
        df['review_time'] = pd.to_datetime(
            df['review_time']).dt.strftime('%Y%m%d')
        df["review_time"] = pd.to_datetime(df["review_time"], format="%Y%m%d")
        return df

    def _product_rating_feature(self, df):
        df = df.copy()
        df["product_rating"] = df["product_rating"].astype(float).astype(int)
        return df

    def _verified_feature(self, df):
        df = df.copy()
        df["verified"] = df["verified"].replace({False: 0, True: 1})
        return df

    def _have_image_feature(self, df):
        df = df.copy()
        df["have_image"] = df["image"].notnull().astype(int)
        return df

    def _review_text_len_feature_transforming(self, df):
        df = df.copy()
        df["review_text_len"] = df["review_text"].apply(
            lambda x: len(x) if x is not None else 0)
        return df

    def _mean_product_rating_feature_transforming(self, df):
        df = df.copy()
        mean_product_rating_df = df.groupby(
            "product_id")["product_rating"].mean().reset_index()
        mean_product_rating_df.rename(
            columns={"product_rating": "mean_product_rating"}, inplace=True)
        return df.merge(mean_product_rating_df, on="product_id", how="inner")

    def _deviation_rating_product_feature_transforming(self, df):
        df = df.copy()
        df["deviation_rating_product"] = df["product_rating"] - \
            df["mean_product_rating"]
        return df

    def _mean_review_rating_feature_transforming(self, df):
        df = df.copy()
        mean_review_rating_df = df.groupby(
            "reviewer_id")["product_rating"].mean().reset_index()
        mean_review_rating_df.rename(
            columns={"product_rating": "mean_reviewer_rating"}, inplace=True)
        return df.merge(mean_review_rating_df, on="reviewer_id", how="inner")

    def _deviation_reviewer_rating_feature_transforming(self, df):
        df = df.copy()
        df["deviation_reviewer_rating"] = df["product_rating"] - \
            df["mean_reviewer_rating"]
        return df

    def _total_products_rated_feature_transforming(self, df):
        df = df.copy()
        total_products_rated = df.groupby("reviewer_id")["product_id"].count(
        ).reset_index().sort_values(by="product_id")
        total_products_rated.rename(
            columns={"product_id": "total_products_rated"}, inplace=True)
        return df.merge(total_products_rated, on="reviewer_id", how="inner")

    def _total_rating_users_feature_transforming(self, df):
        df = df.copy()
        total_rating_users = df.groupby("product_id")["reviewer_id"].nunique(
        ).reset_index().sort_values(by="reviewer_id")
        total_rating_users.rename(
            columns={"reviewer_id": "total_rating_users"}, inplace=True)
        return df.merge(total_rating_users, on="product_id", how="inner")

    def _review_day_to_last_reviewed_day_feature_transforming(self, df):
        df = df.copy()
        last_reviewed_day_df = df.groupby(
            "product_id")["review_time"].max().reset_index()
        last_reviewed_day_df.rename(
            columns={"review_time": "last_reviewed_day"}, inplace=True)
        df = df.merge(last_reviewed_day_df, on="product_id", how="inner")
        df["review_day_to_last_reviewed_day"] = (
            df["last_reviewed_day"] - df["review_time"]).dt.days
        return df.sort_values(by="review_time")

    def _time_diff_feature_transforming(self, df):
        df = df.copy()
        df["times_diff"] = df.groupby('product_id')[
            'review_time'].diff().dt.days
        df.times_diff = df.times_diff.fillna(0)
        return df

    def _avg_purchase_interval_feature_transforming(self, df):
        df = df.copy()
        average_time_diff_df = df.groupby(
            'product_id')['times_diff'].mean().reset_index()
        average_time_diff_df.rename(
            columns={"times_diff": "avg_purchase_interval"}, inplace=True)
        average_time_diff_df = average_time_diff_df.fillna(0)
        return df.merge(average_time_diff_df, on="product_id", how="inner")

    def _days_since_last_review_feature_transforming(self, df):
        df = df.copy()
        df["days_since_last_review"] = (
            df["review_time"].max() - df["last_reviewed_day"]).dt.days
        return df


class ProductFeatureTransforming():
    def __init__(self, config):
        self.config = config

    def product_feature_run(self):
        df = self._get_product_data()
        df = self._product_feature_transforming(df)
        return df

    def _get_product_data(self):
        return pd.read_parquet(self.config["product_data_dir"])

    def _product_feature_transforming(self, df):
        df = df.copy()
        df = self._product_description_feature_transforming(df=df)
        df = self._title_cat_feature_transforming(df=df)
        df = self._category_feature_transforming(df=df)
        df = self._price_feature_transforming(df=df)
        df = self._rank_feature_transforming(df=df)
        return df

    def _product_description_feature_transforming(self, df):
        df = df.copy()
        df["len_product_dscrp"] = df.product_description.apply(len)
        df.loc[df['len_product_dscrp'] == 0,
               'product_description'] = 'no_description'
        df.loc[(df['len_product_dscrp'] > 0) & (
            df['len_product_dscrp'] < 10), 'product_description'] = 'sketchy'
        df.loc[(df['len_product_dscrp'] >= 10) & (
            df['len_product_dscrp'] < 20), 'product_description'] = 'normal'
        df.loc[df['len_product_dscrp'] >= 20,
               'product_description'] = 'detailed'
        return df

    def _title_cat_feature_transforming(self, df):
        df = df.copy()
        df["title_len"] = df["title"].apply(
            lambda x: len(x) if x is not None else 0)
        df.loc[df['title_len'] == 0, 'title_cat'] = 'no_title'
        df.loc[(df['title_len'] > 0) & (df['title_len'] < 100),
               'title_cat'] = 'sketchy_title'
        df.loc[(df['title_len'] >= 100) & (df['title_len'] < 200),
               'title_cat'] = 'normal_title'
        df.loc[(df['title_len'] >= 200) & (df['title_len'] < 500),
               'title_cat'] = 'detailed_title'
        df.loc[df['title_len'] >= 500, 'title_cat'] = 'very_detailed_title'
        return df

    def _category_feature_transforming(self, df):
        df = df.copy()
        df = df[["product_id", "price", "brand_name",
                 "rank", "product_description", "title_cat"]]
        df[["rank", "category", "c", "d", "e", "f"]
           ] = df['rank'].str.split(' in ', expand=True)
        df = df[["product_id", "price", "brand_name", "rank",
                 "product_description", "title_cat", "category"]]
        df["category"] = df["category"].str.split(' \(').str[0]
        return df

    def _price_feature_transforming(self, df):
        df = df.copy()
        df['price'] = df['price'].str.replace('$', '')
        df['price'] = df['price'].str.replace(',', '')
        df = df[~df.price.str.contains(".a-section.a-spacing-mini")]
        df = df[~df.price.str.contains("\n\n\n<script")]
        df = df[~df.price.str.contains(".a-box-inner{background-color")]
        df['price'] = df['price'].replace('', np.nan)
        df['price'] = df['price'].astype(float)
        return df

    def _rank_feature_transforming(self, df):
        df = df.copy()
        df['rank'] = df['rank'].str.replace(',', '')
        df['rank'] = df['rank'].str.replace('[', '')
        df['rank'] = df['rank'].str.replace(']', '')
        df = df[~df['rank'].str.contains(">#")]
        df['rank'] = df['rank'].replace('', np.nan)
        df['rank'] = df['rank'].fillna(0)
        df['rank'] = df['rank'].astype(float).astype(int)
        return df


class FeatureTransforming(
    ReviewFeatureTransforming,
    ProductFeatureTransforming,
):
    def __init__(self, config):
        self.config = config
        self.save_path = self.config["feature_save_dir"]

    def _feature_transforming(self, df):
        df = df.copy()
        df = df.sort_values(by="review_time", ignore_index=True)
        df["review_year"] = df["review_time"].dt.year
        df.drop(columns=['review_time'], inplace=True)
        df['price'].fillna(df['price'].mean(), inplace=True)
        df['rank'] = df['rank'].fillna(0)
        df['brand_name'] = df['brand_name'].fillna("no_brand_name")
        df['product_description'] = df['product_description'].fillna(
            "no_description")
        df['title_cat'] = df['title_cat'].fillna("no_title")
        df['category'] = df['category'].fillna("no_category")
        df = df[['product_rating',
                 'review_year',
                 'verified',
                 'vote',
                 'review_text_len',
                 'mean_product_rating',
                 'mean_reviewer_rating',
                 'deviation_rating_product',
                 'deviation_reviewer_rating',
                 'have_image',
                 'total_products_rated',
                 'total_rating_users',
                 'review_day_to_last_reviewed_day',
                 'days_since_last_review',
                 'times_diff',
                 'avg_purchase_interval',
                 'price',
                 'rank',
                 'product_id',
                 'reviewer_id',
                 'product_format',
                 'product_description',
                 'title_cat',
                 'category',
                 'brand_name'
                 ]]
        return df

    def run(self):
        review_df = self.review_feature_run()
        product_df = self.product_feature_run()
        df = pd.merge(review_df, product_df, how="left", on="product_id")
        df = self._feature_transforming(df=df)
        df.to_csv(self.save_path, index=False)


if __name__ == "__main__":
    config_path = "./configs/feature_manage.yaml"
    config = return_or_load(config_path, dict, load_simple_dict_config)
    FeatureTransforming(config=config).run()
