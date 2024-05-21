# from keras.metrics import Recall
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

from utils.util import load_simple_dict_config, return_or_load

BATCH_SIZE = 256


def dense_feature_processing(df, columns):
    df = df.copy()
    dense_cols = [c for c in columns if "I" in c]
    df[dense_cols] = preprocessing.StandardScaler().fit_transform(df[dense_cols])
    return df


def cat_feature_processing(df, columns):
    df = df.copy()
    cat_cols = [c for c in columns if "C" in c]
    mappings = {
        col: dict(zip(values, range(len(values))))
        for col, values in map(lambda col: (col, df[col].unique()), cat_cols)
    }
    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping.get)
    return df


def data_to_tensor_type(df, dense_cols, cat_cols):
    df = df.copy()
    ds = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(df[dense_cols].values, tf.float32),
                    tf.cast(df[cat_cols].values, tf.int32),
                )
            ),
            tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(
                        tf.keras.utils.to_categorical(
                            df["label"].values, num_classes=5
                        ),
                        tf.float32,
                    )
                )
            ),
        )
    ).shuffle(buffer_size=2048)
    return ds


def MLP(arch, activation="relu", out_activation=None):
    mlp = tf.keras.Sequential()

    for units in arch[:-1]:
        mlp.add(tf.keras.layers.Dense(units, activation=activation))

    mlp.add(tf.keras.layers.Dense(arch[-1], activation=out_activation))

    return mlp


class SecondOrderFeatureInteraction(tf.keras.layers.Layer):
    def __init__(self, self_interaction=False):
        super(SecondOrderFeatureInteraction, self).__init__()
        self.self_interaction = self_interaction

    def call(self, inputs):
        batch_size = tf.shape(inputs[0])[0]
        concat_features = tf.stack(inputs, axis=1)

        dot_products = tf.matmul(concat_features, concat_features, transpose_b=True)

        ones = tf.ones_like(dot_products)
        mask = tf.linalg.band_part(ones, 0, -1)
        out_dim = int(len(inputs) * (len(inputs) + 1) / 2)

        if not self.self_interaction:
            mask = mask - tf.linalg.band_part(ones, 0, 0)
            out_dim = int(len(inputs) * (len(inputs) - 1) / 2)

        flat_interactions = tf.reshape(
            tf.boolean_mask(dot_products, mask), (batch_size, out_dim)
        )
        return flat_interactions


class DLRM(tf.keras.Model):
    def __init__(
        self,
        embedding_sizes,
        embedding_dim,
        arch_bot,
        arch_top,
        self_interaction,
    ):
        super(DLRM, self).__init__()
        self.emb = [
            tf.keras.layers.Embedding(size, embedding_dim) for size in embedding_sizes
        ]

        self.bot_nn_layers = []
        for i in range(len(arch_bot)):
            self.bot_nn_layers.append(
                tf.keras.layers.Dense(arch_bot[i], activation=None)
            )
            if i < len(arch_bot) - 1:
                self.bot_nn_layers.append(tf.keras.layers.ReLU())

        self.bot_nn = tf.keras.Sequential(self.bot_nn_layers)

        # self.bot_nn = MLP(arch_bot, out_activation='relu')

        self.top_nn = MLP(arch_top, out_activation="softmax")

        self.interaction_op = SecondOrderFeatureInteraction(self_interaction)

    def call(self, input):
        input_dense, input_cat = input
        emb_x = [E(x) for E, x in zip(self.emb, tf.unstack(input_cat, axis=1))]
        dense_x = self.bot_nn(input_dense)

        Z = self.interaction_op(emb_x + [dense_x])
        z = tf.concat([dense_x, Z], axis=1)
        p = self.top_nn(z)

        return p


def spliting_data(config):
    columns = [
        "label",
        *(f"I{i}" for i in range(1, 18)),
        *(f"C{i}" for i in range(1, 8)),
    ]
    df = pd.read_csv(config["feature_save_dir"]).fillna(0)
    df.columns = columns
    df["label"] = df["label"] - 1

    df_test = df[df["I1"] == 2018]
    df_val = df[df["I1"] == 2017]
    df_train = df[~df["I1"].isin([2017, 2018])]

    df_train = dense_feature_processing(df_train, columns=columns)
    df_val = dense_feature_processing(df_val, columns=columns)
    df_test = dense_feature_processing(df_test, columns=columns)

    df_train = cat_feature_processing(df_train, columns=columns)
    df_val = cat_feature_processing(df_val, columns=columns)
    df_test = cat_feature_processing(df_test, columns=columns)
    return df, df_train, df_val, df_test


def get_emb_counts(df):
    label_counts = df.groupby("label")["I1"].count()
    print(f"Baseline: {max(label_counts.values) / sum(label_counts.values) * 100}%")
    dense_cols = [c for c in df.columns if "I" in c]
    cat_cols = [c for c in df.columns if "C" in c]
    emb_counts = [len(df[c].unique()) for c in cat_cols]
    return dense_cols, cat_cols, emb_counts


def plot_results(history):
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Plot training and validation loss
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def build_and_train_model(config):
    df, df_train, df_val, df_test = spliting_data(config=config)
    dense_cols, cat_cols, emb_counts = get_emb_counts(df=df)

    ds_train = data_to_tensor_type(df_train, dense_cols=dense_cols, cat_cols=cat_cols)
    ds_valid = data_to_tensor_type(df_val, dense_cols=dense_cols, cat_cols=cat_cols)
    ds_test = data_to_tensor_type(df_test, dense_cols=dense_cols, cat_cols=cat_cols)

    model = DLRM(
        embedding_sizes=emb_counts,
        embedding_dim=5,
        arch_bot=[8, 5],
        arch_top=[128, 64, 32, 16, 5],
        self_interaction=False,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    history = model.fit(
        ds_train.batch(BATCH_SIZE),
        validation_data=ds_valid.batch(BATCH_SIZE),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
        ],
        epochs=3,
        verbose=1,
    )

    results = model.evaluate(ds_test.batch(BATCH_SIZE))
    print(f"Loss {results[0]}, Accuracy {results[1]}")
    plot_results(history=history)


if __name__ == "__main__":
    config_path = "./configs/feature_manage.yaml"
    config = return_or_load(config_path, dict, load_simple_dict_config)
    build_and_train_model(config=config).run()
