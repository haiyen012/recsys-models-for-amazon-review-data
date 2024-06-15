import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tqdm
from deepctr.feature_column import DenseFeat, SparseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical

from models.base.custom_FEFM_deep_ctr import DeepFEFM

print("*" * 20, "Processing data", "*" * 20)
columns = [
    "original_label",
    *(f"I{i}" for i in range(1, 18)),
    *(f"C{i}" for i in range(1, 8)),
]
data = pd.read_csv("./data/processed/features.csv").fillna(0)
data.columns = columns
min_val = data["original_label"].min()
max_val = data["original_label"].max()
# data['label'] = (data['original_label'] - min_val) / (max_val - min_val)
data["label"] = data["original_label"] - 1
data.drop(columns=["original_label"], inplace=True)


sparse_features = ["C" + str(i) for i in range(1, 8)]
dense_features = ["I" + str(i) for i in range(1, 18)]

data[sparse_features] = data[sparse_features].fillna(
    "-1",
)
data[dense_features] = data[dense_features].fillna(
    0,
)
target = ["label"]

print(data.head())

print("*" * 20, "Processing features", "*" * 20)

for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = data[feat].astype(str)
    data[feat] = lbe.fit_transform(data[feat])

mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

# step 3: generate feature cols
fixlen_feature_columns = [
    SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
    for i, feat in enumerate(sparse_features)
] + [
    DenseFeat(
        feat,
        1,
    )
    for feat in dense_features
]
print("*" * 200)
print(fixlen_feature_columns)
dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# step 4: Generate the training samples and train the model
data["I1"] = round(data["I1"], 1)
print("*" * 20, "Training model", "*" * 20)
train = data[~data["I1"].isin([0.8, 1])]
val = data[data["I1"] == 0.8]
test = data[data["I1"] == 1]
print(len(train), len(val), len(test))
print(test["label"].value_counts())
train_model_input = {name: train[name].values for name in feature_names}
val_model_input = {name: val[name].values for name in feature_names}
test_model_input = {name: test[name].values for name in feature_names}


class TQDMCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.progress_bar = tqdm.tqdm(
            total=self.params["steps"], desc=f"Epoch {epoch + 1}", unit="batch"
        )

    def on_batch_end(self, batch, logs=None):
        self.progress_bar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.close()


model = DeepFEFM(
    linear_feature_columns, dnn_feature_columns, task="multiclass", num_classes=5
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print(train[target].values)

train_labels_one_hot = to_categorical(train[target].values, num_classes=5)
val_labels_one_hot = to_categorical(val[target].values, num_classes=5)

history = model.fit(
    train_model_input,
    train_labels_one_hot,
    batch_size=256,
    epochs=5,
    validation_data=(val_model_input, val_labels_one_hot),
    callbacks=[TQDMCallback()],
)

model.save_weights("my_model_weight.h5")
test_predictions = model.predict(test_model_input, batch_size=256)
predicted_labels = tf.argmax(test_predictions, axis=1).numpy()
true_labels = tf.argmax(
    to_categorical(test[target].values, num_classes=5), axis=1
).numpy()
accuracy = (predicted_labels == true_labels).mean()
print(f"Predicted labels: {predicted_labels}")
print(f"Accuracy: {accuracy}")

# Add predicted labels to the test DataFrame
test["predicted_labels"] = predicted_labels  # Add predicted labels
# Add ground truth labels (optional, for reference)
test["true_labels"] = true_labels  # Add ground truth labels
test.to_csv("test_predictions.csv", index=False)

test_loss, test_accuracy = model.evaluate(
    test_model_input,
    to_categorical(
        test[target].values, num_classes=5
    ),  # Ensure labels are one-hot encoded
    batch_size=256,
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# step 5: plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
