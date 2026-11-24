import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import Dense, Dropout

# ---- Load data (upload train.csv and train_bounding_boxes.csv to Colab first) ----
train = pd.read_csv("train.csv")
boxes = pd.read_csv("train_bounding_boxes.csv")

# ---- Build one row per study from the bounding boxes ----
# Boxes exist ONLY for fractured studies, so we train on those 235 studies
# and predict WHICH vertebrae (C1-C7) are fractured.
features = boxes.groupby("StudyInstanceUID").agg(
    n_boxes=("x", "size"),
    x_mean=("x", "mean"),
    y_mean=("y", "mean"),
    w_mean=("width", "mean"),
    h_mean=("height", "mean"),
    slice_min=("slice_number", "min"),
    slice_max=("slice_number", "max"),
    slice_mean=("slice_number", "mean"),
).reset_index()
features["slice_span"] = features["slice_max"] - features["slice_min"]

dataset = features.merge(train, on="StudyInstanceUID")

vertebrae = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
feature_cols = ["n_boxes", "x_mean", "y_mean", "w_mean", "h_mean",
                "slice_min", "slice_max", "slice_mean", "slice_span"]

X = dataset[feature_cols].values
y = dataset[vertebrae].values            # 7 labels per study (multi-label)

# ---- Train / test split and scaling ----
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)      # fit on training data only
x_test = sc.transform(x_test)

# ---- Model ----
model = Sequential()
model.add(Dense(units=32, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.2))

model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=7, activation='sigmoid'))   # one probability per vertebra

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.1, epochs=100, batch_size=16)

# ---- Prediction ----
y_prob = model.predict(x_test)
y_pred = (y_prob > 0.5).astype(int)

# ---- Evaluation: ROC-AUC per vertebra ----
auc = []
for i, v in enumerate(vertebrae):
    if 0 < y_test[:, i].sum() < len(y_test):
        auc.append(roc_auc_score(y_test[:, i], y_prob[:, i]))
    else:
        auc.append(np.nan)               # only one class in this small test set
    print(v, "AUC:", auc[-1])

# ---- Plots ----
plt.plot(history.history['loss'], color='red', label='Training Loss')
plt.plot(history.history['val_loss'], color='blue', label='Validation Loss')
plt.title('Cervical Spine Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.bar(vertebrae, np.nan_to_num(auc), color='blue')
plt.axhline(0.5, color='red', linestyle='--', label='Random guess')
plt.title('ROC-AUC per Vertebra')
plt.xlabel('Vertebra')
plt.ylabel('AUC')
plt.legend()
plt.show()
