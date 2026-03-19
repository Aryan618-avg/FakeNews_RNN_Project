from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_csv("data/reviews.csv")

print("Label Distribution:")
print(df["label"].value_counts())
print(df.columns)
print(df.shape)

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])  
print("Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# CG -> 0 , OR -> 1 (or vice versa automatically)

# Remove missing values
df = df.dropna(subset=["text_"])

# Ensure text is string
df["text_"] = df["text_"].astype(str)

X = df["text_"]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

import numpy as np

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weight_dict)

# Tokenization
vocab_size = 10000
max_length = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

# Build model
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    LSTM(128),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

# Train
model.fit(
    X_train_pad,
    y_train,
    epochs=2,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    class_weight=class_weight_dict
)

# Evaluate
y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
model.save("model/review_model.h5")

with open("model/review_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Review model saved successfully!")