import numpy as np
import pandas as pd
import re
import nltk
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ----------------------------
# DOWNLOAD STOPWORDS
# ----------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ----------------------------
# TEXT CLEANING FUNCTION
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("data/comprehensive_news_balanced.csv")

print("Columns in dataset:")
print(df.columns)

# Use cleaned_text if available, otherwise clean text column
if "cleaned_text" in df.columns:
    df["text"] = df["cleaned_text"]
else:
    df["text"] = df["text"].apply(clean_text)

# Convert label to numeric
df["label"] = df["label"].map({"fake": 0, "real": 1})

X = df["text"]
y = df["label"]

print("\nLabel Distribution:")
print(y.value_counts())

# ----------------------------
# TOKENIZATION
# ----------------------------
max_words = 15000
max_len = 250

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post')

# Save tokenizer
with open("model/news_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# ----------------------------
# TRAIN TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y, test_size=0.2, random_state=42
)

# ----------------------------
# BUILD ADVANCED RNN MODEL
# ----------------------------
model = Sequential()

model.add(Embedding(input_dim=max_words, output_dim=128))

model.add(Bidirectional(LSTM(64, return_sequences=True)))

model.add(GRU(32))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# CALLBACKS
# ----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "model/best_news_model.h5",
    monitor='val_accuracy',
    save_best_only=True
)

# ----------------------------
# TRAIN MODEL
# ----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint]
)

# ----------------------------
# EVALUATION
# ----------------------------
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------------------------
# ACCURACY GRAPH
# ----------------------------
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(['Train', 'Validation'])
plt.show()

# ----------------------------
# SAVE MODEL
# ----------------------------
model.save("model/news_rnn_model.h5")

print("\nNews Model training completed and saved successfully!")