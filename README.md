# 🧠 AI Fake News & Fake Review Detection System

A deep learning-based web application that detects **Fake News 📰** and **Fake Reviews ⭐** using RNN/LSTM models.  
Built as a **final year project** with real-world datasets and explainable AI.

---

## 🚀 Features

✅ Fake News Detection using RNN (LSTM + Bidirectional)  
✅ Fake Review Detection using LSTM  
✅ URL-based News Extraction (paste article link)  
✅ Confidence Score with Progress Bar  
✅ Explainable AI (LIME) – shows important words affecting prediction  
✅ Clean and Interactive UI using Streamlit  

---

## 🧠 Tech Stack

- **Programming Language:** Python  
- **Deep Learning:** TensorFlow / Keras  
- **Frontend:** Streamlit  
- **NLP:** NLTK  
- **Explainability:** LIME  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  

---

## 📂 Project Structure
```
FakeNews_RNN_Project/
│── data/
│ ├── comprehensive_news_balanced.csv
│ ├── reviews.csv
│
│── model/
│ ├── best_model.h5
│ ├── review_model.h5
│ ├── tokenizer.pkl
│ ├── review_tokenizer.pkl
│
│── app.py
│── train_model.py
│── train_review_model.py
│── requirements.txt
│── README.md
```

---

## ⚙️ How It Works

### 🔹 Fake News Detection
- Input: News text or article URL  
- Preprocessing: Cleaning (remove stopwords, symbols)  
- Model: **Bidirectional LSTM (RNN)**  
- Output: Real / Fake + Confidence Score  

### 🔹 Fake Review Detection
- Input: User review text  
- Model: **LSTM Network**  
- Output: Real / Fake + Confidence Score  

---

## 📊 Model Performance

| Model            | Accuracy | F1 Score |
|-----------------|---------|----------|
| Fake News Model | ~94%    | High     |
| Review Model    | ~94%    | High     |

---

## 🔍 Explainable AI (LIME)

The system highlights **important words** that influenced prediction.

Example:
```
great : +0.45
fake : -0.30
amazing : +0.60
```

👉 Positive score → pushes towards REAL  
👉 Negative score → pushes towards FAKE  

---

## 🖥️ How to Run

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/FakeNews-RNN-Project.git
cd FakeNews-RNN-Project
```
### Step 2: Install Dependencies
```
pip install -r requirements.txt
```
### Step 3: Run Application
```
streamlit run app.py
```
## 🌐 Usage
```
1. Select Fake News or Fake Review Detection
2. Enter text OR paste news article URL
3. Click Detect
4. View:
     .Prediction (Real/Fake)
     .Confidence Score
     .Important words (LIME)
```
## 📸 Screenshots
