# 📧 Spam Email Detection using Machine Learning

## 📌 Overview

This project implements a machine learning-based system to classify messages as **Spam** or **Not Spam**. It uses Natural Language Processing (NLP) techniques to process textual data and applies a classification algorithm to detect unwanted messages.

The model is trained on a labeled dataset and can accurately predict whether a given message is spam or not through a simple command-line interface.

---

## 🎯 Objectives

* To build a spam detection model using machine learning
* To preprocess and analyze textual data
* To apply feature extraction using TF-IDF
* To classify messages using Naive Bayes algorithm

---

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorizer
* Multinomial Naive Bayes

---

## 📂 Project Structure

```
spam-email-detector/
│
├── dataset/
│   └── spam.csv
│
├── src/
│   └── main.py
│
├── model/
│
├── README.md
├── requirements.txt
└── report.pdf
```

---

## ⚙️ How It Works

1. The dataset containing labeled messages is loaded
2. Data preprocessing is performed
3. Text is converted into numerical form using TF-IDF
4. A Naive Bayes model is trained on the data
5. The model predicts whether a message is spam or not

---

## ▶️ How to Run the Project

1. Clone the repository:

```
git clone https://github.com/your-username/spam-email-detector
```

2. Navigate to the project folder:

```
cd spam-email-detector/src
```

3. Install dependencies:

```
pip install -r ../requirements.txt
```

4. Run the program:

```
python main.py
```

---

## 📊 Output

* Displays model accuracy
* Takes user input message
* Predicts:

  * Spam ❌
  * Not Spam ✅

---

## 📈 Results

The model achieves approximately **95% accuracy** on the test dataset, making it effective for spam detection tasks.

---

## 🚀 Future Enhancements

* Add GUI-based interface
* Improve accuracy using advanced models
* Deploy as a web application
* Use deep learning techniques

---

## 📚 Dataset

SMS Spam Collection Dataset

---

## 👩‍💻 Author

Tejal Singh
25BCE10060
---

## ⭐ Conclusion

This project demonstrates how machine learning can be effectively used to solve real-world problems like spam detection. It provides a simple yet powerful implementation of text classification using NLP techniques.
