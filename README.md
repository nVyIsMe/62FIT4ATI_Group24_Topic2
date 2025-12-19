# Project 2: Industrial Pump Predictive Maintenance

**Course:** 62FIT4ATI – Fall 2025

**Group:** 24

**Topic:** Build a Recurrent Neural Network (RNN) to predict industrial pump failures based on time-series sensor data.

---

## 📋 Project Overview

This project utilizes **Deep Learning** to analyze sensor data from industrial pumps. The goal is to detect potential failures before they occur by classifying the machine's status based on readings from **52 different sensors**.

### Target Classes

The model classifies machine status into three distinct categories:

* **0: BROKEN** – Critical failure event.
* **1: NORMAL** – Operating within standard parameters.
* **2: RECOVERING** – The phase following a failure.

> [!IMPORTANT]
> **Challenge:** The dataset suffers from extreme **class imbalance**. BROKEN events are rare, making them difficult for standard models to detect without specialized optimization.

---

## 📊 Dataset Specifications

* **Total Samples:** 220,320 time-series entries.
* **Features:** 52 continuous sensor measurements.
* **Data Source:** [Google Drive Link](https://drive.google.com/drive/folders/1nUq198QcmosKNqOQOpheutsdcQTa67VF)

### Preprocessing Pipeline

1. **Missing Values:** Handled via **Forward Fill (ffill)** to maintain time-series continuity.
2. **Scaling:** Applied **MinMaxScaler** (0-1 range) to all sensor inputs.
3. **Encoding:** Label encoding applied to the machine status.
4. **Sliding Window:** Data segmented into sequences with a **Window Size (Time Steps) of 60**.

---

## 🛠 Tech Stack

| Category | Tools |
| --- | --- |
| **Language** | Python 3.13.4 |
| **Deep Learning** | TensorFlow / Keras |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |

---

## 🏗 Model Architecture

We implemented a **Long Short-Term Memory (LSTM)** network, specifically chosen for its ability to capture long-term dependencies in time-series data.

| Layer | Specification |
| --- | --- |
| **Input Layer** | Shape: (60, 52) |
| **LSTM Layer 1** | 64 Units (Returns Sequences) |
| **Dropout** | 0.2 rate |
| **LSTM Layer 2** | 32 Units |
| **Dropout** | 0.2 rate |
| **Output Layer** | Dense (3 units) with **Softmax** activation |

---

## 🚀 Optimization Techniques

To address class imbalance and maximize detection accuracy, the following strategies were implemented:

* **Boosted Class Weights:** The weight for the **BROKEN** class (0) was increased by a **factor of 5.0**, forcing the model to prioritize rare failure events.
* **Focal Loss Function:** Replaced standard cross-entropy with Focal Loss () to focus training on "hard" examples.
* **Dynamic Thresholding:** Instead of a standard 0.5 threshold, we used a calculated threshold () to flag critical failures earlier.
* **Early Stopping:** Monitored `val_loss` to prevent overfitting and ensure the model generalizes well.

---

## 📂 Project Structure

* `62FIT4ATI_Group24_Topic2.ipynb`: Documented Jupyter notebook.
* `pump_lstm_model_final.h5`: The final trained model file.
* `Prọect Report.pdf`: Comprehensive analysis and result documentation.
* `README.md`: This setup and reproduction guide.

---

## 💻 How to Run

### 1. Install Prerequisites

```
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn

```

### 2. Execution Steps

1. **Download Data:** Place `sensor.csv` in your working directory or Google Drive.
2. **Configuration:** If using Google Colab, mount your drive and update the `file_path` in the **"2. Lấy Datasets"** section.
3. **Run:** Execute all cells in the `.ipynb` file sequentially.

---

## 📈 Results

* **Overall Accuracy:** ~89%
* **Key Achievement:** Successfully detected the **single BROKEN instance** in the test set by leveraging the Dynamic Thresholding technique.

---

## 👥 Contributors (Group 24)

* **Nguyễn Tuấn Anh** - 2201140006
* **Nguyễn Quang Thiện** - 2201140090
* **Đặng Văn Minh** - 2201140052
