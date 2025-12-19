Project 2: Industrial Pump Predictive Maintenance
Course: 62FIT4ATI - Fall 2025
Topic: Build a Recurrent Neural Network (RNN) to predict industrial pump failures based on time-series sensor data.
Group: 24

1. Project Overview
This project focuses on Predictive Maintenance, utilizing Deep Learning to analyze sensor data from an industrial pump. The goal is to detect potential failures before they happen by classifying the machine's status into three categories based on 52 sensors' readings:
* 0: BROKEN (Critical failure) 
* 1: NORMAL (Operating normally) 
* 2: RECOVERING (Recovering from a failure) 
The core challenge addressed in this project is the extreme class imbalance, as BROKEN events are very rare compared to NORMAL operations.

2. Dataset
* Source: The dataset consists of 220,320 time-series samples with 52 continuous sensor measurements.
* Data Link: https://drive.google.com/drive/folders/1nUq198QcmosKNqOQOpheutsdcQTa67VF
* Preprocessing:
* Handling missing values using Forward Fill (ffill) .
* Feature Scaling using MinMaxScaler (0-1 range) .
* Label Encoding for machine status .
* Sliding Window: Data is processed into time-series sequences with a window size (Time Steps) of 60 .

3. Tech Stack
* Language: Python 3.13.4
* Deep Learning Framework: TensorFlow / Keras
* Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn .

4. Model Architecture
We implemented a Long Short-Term Memory (LSTM) network, which is effective for time-series forecasting. Based on the notebook implementation :
* Input Layer: Shape (60, 52)
* LSTM Layer: 64 units, returns sequences
* Dropout Layer: 0.2 rate
* LSTM Layer: 32 units
* Dropout Layer: 0.2 rate
* Dense Output Layer: 3 units with Softmax activation

5. Optimization Techniques
To meet the project requirements of researching and applying optimization techniques, we implemented the following strategies to handle the Class Imbalance and improve performance:
* Computed Class Weights (Boosted):
We calculated class weights to handle imbalance. Specifically, the weight for the BROKEN class (Class 0) was boosted by a factor of 5.0 to force the model to pay more attention to rare failure events .
* Focal Loss Function:
Instead of standard Categorical Crossentropy, we implemented a custom Focal Loss (gamma=2.0, alpha=0.25). This loss function down-weights easy examples and focuses training on hard negatives, improving detection of rare classes .
* Dynamic Thresholding (Post-processing):
Standard argmax often misses rare classes. We implemented an automatic probability analysis to set a Dynamic Threshold (approximately 0.0037). Any sample with a BROKEN probability exceeding this threshold is flagged as critical .
* Early Stopping:
We used Early Stopping monitoring val_loss to prevent overfitting during training .

6. Project Structure
The Github project is organized as required:
* 62FIT4ATI_Group_Topic2.ipynb: Main Jupyter notebook with well-documented code.
* pump_lstm_model_final.h5: Trained model file.
* README.md: Setup and reproduction instructions.
* Report.pdf: Written report analyzing results.

7. How to Run
Prerequisites
Ensure you have the following libraries installed:
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn

Execution Steps
* Download the Dataset: Download sensor.csv from the provided link.
* Mount Drive (if using Colab): The notebook assumes the dataset is located at /content/drive/MyDrive/Dataset/sensor.csv. Adjust the file_path variable in the "2. Lấy Datasets" section if your path differs .
* Run the Notebook: Open the .ipynb file and run all cells sequentially. The notebook includes data loading, preprocessing, model training with Focal Loss, and evaluation .

8. Results
* Overall Accuracy: Approximately 89% .
* Key Achievement: The model successfully detected the single BROKEN instance in the test set using the Dynamic Thresholding technique .

9. Contributors
Group: 24
Members:
Nguyễn Tuấn Anh - 2201140006
Nguyễn Quang Thiện - 2201140090
Đặng Văn Minh - 2201140052
