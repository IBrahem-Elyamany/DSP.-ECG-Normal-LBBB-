import math
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox

import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showerror

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scipy.signal import butter, filtfilt
import pywt
#import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



def load_data(file_path):
    try:
        array_of_arrays = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    values = [float(value) for value in line.split('|') if value]
                    array_of_arrays.append(values)
        return array_of_arrays
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []




def preprocess_signal(signal, fs=360, lowcut=0.5, highcut=40):
    # Mean removal
    signal = signal - np.mean(signal)

    # Bandpass Butterworth filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)


    # Normalization
    # Normalize to range 0 to 1
    normalized_signal = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal))
    return normalized_signal



def extract_wavelet_features(signal, wavelet='db4', level=9):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []
    for coeff in coeffs[1:8]:
        features.extend([np.mean(coeff), np.std(coeff)])
    return np.array(features)


def main():
    # Load datasets
    lbbb_train = load_data('Normal&LBBB/LBBB_Train.txt')
    lbbb_test = load_data('Normal&LBBB/LBBB_Test.txt')
    normal_train = load_data('Normal&LBBB/Normal_Train.txt')
    normal_test = load_data('Normal&LBBB/Normal_Test.txt')

    # Preprocess datasets
    preprocessed_normal_train = [preprocess_signal(signal, 360) for signal in normal_train]
    preprocessed_normal_test = [preprocess_signal(signal, 360) for signal in normal_test]
    preprocessed_lbbb_train = [preprocess_signal(signal, 360) for signal in lbbb_train]
    preprocessed_lbbb_test = [preprocess_signal(signal, 360) for signal in lbbb_test]

    # Extract features
    features_train = [extract_wavelet_features(signal) for signal in preprocessed_normal_train + preprocessed_lbbb_train]
    features_test = [extract_wavelet_features(signal) for signal in preprocessed_normal_test + preprocessed_lbbb_test]

    # Combine labels
    labels_train = [0] * len(preprocessed_normal_train) + [1] * len(preprocessed_lbbb_train)
    labels_test = [0] * len(preprocessed_normal_test) + [1] * len(preprocessed_lbbb_test)

    # Train and evaluate the model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(features_train, labels_train)
    y_pred_knn = knn.predict(features_test)

    print("Accuracy knn:", accuracy_score(labels_test, y_pred_knn))
    print("Classification Report knn:\n", classification_report(labels_test, y_pred_knn))

    svm_model = SVC(kernel="linear",C=3)
    svm_model.fit(features_train, labels_train)
    y_pred_svm=svm_model.predict(features_test)
    print("Accuracy SVM:", accuracy_score(labels_test, y_pred_svm))
    print("Classification Report SVM:\n", classification_report(labels_test, y_pred_svm))

    dt_model = DecisionTreeClassifier(max_depth=4)
    dt_model.fit(features_train, labels_train)
    y_pred_DecisionTree = dt_model.predict(features_test)
    print("Accuracy DecisionTree:", accuracy_score(labels_test, y_pred_DecisionTree))
    print("Classification Report DecisionTree:\n", classification_report(labels_test, y_pred_DecisionTree))

    range_knn = range(1, 17)
    accuracy_knn = []
    for i in range_knn:
        model_test = KNeighborsClassifier(n_neighbors=i)
        model_test.fit(features_train, labels_train)
        y_pred_kNN = model_test.predict(features_test)
        val_accuracy_knn = accuracy_score_kNN = accuracy_score(labels_test, y_pred_kNN)
        accuracy_knn.append(val_accuracy_knn)

    # plt.plot(max_depth_range, val_results, 'g-', label='Val score')
    plt.plot(range_knn, accuracy_knn, 'r-', label='Train score')
    plt.ylabel('Accuracy score')
    plt.xlabel('K')
    plt.legend()
    plt.grid(True)
    plt.show()

    range_random_forest = range(1, 350, 10)
    accuracy_random_forest = []
    for i in range_random_forest:
        model_test2 = RandomForestClassifier(n_estimators=i)
        model_test2.fit(features_train, labels_train)
        y_pred_random_forest = model_test2.predict(features_test)
        val_accuracy_random_forest = accuracy_score_random_forest = accuracy_score(labels_test, y_pred_random_forest)
        accuracy_random_forest.append(val_accuracy_random_forest)

    plt.plot(range_random_forest, accuracy_random_forest, 'r-', label='Train score')
    plt.ylabel('Accuracy score')
    plt.xlabel('n_estimators')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
