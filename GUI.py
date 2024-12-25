import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pywt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# Preprocessing functions
def preprocess_signal(signal, fs=360, lowcut=0.5, highcut=40):
    signal = signal - np.mean(signal)
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    normalized_signal = (filtered_signal - np.min(filtered_signal)) / (
            np.max(filtered_signal) - np.min(filtered_signal))
    return normalized_signal

def extract_wavelet_features(signal, wavelet='db4', level=9):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []
    for coeff in coeffs[1:8]:
        features.extend([np.mean(coeff), np.std(coeff)])
    return np.array(features)

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
        messagebox.showerror("Error", f"Error reading file {file_path}: {e}")
        return []

# Training models
def train_models(train_data):
    features, labels = train_data
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(features, labels)

    decision_tree = DecisionTreeClassifier(max_depth=4)
    decision_tree.fit(features, labels)

    svm = SVC(kernel='linear', probability=True, C=3)
    svm.fit(features, labels)

    return {"KNN": knn, "Decision Tree": decision_tree, "SVM": svm}

# Global models
models = {}

def load_and_train():
    try:
        train_file = filedialog.askopenfilename(title="Select Training Dataset")
        test_file = filedialog.askopenfilename(title="Select Testing Dataset")

        if not train_file or not test_file:
            messagebox.showwarning("Warning", "Please select both train and test files.")
            return

        # Load datasets
        normal_train = load_data(train_file)
        normal_test = load_data(test_file)

        # Preprocess datasets
        preprocessed_train = [preprocess_signal(signal) for signal in normal_train]
        preprocessed_test = [preprocess_signal(signal) for signal in normal_test]

        # Extract features
        features_train = [extract_wavelet_features(signal) for signal in preprocessed_train]
        features_test = [extract_wavelet_features(signal) for signal in preprocessed_test]

        # Combine labels
        labels_train = [0] * len(preprocessed_train[:len(preprocessed_train)//2]) + [1] * len(preprocessed_train[len(preprocessed_train)//2:])
        labels_test = [0] * len(preprocessed_test[:len(preprocessed_test)//2]) + [1] * len(preprocessed_test[len(preprocessed_test)//2:])

        # Train models
        global models
        models = train_models((features_train, labels_train))

        messagebox.showinfo("Success", "Models trained successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during training: {e}")

def predict_signal():
    try:
        signal_input = signal_entry.get()
        signal_values = np.array([float(x) for x in signal_input.split('|') if x])

        # Preprocess and extract features
        preprocessed_signal = preprocess_signal(signal_values)
        features = extract_wavelet_features(preprocessed_signal)

        # Get the selected model
        selected_model = classification_option.get()
        model = models.get(selected_model)

        if not model:
            messagebox.showerror("Error", "Models are not trained. Load data and train models first.")
            return

        # Predict using the selected model
        prediction = model.predict([features])[0]
        result = "Normal" if prediction == 0 else "LBBB"
        messagebox.showinfo("Prediction", f"The signal is classified as: {result}")

        # Draw signal
        draw_signal(signal_values)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def draw_signal(signal):
    # Clear existing plots
    for widget in canvas_frame.winfo_children():
        widget.destroy()

    # Create a matplotlib figure
    fig = Figure(figsize=(5, 3), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(signal, color="blue")
    ax.set_title("ECG Signal", fontsize=12)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)

    # Add the plot to the GUI
    canvas = FigureCanvasTkAgg(fig, canvas_frame)
    canvas.get_tk_widget().pack()

# GUI setup
root = tk.Tk()
root.title("ECG Signal Classification")
root.geometry("800x600")
root.configure(bg="white")

# Header
header_frame = tk.Frame(root, bg="#ffffff")
header_frame.pack(fill="x", pady=10)
header_label = tk.Label(
    header_frame, text="ECG Signal Classifier", font=("Helvetica", 18, "bold"), bg="#ffffff", fg="#0078D7"
)
header_label.pack()

# Input Section
input_frame = tk.Frame(root, bg="white")
input_frame.pack(pady=10)
input_label = tk.Label(
    input_frame, text="Enter ECG Signal Line (values separated by '|'):", font=("Helvetica", 12), bg="white", fg="black"
)
input_label.pack(anchor="w")
signal_entry = tk.Entry(input_frame, width=90, font=("Helvetica", 10))
signal_entry.pack(pady=10)

# Classification Type Selector
classification_frame = tk.Frame(root, bg="white")
classification_frame.pack(pady=10)
classification_label = tk.Label(
    classification_frame, text="Select Classification Type:", font=("Helvetica", 12), bg="white", fg="black"
)
classification_label.pack(anchor="w")
classification_option = ttk.Combobox(
    classification_frame, values=["KNN", "Decision Tree", "SVM"], font=("Helvetica", 10), state="readonly", width=30
)
classification_option.set("KNN")
classification_option.pack(pady=5)

# Button Section
button_frame = tk.Frame(root, bg="white")
button_frame.pack(pady=20)
predict_button = tk.Button(
    button_frame, text="Predict", font=("Helvetica", 12, "bold"), bg="#0078D7", fg="white", command=predict_signal
)
predict_button.pack(side="left", padx=10)

train_button = tk.Button(
    button_frame, text="Load Data & Train", font=("Helvetica", 12, "bold"), bg="#28A745", fg="white", command=load_and_train
)
train_button.pack(side="left", padx=10)

# Canvas for Signal Plot
canvas_frame = tk.Frame(root, bg="white")
canvas_frame.pack(pady=20, fill="both", expand=True)

# Footer
footer_frame = tk.Frame(root, bg="white")
footer_frame.pack(fill="x", pady=10)
footer_label = tk.Label(
    footer_frame, text="Created by ECG Analysis Tool", font=("Helvetica", 10, "italic"), bg="white", fg="gray"
)
footer_label.pack()

root.mainloop()