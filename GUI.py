import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pywt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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
    for coef in coeffs:
        features.append(np.mean(coef))
        features.append(np.std(coef))
    return np.array(features)


# Dummy training setup (replace with actual training process)
def train_models():
    normal_signals = [np.sin(np.linspace(0, 2 * np.pi, 300)) for _ in range(50)]
    lbbb_signals = [np.cos(np.linspace(0, 2 * np.pi, 300)) for _ in range(50)]
    normal_features = [extract_wavelet_features(preprocess_signal(sig)) for sig in normal_signals]
    lbbb_features = [extract_wavelet_features(preprocess_signal(sig)) for sig in lbbb_signals]
    X = np.vstack((normal_features, lbbb_features))
    y = np.array([0] * len(normal_features) + [1] * len(lbbb_features))

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    decision_tree = DecisionTreeClassifier(max_depth=4)
    decision_tree.fit(X, y)

    svm = SVC(kernel='linear', probability=True,C=3)
    svm.fit(X, y)

    return {"KNN": knn, "Decision Tree": decision_tree, "SVM": svm}


# Load trained models
models = train_models()


# GUI Functions
def predict_signal():
    try:
        signal_input = signal_entry.get()
        signal_values = np.array([float(x) for x in signal_input.split('|') if x])

        # Preprocess and extract features
        preprocessed_signal = preprocess_signal(signal_values)
        features = extract_wavelet_features(preprocessed_signal)

        # Get the selected model
        selected_model = classification_option.get()
        model = models[selected_model]

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
predict_button.pack()

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
