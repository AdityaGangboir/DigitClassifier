## 🧠 MNIST Digit Classifier 🖋️

This project implements a simple feedforward neural network (Multi-Layer Perceptron) to recognize handwritten digits (0–9) from the **MNIST dataset** using **TensorFlow/Keras**.

---

### 📌 Problem Statement

The MNIST dataset consists of **70,000 grayscale images** (28×28 pixels) of handwritten digits ranging from 0 to 9.
The goal is to build a neural network that can accurately classify these digits.

---

### 🚀 Solution Overview

#### 1. **Data Loading & Preprocessing**

* Load the MNIST dataset using `tensorflow.keras.datasets.mnist`.
* Normalize pixel values to the range **\[0, 1]** (divide by 255).
* Split into training and test sets.

#### 2. **Model Architecture** – Multi-Layer Perceptron (MLP)

A simple sequential model with fully connected layers:

```python
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))            # Flatten 2D image to 1D vector
model.add(Dense(128, activation='relu'))            # Hidden layer with 128 neurons
model.add(Dense(32, activation='relu'))             # Hidden layer with 32 neurons
model.add(Dense(10, activation='softmax'))          # Output layer for 10 classes
```

* **Flatten**: Transforms 28×28 pixels into a 784-length vector.
* **Dense(128, relu)**: Learns intermediate features.
* **Dense(32, relu)**: Further abstraction.
* **Dense(10, softmax)**: Outputs class probabilities for digits 0–9.

---

#### 3. **Compilation & Training**

* **Compiler settings**:

  * Optimizer: `Adam`
  * Loss: `sparse_categorical_crossentropy`
  * Metrics: `accuracy`

* **Training**:

  * Epochs: **100**
  * Validation split: **20%** of training data

Example:

```python
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)
history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2
)
```

---

### ✅ Results

* **Test Accuracy**: \~97.47%
  (computed using `accuracy_score` on `y_test` vs. `y_pred`)

* **Training Dynamics**:

  * Training accuracy quickly approaches 99.9%.
  * Validation accuracy stabilizes around 97.5%.

---

### 📊 Loss Curves

Training and validation loss curves can be visualized with `matplotlib`:

```python
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.show()
```

---

### 📁 Project Structure

```
├── README.md                      # This file
└── MnistDigitClassifier.ipynb     # Jupyter Notebook
```

---

### 📦 Dependencies

Install required packages:

```bash
pip install tensorflow matplotlib numpy scikit-learn
```

---

### 💡 Future Improvements

* Experiment with deeper or wider MLP architectures.
* Incorporate Dropout or Batch Normalization to reduce overfitting.
* Compare performance with Convolutional Neural Networks (CNNs).
