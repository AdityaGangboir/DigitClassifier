## 🧠 MNIST Digit Classifier 🖋️

This project builds a deep learning model to recognize handwritten digits (0–9) from the **MNIST dataset** using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**.

---

### 📌 Problem Statement

The MNIST dataset contains 70,000 grayscale images of handwritten digits (28×28 pixels).
The task is to build a machine learning model that **accurately classifies digits from 0 to 9**.

---

### 🚀 Solution Overview

The approach includes:

1. **Data Preprocessing**

   * Loading MNIST using `tensorflow.keras.datasets`
   * Normalizing pixel values to \[0, 1]
   * One-hot encoding target labels

2. **Model Architecture**
   A CNN was chosen due to its effectiveness in image classification tasks:

   * `Conv2D → ReLU → MaxPooling`
   * `Conv2D → ReLU → MaxPooling`
   * Flatten → Dense → Dropout → Output Dense Layer (Softmax)

   Layers Summary:

   ```text
   Input: (28, 28, 1)
   Conv2D(32 filters, 3x3) + ReLU
   MaxPooling2D(2x2)
   Conv2D(64 filters, 3x3) + ReLU
   MaxPooling2D(2x2)
   Flatten
   Dense(128) + ReLU
   Dropout(0.5)
   Dense(10, activation='softmax')
   ```

3. **Training**

   * Optimizer: `Adam`
   * Loss: `categorical_crossentropy`
   * Metrics: `accuracy`
   * Trained for **10 epochs** with batch size 128

4. **Evaluation**

   * Model achieved **over 98% test accuracy**
   * Visualized prediction results for random test images
   * Confusion matrix and classification report used for deeper insight

---

### 📈 Achieved Results

* **Test Accuracy**: **\~98.7%**
* Fast convergence and good generalization
* Correctly classifies most test images with minimal overfitting

---

Main packages used:

* `tensorflow`
* `numpy`
* `matplotlib`
* `scikit-learn`

---

### 📊 Example Predictions

The notebook visualizes a few predictions using `matplotlib`, showing the model's output on sample test images.

---

### 📁 Files

* `MnistDigitClassifier.ipynb` – Main notebook
* `README.md` – You're reading it :)
