# Brain Tumor Detection 

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify MRI scans into two categories: **Tumor Detected (yes)** and **No Tumor Detected (no)**.

The system uses image augmentation and deep learning layers to achieve high accuracy in medical image classification.

## 🛠️ System Setup
To run this project, you need Python 3.8+ installed on your system.

### 1. Clone or Save the Script
Ensure your Python script (e.g., `main.py`) and the `README.md` are in the same project directory.

### 2. Install Dependencies
Run the following command in your terminal or command prompt to install the required libraries:
```bash
pip install tensorflow numpy matplotlib opencv-python scikit-learn
```

## 📂 Dataset Implementation
The code uses the `ImageDataGenerator` class, which requires a specific folder structure to identify classes automatically. You must organize your images as follows:

### Folder Structure
Create a main folder (e.g., `Brain_Dataset`) and split your images into two subfolders:
```
Brain_Dataset/
├── yes/                # Place all MRI images WITH tumors here
│   ├── image_01.jpg
│   ├── image_02.jpg
└── no/                 # Place all MRI images WITHOUT tumors here
    ├── image_03.jpg
    ├── image_04.jpg
```
**Note:** The folder names `yes` and `no` will be used as the class labels by the model.

### Updating the Path in Code
In the Python script, locate Line 12 and update the `DATASET_PATH` variable with the location of your folder.
- **Windows:** Use a raw string (`r'...'`) to handle backslashes.
```python
dataset_path = r'C:\Users\YourName\Desktop\Brain_Dataset'
```
- **macOS/Linux:** Use standard forward slashes.
```python
dataset_path = '/Users/YourName/Desktop/Brain_Dataset'
```

## 🚀 How to Operate the Code
- **Prepare Data:** Ensure your images are organized in the `yes` and `no` folders.
- **Run the Script:** Execute the script via terminal:
```bash
python main.py
```
- **Training Phase:** The model will begin training for 20 epochs. You will see loss and accuracy metrics update in the console.
- **Review Results:** *Once training is complete, a window will pop up showing Accuracy and Loss graphs.*
- The model will automatically save as `brain_tumor_detector.h5` in your project folder.
- **Deployment:** Use the saved `.h5` file later to make predictions on new images without retraining.

## 🧠 Model Architecture
the model is built using a sequential CNN architecture:
- **Convolutional Layers:** Extracts spatial features (edges, textures) from MRI scans.
- **MaxPooling:** Reduces dimensionality to focus on important features.
- **Dropout (0.5):** Prevents overfitting by randomly ignoring neurons during training.
- **Sigmoid Activation:** Ideal for binary classification (Tumor vs. No Tumor).

## 📈 Tips for Better Performance
dataset size: For best results, use at least 100–200 images per class. If you don't have a dataset, the "Brain MRI Images for Tumor Detection" dataset on Kaggle is highly recommended.
imagery quality: Ensure images are clear and cropped to focus on brain area.
alternative models: If higher accuracy is needed for complex datasets, consider Transfer Learning with pre-trained models like VGG16 or ResNet50.

## ⚖️ Disclaimer
the project is for educational purposes only and should not be used for professional medical diagnosis. Always consult healthcare professionals for medical concerns.

##  About Me
Name: Vaibhav Bharti

Registration No.: 25BCE10081

Course: Fundamentals of AIML
