# Kidney CT Image Classification

A **Streamlit** web application that classifies kidney CT scan images into four categories: **Cyst**, **Normal**, **Stone**, and **Tumor** using a deep learning CNN model.

---
## 📊 Dataset
- **Source**: https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone
- **Classes**: Cyst (3,709 images), Normal (5,077 images), Stone (1,377 images), Tumor (2,283 images)
- **Total**: 12,446 CT scan images

## 🧠 Features

- **Image Classification**: Upload kidney CT scans and get instant predictions
- **4-Class Detection**: Identifies Cyst, Normal, Stone, and Tumor conditions
- **Confidence Scores**: Shows prediction confidence and probability breakdown
- **Interactive UI**: User-friendly Streamlit interface with image preview
- **Real-time Processing**: Fast inference using pre-trained CNN model

---

## 📦 Tech Stack

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning model training and inference
- **Streamlit**: Web application interface
- **PIL/Pillow**: Image processing and manipulation
- **NumPy**: Numerical computations
- **OpenCV**: Image preprocessing (implicit)

---

## 🏗️ Model Architecture

The project uses a **Convolutional Neural Network (CNN)** with the following architecture:
- **Input**: 150x150x3 RGB images
- **Layers**: 3 Conv2D layers with MaxPooling2D, Flatten, Dense layers
- **Output**: 4-class classification (Cyst, Normal, Stone, Tumor)
- **Optimizer**: Adam
- **Loss**: Categorical crossentropy

---

## 🚀 How to Run the Project

Follow these steps to set up and run the application:

```bash
# Step 1: Clone the repository
git clone https://github.com/Raghava-Ram/Kidney-disease-detection.git
cd Kidney-disease-detection

# Step 2: (Optional but recommended) Create and activate a virtual environment

# For Windows
python -m venv venv
venv\Scripts\activate

# For Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Step 3: Install required dependencies
pip install streamlit tensorflow pillow numpy

# Step 4: Run the Streamlit app
streamlit run app.py
```

## 📁 Project Structure

```
Kidney-disease-detection/
├── app.py                          # Streamlit web application
├── kidney_classifier_model.h5      # Pre-trained CNN model
├── dataSplit.ipynb                 # Dataset splitting notebook
├── modelTraining.ipynb             # Model training notebook
├── dataset/                        # Split dataset (train/test)
│   ├── train/
│   │   ├── Cyst/
│   │   ├── Normal/
│   │   ├── Stone/
│   │   └── Tumor/
│   └── test/
│       ├── Cyst/
│       ├── Normal/
│       ├── Stone/
│       └── Tumor/
└── README.md
```

## 🎯 Usage

1. **Launch the app**: Run `streamlit run app.py`
2. **Upload image**: Use the sidebar to upload a kidney CT scan (JPG, JPEG, PNG)
3. **View results**: See the predicted class, confidence score, and probability breakdown
4. **Analyze**: Review all class probabilities for comprehensive analysis

## ⚠️ Important Notes

- The model expects **150x150 pixel** images and will automatically resize uploaded images
- Supported formats: JPG, JPEG, PNG
- This is a **research/educational tool** and should not be used for actual medical diagnosis
- Always consult healthcare professionals for medical decisions
