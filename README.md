# **GAN Image Generator using PyTorch**

## **Overview**  
This project develops a deep learning-based image generation system using Generative Adversarial Networks (GANs).  

The system generates handwritten digit images similar to the MNIST dataset by learning data distribution through adversarial training. It includes model training using PyTorch and deployment through an interactive Streamlit web application.

---

## **Problem Statement**  
Generating realistic images from random noise is a challenging problem in deep learning.  

The objective of this project is to build a GAN model capable of generating handwritten digits by learning patterns from real data and producing visually similar synthetic images.

---

## **Dataset**  
- Dataset: MNIST Handwritten Digits  
- Source: torchvision.datasets  

### **Target Variable**  
- Image generation (unsupervised learning)  

---

## **Tools and Technologies**  
- **Python**  
- **PyTorch** for deep learning  
- **Torchvision** for dataset handling  
- **NumPy** for numerical operations  
- **Matplotlib** for visualization  
- **PIL (Pillow)** for image processing  
- **Streamlit** for deployment  

---

## **Exploratory Data Analysis**  
- Understanding MNIST dataset structure  
- Visualization of handwritten digit samples  
- Normalization of image pixel values  
- Analysis of grayscale image distribution  

---

## **Data Preprocessing**  

### **Image Processing**  
- Conversion to tensor format  
- Normalization to range [-1, 1]  
- Flattening images (28x28 → 784)  

### **Batching**  
- DataLoader used for efficient training  
- Shuffling enabled for randomness  

---

## **Deep Learning Model (GAN)**  

### **Model Components**  
- Generator (G)  
- Discriminator (D)  

---

### **Generator Architecture**  
- Input: Random noise vector (latent space)  
- Fully connected layers  
- ReLU activation  
- Output layer with Tanh activation  

---

### **Discriminator Architecture**  
- Input: Image (real or generated)  
- Fully connected layers  
- LeakyReLU activation  
- Output layer with Sigmoid (real/fake classification)  

---

## **Training Process**  

### **Generator Objective**  
- Fool the discriminator into classifying fake images as real  

### **Discriminator Objective**  
- Distinguish between real and fake images  

### **Loss Function**  
- Binary Cross Entropy Loss (BCELoss)  

### **Optimization**  
- Adam optimizer for both networks  

---

## **Model Evaluation**  

### **Metrics Used**  
- Generator Loss  
- Discriminator Loss  

### **Result**  
The model successfully learns to generate realistic handwritten digits after sufficient training epochs, demonstrating the adversarial learning process.

---

## **Final Model**  
- Generator Model: `generator.pth`  

### **Additional Details**  
- Latent Dimension: 100  
- Epochs: 105  
- Batch Size: 24  

---

## **Prediction Pipeline**  
- Generate random noise vector  
- Pass through trained Generator  
- Convert output to image format  
- Display generated digit  

---

## **Streamlit Application**  

File: `App.py`  

An interactive web application for generating handwritten digits in real-time.

### **Features**  
- Generate multiple images at once  
- Control noise variation  
- Adjustable number of outputs  
- Download generated images  
- Clean and styled UI  

### **Run the App**  
```bash
streamlit run App.py
pip install torch torchvision streamlit matplotlib numpy pillow
