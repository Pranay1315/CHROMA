# **CHROMA: Image Colorizer 🎨**  
_A Deep Learning-based Image Colorization Project_  

## **📌 Overview**  
CHROMA is an advanced image colorization tool built using deep learning techniques. Leveraging Generative Adversarial Networks (GANs) like **Pix2Pix** and **Autoencoders**, this project can restore colors to black-and-white images with high accuracy and realism.  

## **🚀 Features**  
- **GAN-based Colorization** – Uses **Pix2Pix** for conditional image translation.  
- **Autoencoder-based Learning** – Captures image structure and patterns for efficient colorization.  
- **Pretrained Models Support** – Can fine-tune on different datasets.  
<!-- **CLI & GUI Support** – Run colorization via command line or graphical interface.-->

## **🛠️ Installation**  
### **Clone the Repository**  
```sh
git clone https://github.com/Pranay1315/CHROMA.git
cd CHROMA
```
### **Install Dependencies**  
Make sure you have Python installed, then run:  
```sh
pip install -r requirements.txt
```

## **🖼️ Usage**  
### **1. Colorizing a Single Image**  
```sh
python colorize.py --input images/input.jpg --output results/output.jpg
```
<!--### **2. Colorizing a Batch of Images**  
```sh
python colorize.py --input_folder images/ --output_folder results/
```
### **3. Using the GUI (Optional)**  
```sh
python gui.py
```-->

## **📊 Model Training **  
To train the model on a custom dataset:  
```sh
python train.py --dataset path/to/dataset --epochs 50
```

## **📁 Dataset**  
- Default training dataset:  https://www.kaggle.com/datasets/aayush9753/image-colorization-dataset  (OR JUST REFFER TO THE ABOVE "DATASET" FOLDER FOR THE SAME LINK)
- Custom datasets can be used by placing grayscale and colored images in `Dataset/` folder.

<!--## **🔗 References**  
- **Pix2Pix Paper**: [Isola et al., 2017](https://arxiv.org/abs/1611.07004)  
- **Autoencoders**: [Goodfellow et al., Deep Learning Book](https://www.deeplearningbook.org/) --> 

## **📜 License**  
This project is licensed under the MIT License. See `LICENSE` for details.  

---
