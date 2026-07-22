
# 🧠 LostGANs – Modern Reimplementation of ICCV 2019: "Image Synthesis From Reconfigurable Layout and Style"

This repository presents a modernized implementation of the ICCV 2019 paper **"Image Synthesis From Reconfigurable Layout and Style"**, referred to as **LostGANs**. This version is redesigned to work seamlessly with current PyTorch and CUDA versions, addressing real-world hardware limitations while retaining the essence and technical innovation of the original work.

---

## 👥 Team Members

- **Parth Agarwal** – 2310110546  
- **Krishna Garg** – 2310110692  
- **Ayush Tiwari** – 2310110661  
- **Bind Pratap Singh** – 2310110084

---

## 📌 Project Overview

LostGANs focuses on **image synthesis** conditioned on **semantic layout** and **style encoding** using a **generative adversarial network (GAN)** framework. The network aims to generate realistic images by controlling both spatial configuration (layout) and appearance (style).

This project adapts and enhances the official GitHub implementation of the paper. Due to outdated dependencies and APIs in the original repository, we rebuilt the pipeline to work with the **latest versions of PyTorch and CUDA**.

---

## ⚙️ Technical Highlights

### 🧠 GAN Architecture

LostGANs is a multi-stage conditional GAN that includes:

- **Layout Encoder**: Encodes spatial relationships and object semantics.
- **Style Encoder**: Extracts style vectors from reference images or samples from noise.
- **Generator**: Synthesizes images from layout and style vectors.
- **Discriminator**: Judges image realism and consistency with layout/style.

Key architectural enhancements and strategies:
- **Progressive training** across multiple stages.
- **Object-wise conditioning** using bounding boxes and semantic labels.
- **Style reconfiguration** using adaptive instance normalization (AdaIN).

---

## 🔄 Enhancements in Our Implementation

### ✅ Modern Compatibility

- Rewritten for **Python 3.12+**
- Fully compatible with **PyTorch ≥ 2.0** and **CUDA 12.8**
- Updated preprocessing, dataloaders, and utility functions

### 🚀 Efficient Memory Utilization

Due to limited hardware resources, we implemented:
- Dynamic RAM and **virtual memory swapping**
- **Timely GPU cache clearing** to prevent memory overflow
- Reduced image resolution for training while preserving core quality

### 💾 Checkpointing

- Checkpoints saved **every 3 epochs**
- Ability to **resume training from saved checkpoints**
- Automatic optimizer and scheduler state restoration

### 📂 Storage-Aware Development

Due to storage limitations:
- Datasets and `.pth` checkpoint files are **not included**
- Users must manually set dataset paths and download the required files
- Frequent checkpointing consumed large disk space, hence omitted from upload

---


## 🚀 How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Start Training

```bash
python train.py
```



## 🙏 Acknowledgements

We acknowledge the authors of the original LostGANs paper and GitHub repository. Our work builds upon their innovation and adapts it for a modern, accessible development environment.
