# Multimodal Prediction of Rapport Level in Dyadic Child–Child Interaction

**Author:** Aditya Harichandar Aravindan  
**Supervisor:** Marc Fraile Fàbrega  
**Reviewer:** Dr. Ginevra Castellano  
**Department of Information Technology, Uppsala University**  
**MSc in Image Analysis and Machine Learning**  
**Date:** June 2024  

---

## Overview

This repository contains the implementation, analysis pipelines, and experimentation scripts accompanying the Master's thesis titled:

> **“Multimodal Prediction of Rapport Level in Dyadic Child–Child Interaction”**

The project investigates computational methods for predicting **rapport levels** between pairs of children engaged in a collaborative storytelling task. The work combines **audio** and **video modalities** to explore uni-modal and multi-modal learning approaches using both **classical machine learning (ML)** and **deep learning (DL)** architectures.

The study aims to bridge affective computing and social signal processing to better understand **child–child interactions** and develop computational models that may assist educators, psychologists, and robotic systems in understanding **social and emotional dynamics**.

---

## Methodology

### 1. Audio-based Pipeline
- **Pre-processing:** trimming of non-conversational segments, noise reduction via spectral subtraction using the partner’s signal.  
- **Feature extraction:** *eGeMAPS* feature set from OpenSMILE (MFCCs, semitone frequency/envelope, loudness).  
- **Models:** KNN, SVM, Logistic Regression, Random Forests, and MLP.

### 2. Video-based Pipeline
- **Feature extraction:**  
  - Action Units (AUs) – presence & intensity  
  - Eye gaze and head pose vectors  
  - Facial landmarks  
- **Models:**  
  - Classical ML models using feature means per sample  
  - LSTM networks for temporal analysis  
  - End-to-end CNN, 2D-CNN + LSTM encoder-decoder, and ResNet-152 encoder architectures.

### 3. Multimodal Fusion Strategies
- **Early Fusion:** concatenation of synchronized audio–video features at input-level before training.  
- **Late Fusion:** combination of logits or predictions from unimodal networks via ML classifiers.

### 4. Evaluation Metrics
- **Accuracy**, **F1-score**, and **Matthews Correlation Coefficient (MCC)**.  
- MCC chosen as the primary metric for robust evaluation under class imbalance.

---
