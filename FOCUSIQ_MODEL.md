# FocusIQ Machine Learning Model
notebook: https://drive.google.com/file/d/1-fhg6pCtatUlo8im9Xaz6Gjy2zRf1blt/view?usp=sharing

dataset: https://drive.google.com/file/d/1JD3CRBqb4daqyLtT07NXZiofeMY0Y_cw/view?usp=sharing

## Overview

FocusIQ is the machine learning engine powering Calmin30.

It predicts a student's Focus Score and estimates the risk of procrastination using behavioural and lifestyle data.

---

## Problem Statement

Students often struggle with maintaining focus due to multiple interconnected factors such as sleep quality, stress, phone usage, study habits, exercise, and motivation.

FocusIQ analyzes these patterns to provide meaningful insights and encourage healthier productivity habits.

---

## Input Features

- Sleep Hours
- Study Hours
- Phone Usage
- Stress Level
- Motivation
- Exercise

---

## Machine Learning Algorithms

### Random Forest

Predicts Focus Score.

### Support Vector Machine

Predicts procrastination risk.

### K-Means Clustering

Group students with similar behavioural patterns to generate personalized insights.

---

## Prediction Pipeline

User Input

↓

Data Cleaning

↓

Feature Scaling

↓

Model Prediction

↓

Focus Score

↓

Risk Classification

↓

Recommendations

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## Future Work

- Deep Learning models
- Explainable AI (SHAP)
- Real-time model retraining
- Larger datasets
- Mobile integration
