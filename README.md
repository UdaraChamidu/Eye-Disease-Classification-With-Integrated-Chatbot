
# Eye Disease Classification with Chatbot Integration

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen) ![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue) ![License](https://img.shields.io/badge/License-MIT-yellowgreen)

## Overview

This project develops an AI-powered **chatbot** system for diagnosing eye diseases using **user-provided clinical data** (age, blood pressure, sugar level, gender) and **retinal scan images**. It also provides actionable recommendations and connects users to healthcare services.

The goal is to provide early detection and guidance for managing eye health, reducing healthcare burdens by providing diagnostic services remotely.

## Key Features
- **AI-based Disease Classification**: Uses **Convolutional Neural Networks (CNN)** to analyze retinal images for diseases like diabetic retinopathy, glaucoma, and cataracts.
- **Multi-Modal Integration**: Combines **clinical data** and **retinal scan images** to improve diagnostic accuracy.
- **Chatbot Interface**: A conversational AI chatbot for collecting data and delivering diagnosis and health advice.
- **Additional Services**: Provides **preventive care tips**, **doctor recommendations**, and **eye health updates**.

## Table of Contents

- [Features](#key-features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Technologies Used

- **Deep Learning**: TensorFlow, Keras for training CNN models.
- **Machine Learning**: Scikit-learn for processing clinical data (Random Forest, Logistic Regression).
- **NLP & Chatbot Frameworks**: Rasa/Dialogflow for the chatbot integration.
- **Backend**: Python (Flask/FastAPI) for serving models and APIs.
- **Frontend**: HTML, CSS, and JavaScript for UI.
- **Image Processing**: OpenCV, PIL for image preprocessing and augmentation.

## Installation

To get started with the project, clone this repository and set up the environment:

### 1. Clone the repository
```bash
git clone https://github.com/UdaraChamidu/eye-disease-classifier.git
cd eye-disease-classifier

