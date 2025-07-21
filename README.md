
# 🧠 Eye Disease Classification with Integrated Chatbot

This repository contains our final year research project: **Eye Disease Classification with Integrated Chatbot**, which combines **deep learning-based image classification** using a pretrained CNN model and an **intelligent chatbot assistant** to provide real-time diagnostic support for patients and healthcare providers.

---

## 📌 Project Overview

Our system performs **automated classification of eye diseases** using retinal/OCT images and provides interactive **chat-based assistance** for users to understand their condition, symptoms, and recommended actions. It aims to support **early detection** and improve **eye care accessibility** through AI.

---

## 🧪 Key Features

- 🔍 **Pretrained CNN Model** (e.g., VGG16, ResNet50) for accurate eye disease classification
- 🤖 **Chatbot Assistant** to interact with users and explain the diagnosis
- 📊 Real-time predictions with disease probability
- 🗂️ Support for multiple eye conditions (e.g., Cataract, Glaucoma, Diabetic Retinopathy, etc.)
- 💬 Symptom-based reasoning integrated into chatbot dialogue

---

## 🧠 Model Architecture

- **Image Classifier**: Pretrained CNN (e.g., ResNet50) + custom dense layers
- **Chatbot**: Rule-based or LLM-driven conversational bot with knowledge base integration
- **Integration**: The chatbot presents results from the CNN model and provides guidance based on predicted disease

---

## 🛠️ Tech Stack

| Component       | Technology                |
|-----------------|---------------------------|
| Model           | TensorFlow / Keras (CNN)  |
| Chatbot         | Python, NLTK / Transformers |
| Backend         | Flask / FastAPI           |
| Frontend        | HTML, CSS, JavaScript     |
| Dataset         | Public Eye Disease Datasets (OCT, Fundus) |

---

### To run the RAG application
```
streamlit run .\rag_app.py
```
