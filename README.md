# Eye Disease Classification with Integrated Chatbot


---

## 📌 Overview
This research project focuses on developing a **multimodal AI system** for predicting eye diseases by integrating **OCT image analysis** and **text-based symptom descriptions**. The system uses a **Vision Transformer/CNN** for image data and a **Retrieval-Augmented Generation (RAG)** pipeline for text data, combined into a chatbot interface for user-friendly interactions.

---

## 🎯 Aim
To develop a **multimodal prediction system** that integrates medical images and patient-reported data for **accurate and explainable** eye disease classification.

---
## Screenshots
<img width="1077" height="600" alt="Screenshot 2025-08-08 172518" src="https://github.com/user-attachments/assets/68ef1154-f85e-48c2-9be1-7f8e40ea4dbb" />
---
## 🥅 Objectives
- Enhance disease prediction accuracy using **multiple models**.
- Develop a **Generative AI-based chatbot** for seamless user interaction.
- Compare **unimodal vs. multimodal** model performance.
- Provide **explainable predictions** for better clinical trust.

---

## 🔍 Research Gap
Most existing systems rely on **single-modal data** (images **or** text).  
Exploring the **predictive power of multimodal fusion** with Generative AI in ophthalmology is **still under-researched**.

---

## 🧠 Methodology
1. **Data Preprocessing**
   - Images: Resized & normalized for CNN/InceptionV3.
   - Text: Cleaned, tokenized, and embedded using **HuggingFace models**.
2. **Model Building**
   - **Image Model:** CNN/InceptionV3 for OCT images.
   - **Text Model:** RAG pipeline for symptom data.
   - **Fusion Model:** Combined image features + text embeddings.
3. **Evaluation**
   - CNN & Fusion: Accuracy, Precision, Recall, AUROC.
   - Text/RAG: Precision, Recall, F1 Score.
   - Compare **fusion vs. individual** models.

---

## 🏗 Progress So Far
- ✅ Image model (CNN) trained — **94% accuracy**.
- ✅ RAG pipeline developed for **explainable text predictions**.
- ✅ Basic chatbot interface created.
- 🔄 Fusion model trained — **76.6% accuracy** (limited by dataset size).
- 🛠 Data preprocessing & augmentation implemented.

---

## 📊 Findings So Far
- Multimodal approach improves prediction accuracy compared to single-modal.
- Image model achieved **94% accuracy**.
- RAG pipeline offers **explainable, evidence-backed** predictions.
- Image-only models lack **contextual depth** without text.
- Data quality significantly impacts performance.
- Fusion models capture **complementary features** from both modalities.
- Preprocessing improved model stability.
- Performance varies across datasets from different sources.

---

## 🚀 Future Plans
- Combine **all input modes** into one chatbot interface.
- Enhance UI for better user experience.
- Add **chat history** for past queries and results.
- Deploy system for public use.
- Expand dataset for better multimodal accuracy.

---

## 🗂 Dataset Sources
- Kaggle OCT image datasets  
- Ocular Disease Recognition (ODIR-5K) dataset  
- *Kanski's Clinical Ophthalmology* textbook for symptom descriptions

---

## 🛠 Technologies Used
- **Python**, **TensorFlow/Keras**, **PyTorch**
- **HuggingFace Transformers**
- **RAG (Retrieval-Augmented Generation)**
- **CNN / InceptionV3**
- **Vision Transformer (ViT)**
- **FastAPI**
- **React (Frontend)**
- **Streamlit** (Prototype UI)

---

## 📜 References
(Include your reference list here from the presentation slides.)

```
assets/
├── project_image.png ← Main project diagram/screenshot
├── ui_example.png ← Example of chatbot interface
├── model_results.png ← Accuracy/Confusion matrix
```


**👩‍💻 Authors:**  
- G.A.S. De Silva – 2021/E/037  
- H.M.U.C. Herath – 2021/E/049  

**📅 Supervisor:** Dr. A. Kaneswaran  
**🛠 Co-Supervisor:** Eng. Y. Pirunthapan


## 📷 Project Demo
Add screenshots or diagrams here:  
