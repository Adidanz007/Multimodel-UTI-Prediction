# Multimodal UTI Prediction System

## 🚀 Overview
The **Multimodal UTI Prediction System** is a state-of-the-art AI-powered platform designed to accurately predict the risk of Urinary Tract Infections (UTI) by fusing two distinct data modalities:
1. **Clinical Tabular Data:** Patient demographics, symptoms, and urinalysis biomarkers.
2. **Ultrasound Imaging:** Bladder ultrasound scans for visual anomaly detection.

By leveraging a late-fusion ensemble approach, this system intelligently combines a clinical machine learning model (XGBoost) and a deep learning image feature extractor (EfficientNetB3) through a Meta-learner (Logistic Regression). This multimodal strategy achieves superior diagnostic performance compared to unimodal approaches, providing a robust, research-grade screening tool.

## ✨ Key Features
- **Multimodal AI Fusion:** Achieves an impressive **AUC of 0.9145** by intelligently combining structured clinical data with unstructured medical imaging.
- **Deep Image Analysis:** Uses a fine-tuned EfficientNetB3 backbone with custom preprocessing (CLAHE, center-cropping, and artifact filtering) to extract anatomical bladder features.
- **Explainable AI (XAI):** Implements multimodal Grad-CAM to generate visual heatmaps, highlighting exactly where the model focuses on the ultrasound scan, fostering clinical trust.
- **Clinical Biomarker Engineering:** Utilizes an optimized XGBoost pipeline for comprehensive tabular data analysis.
- **Professional Web Application:** Features a beautiful, responsive dark/light-themed Flask web dashboard built for both patients and physicians, complete with a step-by-step screening wizard, visual processing states, and a dedicated physician portal.
- **Automated PDF Reports:** Auto-generates detailed medical summary PDFs based on AI thresholds, ready for clinical review.

## 📊 Model Performance
| Model Modality | Architecture | AUC | Accuracy |
|----------------|--------------|-----|----------|
| Clinical Only | XGBoost | 0.8105 | 74.0% |
| Imaging Only | EfficientNetB3 | 0.8843 | 80.3% |
| **Fusion AI** | **Calibrated LR Meta-learner**| **0.9145** | **84.2%** |

## 🏗️ Project Architecture

```
project/
├── data/
│   ├── raw/                 # Original tabular and image datasets
│   └── processed/           # Cleaned and augmented datasets
├── models/                  # Serialized Keras and Scikit-Learn models
├── results/                 # Training logs, ROC curves, and Grad-CAM grids
├── src/                     # Core ML source code
│   ├── multimodal_fusion_v2.py # Meta-learning and fusion logic
│   ├── predict_v2.py        # Centralized inference script
│   └── extract_embeddings...# EfficientNetB3 image processing
└── webapp/                  # Production-ready Flask application
    ├── app.py               # Flask backend routing and API logic
    ├── static/              # CSS (Light/Dark themes), JS, and assets
    └── templates/           # Jinja2 HTML templates
```

## 🧠 How the Multimodal Pipeline Works
1. **Clinical Branch:** Tabular data (31 features) is cleaned, normalized, and processed by an XGBoost classifier, which outputs a clinical probability score.
2. **Image Branch:** Bladder ultrasound images undergo rigorous preprocessing (CLAHE contrast enhancement, 15% top/bottom cropping to remove UI artifacts) before being passed through a frozen EfficientNetB3 backbone. The fully connected layers map the image to an imaging probability score.
3. **Meta-Fusion Layer:** The outputs from both independent branches are concatenated alongside high-importance clinical features and fed into a Calibrated Logistic Regression model. This meta-learner learns to weigh the reliability of each modality dynamically, producing a final Fusion Risk Score.
4. **Grad-CAM Generation:** The system computes the gradient of the predicted class with respect to the final convolutional layer of the image backbone, generating a color-mapped heatmap overlaid on the original ultrasound.

## 💻 Running the Web Application
The web app is the best way to interact with the trained models.

### Prerequisites
Make sure you have installed the required dependencies. A virtual environment is highly recommended.
```bash
# Install backend model dependencies
pip install -r requirements.txt

# Install web application dependencies
cd webapp
pip install -r requirements_webapp.txt
```

### Starting the Server
From the root directory of the project, navigate to the `webapp` folder and start the Flask server:
```bash
cd webapp
python app.py
```
The server will start on `port 5000`. Open your browser and navigate to: **[http://localhost:5000](http://localhost:5000)**

## 🩺 Usage Guide
1. **Landing Page:** Outlines the multimodal capabilities and model metrics.
2. **Screening Wizard (`/screening`):** A 3-step form where users enter patient info, symptoms/lab values, and upload a bladder ultrasound image.
3. **Processing Engine:** Simulates the multimodal pipeline execution visually.
4. **Result Dashboard:** Displays the overall Risk Gauge, Grad-CAM heatmap, prediction confidence, clinical interpretations, and model agreement breakdown.
5. **Physician Portal (`/doctor`):** A centralized dashboard for clinicians to review historical automated screenings and download PDF reports.

## ⚠️ Disclaimer
*This project is an AI research prototype intended for screening assistance and educational purposes only. It is not an FDA-approved diagnostic tool and does not substitute professional medical advice. Always consult a licensed healthcare provider for an official diagnosis and treatment plan.*