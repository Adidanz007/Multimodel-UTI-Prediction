# Multimodal UTI Prediction System - Web Application

This folder contains the complete, production-ready Flask Web Application wrapping the Multimodal AI Pipeline (Phase 9 of the project).

## 🚀 How to Run the Application

The web app is separated from the ML training code. All required assets are saved correctly.

### 1. Prerequisites
Ensure you have the required packages installed in your Python environment.
A dedicated `requirements_webapp.txt` was created for the web phase:

```bash
cd webapp
pip install -r requirements_webapp.txt
```

*(Note: The main `models` and `src` directories in the root are required for inference. The app imports them securely).*

### 2. Start the Server
From the root directory of the project, run:

```bash
cd webapp
python app.py
```

### 3. Access the Dashboard
The server will start on port 5000. Open your web browser and navigate to:
**[http://localhost:5000](http://localhost:5000)**

---

## 📂 Web App Structure

```
webapp/
├── app.py                 # Main Flask server & API endpoints
├── requirements_webapp.txt# Dependencies for rendering/web
├── README_webapp.md       # This file
├── static/
│   ├── css/
│   │   └── style.css      # Dark-themed modern UI styles
│   └── js/
│       └── main.js        # Multi-step forms & Canvas animations
└── templates/
    ├── index.html         # Landing page promoting AI performance
    ├── screening.html     # 3-Step Wizard for Clinical & Image Input
    ├── processing.html    # Timed animation mimicking ML processing
    ├── result.html        # Final Dashboard with Heatmaps & Fusion Score
    └── doctor.html        # Physician Portal listing recent predictions
```

## ✨ Application Features

- **Full Stack Integration**: Native linkage to the final `predict_final.py` models inside `../models/`.
- **Multi-Step Form Validation**: Captures 31 clinical biomarkers and an ultrasound image systematically.
- **Multimodal AI Real-Time Processing**: The backend invokes EfficientNetB3, Random Forest, and a Meta-learner dynamically.
- **Grad-CAM Visualization**: Automatically saves generated heatmaps to `/static/uploads/` and links them to the UI securely.
- **Dynamic PDF Reporting**: Built with ReportLab, auto-generates a medical summary download for patients/doctors dynamically based on AI thresholds (AUC=0.91 logic).
- **Responsive Dark UI**: Designed for medical tablet and desktop views with CSS3 fluid animations.

*End of Project Generation*