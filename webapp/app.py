from flask import Flask, render_template, request, jsonify, session, send_file
from flask_session import Session
import os, sys, uuid, json, time
from datetime import datetime
from werkzeug.utils import secure_filename

# Add parent directory to path so we can import UTIPredictor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict_final import UTIPredictor

WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(WEBAPP_DIR, 'templates'), static_folder=os.path.join(WEBAPP_DIR, 'static'))
app.secret_key = 'uti-prediction-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(os.path.join(WEBAPP_DIR, 'uploads'), exist_ok=True)
os.makedirs(os.path.join(WEBAPP_DIR, 'reports'), exist_ok=True)

print(f"[APP] Static folder: {app.static_folder}")
print(f"[APP] style.css exists: {os.path.exists(os.path.join(app.static_folder, 'css', 'style.css'))}")
print(f"[APP] main.js exists: {os.path.exists(os.path.join(app.static_folder, 'js', 'main.js'))}")


# Load models ONCE at startup
print("[APP] Loading AI models...")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_file = os.path.join(BASE_DIR, 'models', 'inference_config_final.json')
predictor = UTIPredictor(config_path=config_file)
print("[APP] Models loaded. Starting server...")

# In-memory result store (in production use a database)
results_store = {}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/screening')
def screening():
    return render_template('screening.html')

@app.route('/processing/<result_id>')
def processing(result_id):
    return render_template('processing.html', result_id=result_id)

@app.route('/result/<result_id>')
def result(result_id):
    result_data = results_store.get(result_id)
    if not result_data:
        return render_template('error.html'), 404
    return render_template('result.html', result=result_data, result_id=result_id)

@app.route('/test-static')
def test_static():
    return f"Static folder: {app.static_folder}\nstyle.css: {os.path.exists(os.path.join(app.static_folder, 'css', 'style.css'))}\nmain.js: {os.path.exists(os.path.join(app.static_folder, 'js', 'main.js'))}", 200, {'Content-Type': 'text/plain'}

@app.route('/api/check-result/<result_id>')
def check_result(result_id):
    if result_id in results_store:
        return jsonify({'ready': True})
    return jsonify({'ready': False}), 202

@app.route('/doctor')
def doctor():
    # Pass all results for doctor dashboard
    all_results = list(results_store.values())
    return render_template('doctor.html', patients=all_results)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get clinical data from form
        clinical_data = {}
        form_data = request.form.to_dict()

        # Parse all 31 clinical features
        feature_names_path = os.path.join(BASE_DIR, 'models', 'clinical_feature_names_4k.json')
        with open(feature_names_path) as f:
            feature_names = json.load(f)
        for feature in feature_names:
            val = form_data.get(feature, '0')
            try:
                clinical_data[feature] = float(val)
            except:
                clinical_data[feature] = 0.0

        # Handle image upload
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        if not allowed_file(image_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save uploaded image
        img_filename = f"{uuid.uuid4().hex}.jpg"
        img_path = os.path.join(WEBAPP_DIR, 'uploads', img_filename)
        image_file.save(img_path)

        # Generate Result ID
        result_id = uuid.uuid4().hex[:8].upper()
        
        # Run prediction synchronously to avoid TF thread crashes
        result = predictor.predict(clinical_data, img_path)

        # Generate Grad-CAM heatmap
        heatmap_filename = f"heatmap_{img_filename}"
        heatmap_path = os.path.join(WEBAPP_DIR, 'static', 'heatmaps', heatmap_filename)
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        generate_gradcam(img_path, heatmap_path)
        result['heatmap_url'] = f"/static/heatmaps/{heatmap_filename}"
        result['original_image_url'] = f"/uploads/{img_filename}"

        # Add metadata
        result_id = uuid.uuid4().hex[:8].upper()
        result['result_id'] = result_id
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result['patient_age'] = int(clinical_data.get('age', 0))
        result['patient_gender'] = 'Female' if clinical_data.get('gender', 0) == 0 else 'Male'
        
        # --- NEW: Biomarker Flags & Model Agreement ---
        biomarker_flags = []
        if clinical_data.get('leukocytes', 0) > 0:
            biomarker_flags.append({"name": "High Leukocytes", "level": "danger", "value": f"+{clinical_data.get('leukocytes', 0)}"})
        if clinical_data.get('nitrites', 0) > 0:
            biomarker_flags.append({"name": "Positive Nitrites", "level": "danger", "value": "Found"})
        if clinical_data.get('blood', 0) > 0:
            biomarker_flags.append({"name": "Blood in Urine", "level": "warning", "value": "Present"})
        if clinical_data.get('protein', 0) > 0:
            biomarker_flags.append({"name": "Proteinuria", "level": "warning", "value": "Elevated"})
        if not biomarker_flags:
            biomarker_flags.append({"name": "No major flags", "level": "success", "value": "Normal"})
            
        result['biomarker_flags'] = biomarker_flags
        
        # Calculate Model Agreement
        cs = result.get('clinical_score', 0)
        u_s = result.get('image_score', 0)
        fs = result.get('fusion_score', 0)
        diff = abs(cs - u_s)
        
        clinical_pred = 'Positive' if cs > 0.5 else 'Negative'
        image_pred = 'Positive' if u_s > 0.5 else 'Negative'
        fusion_pred = 'Positive' if fs > 0.5 else 'Negative'
        
        if diff < 0.15:
            consensus_text = "Strong Agreement"
            consensus_color = "accent-teal" # from your CSS, or "success" class
        elif diff < 0.35:
            consensus_text = "Partial Agreement"
            consensus_color = "primary"
        else:
            consensus_text = "Divergent Predictions"
            consensus_color = "accent-orange"
            
        result['model_agreement'] = {
            "clinical": clinical_pred,
            "image": image_pred,
            "fusion": fusion_pred,
            "consensus": consensus_text,
            "consensus_color": consensus_color,
            "difference_score": round(diff * 100, 1)
        }
        # -----------------------------------------------

        # Add additional metadata for rendering
        result['id'] = result_id
        result['timestamp'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        
        age = clinical_data.get('age', clinical_data.get('Age', 0))
        gender_val = clinical_data.get('gender', clinical_data.get('Gender', 0))
        result['patient_age'] = int(age)
        result['patient_gender'] = "Male" if int(gender_val) == 1 else "Female"
        
        if result['fusion_score'] >= 0.5:
            result['prediction'] = 'UTI Positive'
        else:
            result['prediction'] = 'UTI Negative'

        result['img_filename'] = img_filename
        result['heatmap_filename'] = heatmap_filename
        result['interpretation'] = get_clinical_interpretation(result)
        result['recommendations'] = get_recommendations(result)
        result['urgency_level'] = get_urgency_level(result)

        # Store result
        results_store[result_id] = result

        return jsonify({'success': True, 'result_id': result_id})

    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(WEBAPP_DIR, 'uploads', filename))

@app.route('/api/report/<result_id>')
def download_report(result_id):
    result_data = results_store.get(result_id)
    if not result_data:
        return jsonify({'error': 'Result not found'}), 404
    pdf_path = generate_pdf_report(result_data, result_id)
    return send_file(pdf_path, as_attachment=True,
                     download_name=f'UTI_Report_{result_id}.pdf')

def get_clinical_interpretation(result):
    fs = result['fusion_score']
    cs = result['clinical_score']
    is_ = result['image_score']

    if fs >= 0.75:
        return ("High probability of UTI detected. Both clinical biomarkers "
                "and ultrasound imaging show significant abnormality indicators. "
                "Immediate medical consultation recommended.")
    elif fs >= 0.50:
        return ("Moderate UTI risk detected. Clinical markers suggest infection "
                "presence. Ultrasound shows possible bladder changes. "
                "Medical evaluation advised within 24 hours.")
    elif fs >= 0.35:
        return ("Low UTI risk. Some mild indicators present but below threshold. "
                "Monitor symptoms. Consult doctor if symptoms worsen.")
    else:
        return ("Minimal UTI risk detected. Clinical biomarkers and ultrasound "
                "appear within normal range. Continue monitoring if symptoms persist.")

def get_recommendations(result):
    fs = result['fusion_score']
    recs = []
    if fs >= 0.65:
        recs = [
            "Consult a urologist immediately",
            "Begin urine culture test (UCX)",
            "Consider empirical antibiotic therapy",
            "Increase fluid intake (2-3 litres/day)",
            "Follow-up ultrasound in 1 week"
        ]
    elif fs >= 0.48:
        recs = [
            "Schedule doctor consultation within 24-48 hours",
            "Collect midstream urine sample for culture",
            "Monitor temperature every 6 hours",
            "Increase fluid intake",
            "Avoid urinary irritants (caffeine, alcohol)"
        ]
    else:
        recs = [
            "Monitor symptoms for 48 hours",
            "Maintain adequate hydration",
            "Repeat test if symptoms develop",
            "Practice good urinary hygiene",
            "Consider probiotic supplements"
        ]
    return recs

def get_urgency_level(result):
    fs = result['fusion_score']
    if fs >= 0.75:
        return {'level': 'URGENT', 'color': '#E24B4A', 'icon': '🚨'}
    elif fs >= 0.50:
        return {'level': 'MODERATE', 'color': '#F5A623', 'icon': '⚠️'}
    elif fs >= 0.35:
        return {'level': 'LOW', 'color': '#4FC3F7', 'icon': '📋'}
    else:
        return {'level': 'MINIMAL', 'color': '#00A896', 'icon': '✅'}

def generate_gradcam(img_path, save_path):
    """Generate Grad-CAM heatmap overlay for the uploaded image."""
    import numpy as np
    import cv2
    import tensorflow as tf

    img = predictor.preprocess_image(img_path)
    img_batch = np.expand_dims(img, 0)

    # In Keras 3, recursively reconstruct the graph to keep tensors connected
    x = predictor.img_model.input
    conv = predictor.img_model.get_layer('efficientnetb3')(x)
    
    current = conv
    start_index = predictor.img_model.layers.index(predictor.img_model.get_layer('efficientnetb3')) + 1
    for layer in predictor.img_model.layers[start_index:]:
        current = layer(current)
        
    grad_model = tf.keras.Model(inputs=x, outputs=[conv, current])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    # Resize and colorize
    original = cv2.imread(img_path)
    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
    cv2.imwrite(save_path, overlay)

def generate_pdf_report(result, result_id):
    """Generate downloadable PDF medical report."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.units import inch

    pdf_path = os.path.join(WEBAPP_DIR, 'reports', f"UTI_Report_{result_id}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Header
    title_style = ParagraphStyle('Title', parent=styles['Title'],
        fontSize=20, textColor=colors.HexColor('#028090'), spaceAfter=6)
    story.append(Paragraph("AI-Assisted UTI & Bladder Screening Report", title_style))
    story.append(Paragraph(f"Report ID: {result_id}  |  {result['timestamp']}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Patient info
    story.append(Paragraph("Patient Information", styles['Heading2']))
    patient_data = [
        ['Age', str(result.get('patient_age', 'N/A'))],
        ['Gender', result.get('patient_gender', 'N/A')],
        ['Test Date', result['timestamp']],
    ]
    t = Table(patient_data, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#E8ECF4')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))

    # AI Results
    story.append(Paragraph("AI Analysis Results", styles['Heading2']))
    result_color = colors.red if result['fusion_score'] >= 0.5 else colors.green
    results_data = [
        ['Metric', 'Score', 'Interpretation'],
        ['Clinical Risk Score', f"{result['clinical_score']:.1%}", 'Biomarker analysis'],
        ['Image Risk Score', f"{result['image_score']:.1%}", 'Ultrasound analysis'],
        ['Combined Fusion Score', f"{result['fusion_score']:.1%}", 'Multimodal AI'],
        ['Final Prediction', result['prediction'], result['confidence']],
    ]
    t2 = Table(results_data, colWidths=[2.2*inch, 1.5*inch, 2.8*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E2761')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.white, colors.HexColor('#F4F7FF')]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.2*inch))

    # Interpretation
    story.append(Paragraph("Clinical Interpretation", styles['Heading2']))
    story.append(Paragraph(result.get('clinical_interpretation', result.get('interpretation', '')), styles['Normal']))
    story.append(Spacer(1, 0.15*inch))

    # Recommendations
    story.append(Paragraph("Recommendations", styles['Heading2']))
    for rec in result['recommendations']:
        story.append(Paragraph(f"• {rec}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Heatmap image if exists
    heatmap_path = os.path.join(WEBAPP_DIR, 'static', 'heatmaps', os.path.basename(result.get('heatmap_url', '').lstrip('/')))
    if os.path.exists(heatmap_path):
        story.append(Paragraph("Ultrasound Grad-CAM Heatmap", styles['Heading2']))
        story.append(RLImage(heatmap_path, width=4*inch, height=3*inch))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "Heatmap shows regions the AI focused on — red/yellow = high attention areas",
            styles['Italic']))

    # Disclaimer
    story.append(Spacer(1, 0.3*inch))
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'],
        fontSize=8, textColor=colors.grey)
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI model (AUC=0.9145) for "
        "screening purposes only. It is not a substitute for professional medical "
        "diagnosis. Always consult a qualified healthcare provider.",
        disclaimer_style))

    doc.build(story)
    return pdf_path

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)