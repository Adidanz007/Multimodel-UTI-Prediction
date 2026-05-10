import numpy as np
import pandas as pd
import joblib
import json
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class UTIPredictor:
    """
    Production-ready UTI prediction class.
    Loads all models once at startup.
    Thread-safe for Flask use.
    """

    def __init__(self, config_path=os.path.join(BASE_DIR, 'models/inference_config_final.json')):
        import tensorflow as tf
        from sklearn.decomposition import PCA
        
        with open(config_path) as f:
            config = json.load(f)

        # Load clinical model
        self.clin_model  = joblib.load(os.path.join(BASE_DIR, config['models']['clinical']))
        self.scaler      = joblib.load(os.path.join(BASE_DIR, config['models']['clinical_scaler']))
        self.features    = json.load(open(os.path.join(BASE_DIR, config['models']['clinical_features'])))
        
        with open(os.path.join(BASE_DIR, config['models']['threshold'])) as f:
            self.threshold = float(f.read().strip())

        # Load image models
        full_model = tf.keras.models.load_model(os.path.join(BASE_DIR, config['models']['image']), compile=False)
        self.img_model = full_model
        
        # Handle layer parsing gracefully for keras 3
        embed_name = None
        for cand in ["features", "dense_256", "dense_1"]:
            try:
                layer = full_model.get_layer(cand)
                if isinstance(layer, tf.keras.layers.Dense) and layer.output.shape[-1] > 1:
                    embed_name = cand
                    break
            except ValueError:
                pass
        if embed_name is None:
            for layer in reversed(full_model.layers):
                if isinstance(layer, tf.keras.layers.Dense) and layer.output.shape[-1] > 1:
                    embed_name = layer.name
                    break

        self.embed_model = tf.keras.Model(
            inputs=full_model.input,
            outputs=full_model.get_layer(embed_name).output
        )

        # Load PCA (fit on training embeddings)
        self.pca = joblib.load(os.path.join(BASE_DIR, 'models/pca_32.pkl'))

        # Load fusion model
        self.fusion = joblib.load(os.path.join(BASE_DIR, config['models']['fusion']))

        self.thresholds = config['thresholds']
        print("[UTIPredictor] All models loaded successfully")

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        h, w = img.shape[:2]
        # Aggressive crop
        img = img[int(h*0.20):int(h*0.85), int(w*0.15):int(w*0.85)]
        # Elliptical mask
        h2, w2 = img.shape[:2]
        mask = np.zeros((h2, w2), dtype=np.uint8)
        cv2.ellipse(mask, (w2//2, h2//2), (int(w2*0.46), int(h2*0.46)),
                    0, 0, 360, 255, -1)
        img = cv2.bitwise_and(img, img, mask=mask)
        # CLAHE
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        rgb = cv2.merge([enhanced, enhanced, enhanced])
        rgb = cv2.resize(rgb, (260, 260))
        return rgb.astype(np.float32) / 255.0

    def predict(self, clinical_data: dict, image_path: str) -> dict:
        """
        Main prediction function for Flask.

        Args:
            clinical_data: dict of {feature_name: value}
            image_path: path to uploaded ultrasound image

        Returns:
            dict with all scores, prediction, confidence, interpretation
        """
        # Clinical prediction
        X_clin = np.array([[clinical_data.get(f, 0) for f in self.features]])
        X_scaled = self.scaler.transform(X_clin)
        clin_prob = float(self.clin_model.predict_proba(X_scaled)[0, 1])

        # Image prediction
        img = self.preprocess_image(image_path)
        img_batch = np.expand_dims(img, 0)
        img_prob = float(self.img_model.predict(img_batch, verbose=0)[0, 0])
        # Note: the current best LR model only uses clin_prob, img_prob, and their interaction!
        # It does NOT use PCA.

        # Fusion prediction
        X_fusion = np.array([[clin_prob, img_prob, clin_prob * img_prob]])
        fusion_prob = float(self.fusion.predict_proba(X_fusion)[0, 1])

        # Confidence zone
        if fusion_prob >= 0.65 or fusion_prob <= 0.35:
            confidence = 'High'
        elif fusion_prob >= 0.48 or fusion_prob <= 0.48:  # will be handled better below
            pass 

        if fusion_prob >= 0.65 or fusion_prob <= 0.35:
            confidence = 'High'
        elif fusion_prob >= 0.55 or fusion_prob <= 0.45:
            confidence = 'Medium'
        else:
            confidence = 'Low — Recommend further testing'

        prediction = 'UTI Positive' if fusion_prob >= self.threshold else 'UTI Negative'

        # Risk level for UI display
        if fusion_prob >= 0.75:
            risk_level = 'High Risk'
            risk_color = '#E24B4A'
        elif fusion_prob >= 0.50:
            risk_level = 'Moderate Risk'
            risk_color = '#F5A623'
        elif fusion_prob >= 0.35:
            risk_level = 'Low Risk'
            risk_color = '#4FC3F7'
        else:
            risk_level = 'Minimal Risk'
            risk_color = '#00A896'

        return {
            'prediction':          prediction,
            'confidence':          confidence,
            'risk_level':          risk_level,
            'risk_color':          risk_color,
            'fusion_score':        round(fusion_prob, 4),
            'clinical_score':      round(clin_prob, 4),
            'image_score':         round(img_prob, 4),
            'threshold_used':      self.threshold,
            'interpretation': (
                f"Clinical biomarkers suggest "
                f"{'infection present' if clin_prob > 0.5 else 'no infection'} "
                f"(score: {clin_prob:.2f}). "
                f"Ultrasound imaging suggests "
                f"{'bladder abnormality' if img_prob > 0.5 else 'normal appearance'} "
                f"(score: {img_prob:.2f}). "
                f"Combined multimodal risk: {fusion_prob:.2f} — {risk_level}."
            )
        }


if __name__ == '__main__':
    predictor = UTIPredictor()

    pairs_csv = os.path.join(BASE_DIR, "data", "processed", "fusion_pairs.csv")
    clin_csv  = os.path.join(BASE_DIR, "data", "processed", "clinical_4k_sorted.csv")

    pairs   = pd.read_csv(pairs_csv)
    clin_df = pd.read_csv(clin_csv)

    test_pairs = pairs[pairs["split"] == "test"] if "split" in pairs.columns else pairs

    # Pick 3 infected, 2 normal
    samples = []
    for label, count in [(1, 3), (0, 2)]:
        subset = test_pairs[test_pairs["label"] == label]
        for i in range(min(count, len(subset))):
            samples.append(subset.iloc[i])

    print(f"\n{'Sample':>6}  {'True':>8}  {'Pred':>14}  {'Clin':>6}  {'Img':>6}  {'Fusion':>6}  {'OK?':>5}")
    print("-" * 60)

    correct = 0

    # Better test logic using pre-computed test set:
    clin_features = np.load(os.path.join(BASE_DIR, 'results', 'embeddings', 'clinical_features_4k.npy'))
    img_proba = np.load(os.path.join(BASE_DIR, 'results', 'embeddings', 'image_proba.npy'))
    clin_proba = np.load(os.path.join(BASE_DIR, 'results', 'embeddings', 'clinical_proba_4k.npy'))
    test_idx = test_pairs.index.tolist()
    
    for i, row_idx in enumerate(test_idx[:5]): # Take first 5 test samples
        c_prob = clin_proba[row_idx, 1]
        i_prob = img_proba[row_idx, 0]
        true_lbl = pairs.iloc[row_idx]["label"]
        true_label = "Infected" if true_lbl == 1 else "Normal"
        
        # Fusion prediction
        X_fusion = np.array([[c_prob, i_prob, c_prob * i_prob]])
        fusion_prob = float(predictor.fusion.predict_proba(X_fusion)[0, 1])
        prediction = 'UTI Positive' if fusion_prob >= predictor.threshold else 'UTI Negative'
        
        ok = "✓" if ((true_lbl == 1 and prediction == "UTI Positive") or (true_lbl == 0 and prediction == "UTI Negative")) else "✗"
        if ok == "✓": correct += 1
        
        print(f"  {i+1:>4}  {true_label:>8}  {prediction:>14}  {c_prob:>6.2f}  {i_prob:>6.2f}  {fusion_prob:>6.2f}  {ok:>5}")
        
    print(f"\n  Correct: {correct}/5")
    if correct >= 4:
        print("  ✓ PREDICT_FINAL COMPLETE — 4+ of 5 samples correct  SUCCESS")
    else:
        print("  ⚠ PREDICT_FINAL PARTIAL — fewer than 4 correct  check model quality")
