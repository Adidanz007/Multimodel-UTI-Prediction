import os
import json
import joblib
import numpy as np

def run_tests():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    print("="*63)
    print("  DIRECT MODEL TEST STARTING")
    print("="*63)
    
    # STEP 1: Load all models and print confirmation
    print("\n--- STEP 1: Loading Models ---")
    models = {}
    try:
        models['clinical_model'] = joblib.load(os.path.join(MODELS_DIR, 'clinical_model_4k.pkl'))
        print(f"Loaded clinical_model_4k.pkl: {type(models['clinical_model'])}")
        
        models['clinical_scaler'] = joblib.load(os.path.join(MODELS_DIR, 'clinical_scaler_4k.pkl'))
        print(f"Loaded clinical_scaler_4k.pkl: {type(models['clinical_scaler'])}")
        
        with open(os.path.join(MODELS_DIR, 'clinical_feature_names_4k.json'), 'r') as f:
            models['feature_names'] = json.load(f)
        print(f"Loaded clinical_feature_names_4k.json: list of {len(models['feature_names'])} features")
        
        models['fusion_model'] = joblib.load(os.path.join(MODELS_DIR, 'fusion_model_calibrated.pkl'))
        print(f"Loaded fusion_model_calibrated.pkl: {type(models['fusion_model'])}")
        
        models['pca'] = joblib.load(os.path.join(MODELS_DIR, 'pca_32.pkl'))
        print(f"Loaded pca_32.pkl: {type(models['pca'])}")
        
        with open(os.path.join(MODELS_DIR, 'fusion_threshold.txt'), 'r') as f:
            models['fusion_threshold'] = float(f.read().strip())
        print(f"Loaded fusion_threshold.txt: {models['fusion_threshold']}")
        
    except Exception as e:
        print(f"FAILED to load models. Error: {e}")
        return

    # STEP 2: Print exactly what the model expects
    print("\n--- STEP 2: Checking Expectations ---")
    feature_names = models['feature_names']
    print(f"Ordered feature names:\n{feature_names}")
    
    expected_scaler_features = models['clinical_scaler'].n_features_in_
    print(f"Scaler expects {expected_scaler_features} features.")
    
    if hasattr(models['clinical_model'], 'n_features_in_'):
        expected_xgb_features = models['clinical_model'].n_features_in_
    else:
        try:
            expected_xgb_features = models['clinical_model'].get_booster().num_features
        except:
            expected_xgb_features = expected_scaler_features
            
    print(f"XGBoost model expects {expected_xgb_features} features.")
    
    if len(feature_names) == expected_scaler_features and len(feature_names) == expected_xgb_features:
        print("Feature counts match: PASS")
    else:
        print("Feature counts DO NOT MATCH: FAIL")

    def predict_case(features_dict, expected_str, expected_bool, test_name, img_prob=None):
        print(f"\n--- STEP: Test with {test_name} ---")
        x_clinical = np.zeros((1, len(feature_names)))
        
        # Helper to do case-insensitive match for keys
        lower_features_dict = {k.lower(): v for k, v in features_dict.items()}
        
        for i, f_name in enumerate(feature_names):
            val = lower_features_dict.get(f_name.lower(), 0.0)
            x_clinical[0, i] = float(val)
            
        x_scaled = models['clinical_scaler'].transform(x_clinical)
        clin_prob = models['clinical_model'].predict_proba(x_scaled)[0, 1]
        print(f"Raw clinical probability: {clin_prob:.4f}")
        
        # If the user didn't give an image, mock img_prob as clin_prob to simulate matching modalities
        # The fusion model expects [clin_prob, img_prob, clin_prob * img_prob]
        if img_prob is None:
            img_prob = clin_prob
            
        x_fusion = np.array([[clin_prob, img_prob, clin_prob * img_prob]])
        fusion_prob = models['fusion_model'].predict_proba(x_fusion)[0, 1]
        print(f"Fusion probability: {fusion_prob:.4f}")
        
        is_positive = fusion_prob >= models['fusion_threshold']
        pred_str = 'UTI Positive' if is_positive else 'UTI Negative'
        print(f"Prediction: {pred_str}")
        
        if expected_bool is None:
            match_str = "N/A"
            expected_str = "Either"
        else:
            match_str = "YES" if (is_positive == expected_bool) else "NO"
            
        print(f"Matches expected ({expected_str})? {match_str}")
        
        return clin_prob, fusion_prob, pred_str, expected_str, match_str

    # STEP 3: Strongly Infected Case
    infected_features = {
        'nitrite': 1, 'leukocyte_esterase': 1, 'urine_bacteria': 1, 'urine_wbc': 1, 'urine_blood': 1,
        'fever': 1, 'burning_urination': 1, 'back_pain': 1, 'abdominal_pain': 1,
        'urine_ph': 8.0, 'specific_gravity': 1.025, 'urine_rbc': 1,
        'WBC': 15.0, 'RBC': 3.5, 'Temperature': 38.8,
        'age': 35, 'gender': 0
    }
    c_prob1, f_prob1, p_str1, e_str1, match1 = predict_case(infected_features, "Positive", True, "a STRONGLY INFECTED case")

    # STEP 4: Clearly Normal Case
    normal_features = {
        'nitrite': 0, 'leukocyte_esterase': 0, 'urine_bacteria': 0, 'urine_wbc': 0, 'urine_blood': 0,
        'fever': 0, 'burning_urination': 0, 'back_pain': 0, 'abdominal_pain': 0,
        'urine_ph': 6.0, 'specific_gravity': 1.015, 'urine_rbc': 0,
        'WBC': 7.0, 'RBC': 4.5, 'Temperature': 36.8,
        'age': 28, 'gender': 0
    }
    c_prob2, f_prob2, p_str2, e_str2, match2 = predict_case(normal_features, "Negative", False, "a CLEARLY NORMAL case")

    # STEP 5: Borderline Case
    borderline_features = {
        'nitrite': 0, 'leukocyte_esterase': 1, 'urine_bacteria': 1, 'urine_wbc': 1,
        'fever': 1, 'burning_urination': 0, 'back_pain': 1,
        'urine_ph': 7.2, 'specific_gravity': 1.020,
        'WBC': 12.0, 'Temperature': 37.8,
        'age': 45, 'gender': 0
    }
    c_prob3, f_prob3, p_str3, e_str3, match3 = predict_case(borderline_features, "Either", None, "a BORDERLINE case")

    # STEP 6: Summary Table
    print("\n--- STEP 6: Summary Table ---")
    print("═══════════════════════════════════════════════════════════════")
    print("  DIRECT MODEL TEST RESULTS")
    print("═══════════════════════════════════════════════════════════════")
    print(f"  {'Test Case':<18} {'Clinical%':<10} {'Fusion%':<9} {'Prediction':<12} {'Expected':<10} {'Match?'}")
    
    def format_row(tc, c, f, p, e, m):
        c_str = f"{c*100:.1f}%"
        f_str = f"{f*100:.1f}%"
        print(f"  {tc:<18} {c_str:<10} {f_str:<9} {p:<12} {e:<10} {m}")
        
    format_row('Strong Infected', c_prob1, f_prob1, p_str1, e_str1, match1)
    format_row('Clearly Normal', c_prob2, f_prob2, p_str2, e_str2, match2)
    format_row('Borderline', c_prob3, f_prob3, p_str3, e_str3, match3)
    
    print("═══════════════════════════════════════════════════════════════")
    print("  If Strong Infected shows Negative → MODEL OR PREPROCESSING IS BROKEN")
    print("  If Clearly Normal shows Positive  → MODEL IS BIASED TOWARD POSITIVE")
    print("  If both correct                   → MODEL IS WORKING, ISSUE IS IN WEB FORM")
    print("═══════════════════════════════════════════════════════════════")

if __name__ == '__main__':
    run_tests()
