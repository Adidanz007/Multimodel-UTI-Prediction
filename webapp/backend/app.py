from __future__ import annotations

import json
import os
import sys
import uuid
from numbers import Number
from pathlib import Path
from typing import Any

from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict_final import UTIPredictor  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
FEATURES_PATH = ROOT_DIR / "models" / "clinical_feature_names_4k.json"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "uti-prediction-secret")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

PREDICTOR: UTIPredictor | None = None


FIELD_SPECS: list[dict[str, Any]] = [
    {
        "name": "UCX_abnormal",
        "label": "UCX abnormal",
        "section": "Urine panel",
        "kind": "select",
        "options": [("no", "No"), ("yes", "Yes")],
        "default": "no",
    },
    {
        "name": "urine_bacteria",
        "label": "Urine bacteria",
        "section": "Urine panel",
        "kind": "select",
        "options": [
            ("none", "None"),
            ("few", "Few"),
            ("moderate", "Moderate"),
            ("many", "Many"),
            ("marked", "Marked"),
        ],
        "default": "none",
    },
    {
        "name": "urine_bilirubin",
        "label": "Urine bilirubin",
        "section": "Urine panel",
        "kind": "select",
        "options": [
            ("negative", "Negative"),
            ("small", "Small"),
            ("moderate", "Moderate"),
            ("large", "Large"),
        ],
        "default": "negative",
    },
    {
        "name": "urine_blood",
        "label": "Urine blood",
        "section": "Urine panel",
        "kind": "select",
        "options": [
            ("negative", "Negative"),
            ("small", "Small"),
            ("moderate", "Moderate"),
            ("large", "Large"),
        ],
        "default": "negative",
    },
    {
        "name": "urine_clarity",
        "label": "Urine clarity",
        "section": "Urine panel",
        "kind": "select",
        "options": [("clear", "Clear"), ("not_clear", "Not clear")],
        "default": "clear",
    },
    {
        "name": "urine_color",
        "label": "Urine color",
        "section": "Urine panel",
        "kind": "select",
        "options": [
            ("colorless", "Colorless"),
            ("yellow", "Yellow"),
            ("amber", "Amber"),
            ("orange", "Orange"),
            ("red", "Red"),
            ("other", "Other"),
        ],
        "default": "yellow",
    },
    {
        "name": "epithelial_cells",
        "label": "Epithelial cells",
        "section": "Urine panel",
        "kind": "select",
        "options": [
            ("negative", "Negative"),
            ("small", "Small"),
            ("moderate", "Moderate"),
            ("large", "Large"),
        ],
        "default": "negative",
    },
    {
        "name": "urine_glucose",
        "label": "Urine glucose",
        "section": "Urine panel",
        "kind": "select",
        "options": [
            ("negative", "Negative"),
            ("small", "Small"),
            ("moderate", "Moderate"),
            ("large", "Large"),
            ("4+", "4+"),
        ],
        "default": "negative",
    },
    {
        "name": "urine_ketones",
        "label": "Urine ketones",
        "section": "Urine panel",
        "kind": "select",
        "options": [
            ("negative", "Negative"),
            ("small", "Small"),
            ("moderate", "Moderate"),
            ("large", "Large"),
            ("4+", "4+"),
        ],
        "default": "negative",
    },
    {
        "name": "leukocyte_esterase",
        "label": "Leukocyte esterase",
        "section": "Urine panel",
        "kind": "select",
        "options": [
            ("negative", "Negative"),
            ("small", "Small"),
            ("moderate", "Moderate"),
            ("large", "Large"),
        ],
        "default": "negative",
    },
    {
        "name": "nitrite",
        "label": "Nitrite",
        "section": "Urine panel",
        "kind": "select",
        "options": [("negative", "Negative"), ("positive", "Positive")],
        "default": "negative",
    },
    {
        "name": "urine_ph",
        "label": "Urine pH",
        "section": "Urine panel",
        "kind": "number",
        "step": "0.1",
        "min": "0",
        "max": "14",
        "default": "6.0",
    },
    {
        "name": "urine_protein",
        "label": "Urine protein",
        "section": "Urine panel",
        "kind": "select",
        "options": [
            ("negative", "Negative"),
            ("small", "Small"),
            ("moderate", "Moderate"),
            ("large", "Large"),
        ],
        "default": "negative",
    },
    {
        "name": "urine_rbc",
        "label": "Urine RBC",
        "section": "Urine panel",
        "kind": "select",
        "options": [
            ("negative", "Negative"),
            ("small", "Small"),
            ("moderate", "Moderate"),
            ("large", "Large"),
            ("other", "Other"),
        ],
        "default": "negative",
    },
    {
        "name": "specific_gravity",
        "label": "Specific gravity",
        "section": "Urine panel",
        "kind": "number",
        "step": "0.001",
        "min": "1.000",
        "max": "1.050",
        "default": "1.015",
    },
    {
        "name": "urobilinogen",
        "label": "Urobilinogen",
        "section": "Urine panel",
        "kind": "select",
        "options": [("negative", "Negative"), ("positive", "Positive")],
        "default": "negative",
    },
    {
        "name": "urine_wbc",
        "label": "Urine WBC",
        "section": "Urine panel",
        "kind": "select",
        "options": [
            ("negative", "Negative"),
            ("small", "Small"),
            ("moderate", "Moderate"),
            ("large", "Large"),
        ],
        "default": "negative",
    },
    {
        "name": "abdominal_tenderness",
        "label": "Abdominal tenderness",
        "section": "Symptoms",
        "kind": "select",
        "options": [("0", "No"), ("1", "Yes")],
        "default": "0",
    },
    {
        "name": "back_pain",
        "label": "Back pain",
        "section": "Symptoms",
        "kind": "select",
        "options": [("0", "No"), ("1", "Yes")],
        "default": "0",
    },
    {
        "name": "fatigue",
        "label": "Fatigue",
        "section": "Symptoms",
        "kind": "select",
        "options": [("0", "No"), ("1", "Yes")],
        "default": "0",
    },
    {
        "name": "fever",
        "label": "Fever",
        "section": "Symptoms",
        "kind": "select",
        "options": [("0", "No"), ("1", "Yes")],
        "default": "0",
    },
    {
        "name": "abdominal_pain",
        "label": "Abdominal pain",
        "section": "Symptoms",
        "kind": "select",
        "options": [("0", "No"), ("1", "Yes")],
        "default": "0",
    },
    {
        "name": "burning_urination",
        "label": "Burning urination",
        "section": "Symptoms",
        "kind": "select",
        "options": [("0", "No"), ("1", "Yes")],
        "default": "0",
    },
    {
        "name": "diff_urinating",
        "label": "Difficulty urinating",
        "section": "Symptoms",
        "kind": "select",
        "options": [("0", "No"), ("1", "Yes")],
        "default": "0",
    },
    {
        "name": "age",
        "label": "Age",
        "section": "Vitals & labs",
        "kind": "number",
        "step": "1",
        "min": "0",
        "max": "120",
        "default": "40",
    },
    {
        "name": "Temperature",
        "label": "Temperature (°C)",
        "section": "Vitals & labs",
        "kind": "number",
        "step": "0.1",
        "min": "30",
        "max": "45",
        "default": "37.0",
    },
    {
        "name": "RBC",
        "label": "RBC",
        "section": "Vitals & labs",
        "kind": "number",
        "step": "0.1",
        "min": "0",
        "default": "4.5",
    },
    {
        "name": "WBC",
        "label": "WBC",
        "section": "Vitals & labs",
        "kind": "number",
        "step": "0.1",
        "min": "0",
        "default": "7.0",
    },
    {
        "name": "gender",
        "label": "Gender",
        "section": "History",
        "kind": "select",
        "options": [("Male", "Male"), ("Female", "Female")],
        "default": "Male",
    },
    {
        "name": "Calculus_of_urinary_tract",
        "label": "Calculus of urinary tract",
        "section": "History",
        "kind": "select",
        "options": [("No", "No"), ("Yes", "Yes")],
        "default": "No",
    },
    {
        "name": "Urinary_tract_infections",
        "label": "Urinary tract infections",
        "section": "History",
        "kind": "select",
        "options": [("No", "No"), ("Yes", "Yes")],
        "default": "No",
    },
]

FIELD_MAP = {field["name"]: field for field in FIELD_SPECS}

ORDINAL_ENCODINGS: dict[str, dict[str, int]] = {
    "urine_bacteria": {"none": 0, "few": 1, "moderate": 2, "many": 3, "marked": 4},
    "urine_bilirubin": {"negative": 0, "small": 1, "moderate": 2, "large": 3},
    "urine_blood": {"negative": 0, "small": 1, "moderate": 2, "large": 3, "other": 1},
    "urine_clarity": {"clear": 0, "not_clear": 1},
    "urine_color": {"colorless": 0, "yellow": 1, "amber": 2, "orange": 3, "red": 4, "other": 2},
    "epithelial_cells": {"negative": 0, "small": 1, "moderate": 2, "large": 3},
    "urine_glucose": {"negative": 0, "small": 1, "moderate": 2, "large": 3, "4+": 4},
    "urine_ketones": {"negative": 0, "small": 1, "moderate": 2, "large": 3, "4+": 4},
    "leukocyte_esterase": {"negative": 0, "small": 1, "moderate": 2, "large": 3},
    "nitrite": {"negative": 0, "positive": 1},
    "urine_protein": {"negative": 0, "small": 1, "moderate": 2, "large": 3},
    "urine_rbc": {"negative": 0, "small": 1, "moderate": 2, "large": 3, "other": 1},
    "urobilinogen": {"negative": 0, "positive": 1},
    "urine_wbc": {"negative": 0, "small": 1, "moderate": 2, "large": 3},
    "UCX_abnormal": {"no": 0, "yes": 1},
    "gender": {"Male": 0, "Female": 1},
    "Calculus_of_urinary_tract": {"No": 0, "Yes": 1},
    "Urinary_tract_infections": {"No": 0, "Yes": 1},
}

BINARY_COLS = {
    "abdominal_tenderness",
    "back_pain",
    "fatigue",
    "fever",
    "abdominal_pain",
    "burning_urination",
    "diff_urinating",
}

NUMERIC_COLS = {"urine_ph", "specific_gravity", "age", "Temperature", "RBC", "WBC"}

FORM_SECTIONS: list[dict[str, Any]] = []
for field in FIELD_SPECS:
    section_name = field["section"]
    section = next((item for item in FORM_SECTIONS if item["name"] == section_name), None)
    if section is None:
        section = {"name": section_name, "fields": []}
        FORM_SECTIONS.append(section)
    section["fields"].append(field)


def _get_predictor() -> UTIPredictor:
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = UTIPredictor()
    return PREDICTOR


def _load_feature_names() -> list[str]:
    if FEATURES_PATH.exists():
        with open(FEATURES_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return [field["name"] for field in FIELD_SPECS]


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _save_upload(file_storage) -> str:
    original_name = secure_filename(file_storage.filename or "image")
    suffix = Path(original_name).suffix.lower()
    filename = f"{uuid.uuid4().hex}{suffix}"
    target = UPLOAD_DIR / filename
    file_storage.save(target)
    return filename


def _build_form_values(form_data: dict[str, str] | None = None) -> dict[str, str]:
    form_data = form_data or {}
    values: dict[str, str] = {}
    for field in FIELD_SPECS:
        values[field["name"]] = form_data.get(field["name"], field["default"])
    return values


def _validate_form(form_values: dict[str, str]) -> list[str]:
    errors: list[str] = []

    for field in FIELD_SPECS:
        value = form_values.get(field["name"], "")
        if value == "":
            errors.append(f"{field['label']} is required.")

    numeric_fields = {"urine_ph", "specific_gravity", "age", "Temperature", "RBC", "WBC"}
    for field_name in numeric_fields:
        value = form_values.get(field_name, "")
        try:
            float(value)
        except (TypeError, ValueError):
            errors.append(f"{FIELD_MAP[field_name]['label']} must be a valid number.")

    return errors


def _encode_form_value(name: str, value: str) -> int | float:
    if name in NUMERIC_COLS:
        return float(value)

    if name in BINARY_COLS:
        return 1 if str(value).strip() in {"1", "yes", "Yes", "true", "True"} else 0

    mapping = ORDINAL_ENCODINGS.get(name)
    if mapping is not None:
        normalized = str(value).strip()
        if name in {"gender", "Calculus_of_urinary_tract", "Urinary_tract_infections"}:
            normalized = normalized[:1].upper() + normalized[1:] if normalized else normalized
        if normalized in mapping:
            return mapping[normalized]

        lowered = str(value).strip().lower()
        if lowered in mapping:
            return mapping[lowered]

        raise ValueError(f"Unsupported value '{value}' for {name}")

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported numeric value '{value}' for {name}") from exc


def _prepare_clinical_payload(form_values: dict[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in FIELD_SPECS:
        name = field["name"]
        value = form_values.get(name, field["default"])
        payload[name] = _encode_form_value(name, value)
    return payload


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_serializable(item) for item in value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(value, Number):
        return float(value) if isinstance(value, float) else int(value)
    return value


def _format_prediction(result: dict[str, Any]) -> dict[str, Any]:
    normalized = {str(key): _to_serializable(value) for key, value in result.items()}

    prediction_label = None
    for key in ("prediction_label", "predicted_label", "label", "class_label", "prediction_class"):
        if normalized.get(key) not in (None, ""):
            prediction_label = str(normalized[key])
            break

    if prediction_label is None and normalized.get("prediction") not in (None, ""):
        prediction_value = normalized["prediction"]
        if isinstance(prediction_value, str):
            prediction_label = prediction_value
        else:
            try:
                numeric_prediction = int(float(prediction_value))
                prediction_label = "Infected" if numeric_prediction == 1 else "Not infected"
            except (TypeError, ValueError):
                prediction_label = str(prediction_value)

    if prediction_label is None:
        prediction_label = "Prediction unavailable"

    confidence = None
    for key in (
        "confidence",
        "probability",
        "proba",
        "score",
        "risk_score",
        "fusion_probability",
        "clinical_probability",
        "image_probability",
    ):
        value = normalized.get(key)
        if isinstance(value, (int, float)):
            confidence = float(value)
            break

    if confidence is None and isinstance(normalized.get("probability"), (int, float)):
        confidence = float(normalized["probability"])

    return {
        "prediction_label": prediction_label,
        "confidence": confidence,
        "confidence_pct": f"{confidence * 100:.1f}%" if isinstance(confidence, (int, float)) else None,
        "raw": normalized,
    }


@app.context_processor
def inject_globals():
    return {
        "app_name": "AI UTI Prediction",
        "feature_names": _load_feature_names(),
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        form_sections=FORM_SECTIONS,
        form_values=_build_form_values(),
        errors=[],
        result=None,
        result_summary=None,
        uploaded_image_url=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    form_values = _build_form_values(request.form.to_dict(flat=True))
    errors = _validate_form(form_values)

    image_file = request.files.get("image")
    if image_file is None or image_file.filename == "":
        errors.append("Ultrasound image is required.")
    elif not _allowed_file(image_file.filename):
        errors.append("Unsupported image format. Please upload PNG, JPG, JPEG, BMP, WEBP, TIF, or TIFF.")

    if errors:
        for message in errors:
            flash(message, "error")
        return render_template(
            "index.html",
            form_sections=FORM_SECTIONS,
            form_values=form_values,
            errors=errors,
            result=None,
            result_summary=None,
            uploaded_image_url=None,
        ), 400

    filename = _save_upload(image_file)
    image_path = str(UPLOAD_DIR / filename)

    try:
        predictor = _get_predictor()
        clinical_payload = _prepare_clinical_payload(form_values)
        raw_result = predictor.predict(clinical_payload, image_path)
        if not isinstance(raw_result, dict):
            raw_result = {"prediction": raw_result}
        result_summary = _format_prediction(raw_result)
    except Exception as exc:  # pragma: no cover - runtime safety
        error_message = f"Prediction failed: {exc}"
        flash(error_message, "error")
        return render_template(
            "index.html",
            form_sections=FORM_SECTIONS,
            form_values=form_values,
            errors=[error_message],
            result=None,
            result_summary=None,
            uploaded_image_url=url_for("uploaded_file", filename=filename),
        ), 500

    return render_template(
        "index.html",
        form_sections=FORM_SECTIONS,
        form_values=form_values,
        errors=[],
        result=raw_result,
        result_summary=result_summary,
        uploaded_image_url=url_for("uploaded_file", filename=filename),
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.ico", mimetype="image/svg+xml")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
