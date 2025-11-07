import os
import json
import shutil
from typing import List, Dict, Any, Tuple

from flask import Flask, jsonify, request, render_template, send_from_directory

# Ensure project imports work regardless of where the app is launched from
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_model  # type: ignore
from src.pipeline.predict_pipeline import PredictPipeline  # type: ignore


ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DEFAULT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
DEFAULT_PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
FALLBACK_MODEL_PATH = os.path.join(PROJECT_ROOT, "notebook", "data", "best_model.pkl")
FALLBACK_PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "notebook", "data", "scaler.pkl")


def ensure_model_artifact():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    # If the primary model is missing but a fallback exists, copy it
    if not os.path.exists(DEFAULT_MODEL_PATH) and os.path.exists(FALLBACK_MODEL_PATH):
        shutil.copyfile(FALLBACK_MODEL_PATH, DEFAULT_MODEL_PATH)
    # Ensure the matching preprocessor exists as well. If missing but a fallback scaler exists,
    # copy it so that the model and preprocessor remain in sync.
    if not os.path.exists(DEFAULT_PREPROCESSOR_PATH) and os.path.exists(FALLBACK_PREPROCESSOR_PATH):
        shutil.copyfile(FALLBACK_PREPROCESSOR_PATH, DEFAULT_PREPROCESSOR_PATH)


def validate_and_sync_artifacts():
    """Ensure model and preprocessor expect the same number of features.

    If they mismatch and a fallback preprocessor exists, replace the current
    preprocessor with the fallback scaler (trained alongside the fallback model).
    """
    try:
        model = load_model(filepath=DEFAULT_MODEL_PATH)
        pre = load_model(filepath=DEFAULT_PREPROCESSOR_PATH)

        # Determine feature counts
        try:
            model_n = int(getattr(model, "n_features_in_", None))
        except Exception:
            model_n = None

        try:
            pre_names = _names_in(pre)
            pre_n = len(pre_names) if pre_names else None
        except Exception:
            pre_n = None

        # If both known and mismatch, try to fix by swapping in fallback preprocessor
        if model_n is not None and pre_n is not None and model_n != pre_n:
            if os.path.exists(FALLBACK_PREPROCESSOR_PATH):
                shutil.copyfile(FALLBACK_PREPROCESSOR_PATH, DEFAULT_PREPROCESSOR_PATH)
            # Optionally, we could re-load to be safe
    except FileNotFoundError:
        # If either artifact is missing, ensure copy step handles it elsewhere
        return
    except Exception:
        # Do not fail app startup due to validation
        return


def _names_in(preprocessor) -> List[str]:
    names = getattr(preprocessor, "feature_names_in_", None)
    if names is None:
        return []
    try:
        return list(names)
    except Exception:
        return [str(n) for n in names]


def extract_feature_info(preprocessor) -> List[Dict[str, Any]]:
    # Always base on feature_names_in_ to match fit-time input exactly
    names = _names_in(preprocessor)

    # Step 1: Try to infer numeric membership via ColumnTransformer
    numeric_set = set()
    try:
        for t_name, _, cols in getattr(preprocessor, "transformers_", []):
            if str(t_name).lower().startswith("num"):
                if isinstance(cols, (list, tuple)):
                    for c in cols:
                        numeric_set.add(str(c))
                else:
                    numeric_set.add(str(cols))
    except Exception:
        pass

    # Step 2: If we couldn't determine numeric columns (e.g., scaler without transformers),
    # fall back to dtypes from artifacts/train.csv
    if not numeric_set:
        try:
            import pandas as pd
            train_csv_path = os.path.join(ARTIFACTS_DIR, "train.csv")
            if os.path.exists(train_csv_path):
                df = pd.read_csv(train_csv_path)
                for col in names:
                    if col in df.columns:
                        try:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                numeric_set.add(str(col))
                        except Exception:
                            continue
        except Exception:
            pass

    # Step 3: Heuristic fallback by name if still unknown
    if not numeric_set:
        likely_numeric_keywords = [
            "mean_motion", "inclination", "eccentricity", "ra_of_asc_node",
            "arg_of_pericenter", "mean_anomaly", "bstar", "dot", "ddot",
            "semimajor", "axis", "period", "apoapsis", "periapsis", "rev_at_epoch"
        ]
        for n in names:
            lower = str(n).lower()
            if any(k in lower for k in likely_numeric_keywords):
                numeric_set.add(str(n))

    feature_info: List[Dict[str, Any]] = []
    for n in names:
        ftype = "numeric" if str(n) in numeric_set else "categorical"
        feature_info.append({"name": str(n), "type": ftype})
    return feature_info


def coerce_by_types(payload: Dict[str, Any], feature_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    type_map = {f["name"]: f.get("type", "unknown") for f in feature_info}
    coerced: Dict[str, Any] = {}
    for k, v in payload.items():
        ftype = type_map.get(k, "unknown")
        if v is None:
            coerced[k] = None
            continue
        if isinstance(v, (int, float)):
            coerced[k] = v
            continue
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                coerced[k] = None
                continue
            if ftype == "numeric":
                try:
                    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                        coerced[k] = int(s)
                    else:
                        coerced[k] = float(s)
                except Exception:
                    coerced[k] = None
            else:
                coerced[k] = s
            continue
        coerced[k] = v
    return coerced


def create_app() -> Flask:
    ensure_model_artifact()
    validate_and_sync_artifacts()

    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/schema", methods=["GET"])
    def schema():
        try:
            pre = load_model(filepath=DEFAULT_PREPROCESSOR_PATH)
            feature_info = extract_feature_info(pre)
            return jsonify({
                "ok": True,
                "features": feature_info,
            })
        except FileNotFoundError:
            return jsonify({
                "ok": False,
                "error": "Preprocessor not found. Ensure artifacts/preprocessor.pkl exists."
            }), 500
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/example", methods=["GET"])
    def example_values():
        """Return representative example values aligned to preprocessor feature order.

        Strategy:
        - Load preprocessor to get feature_names_in_ (authoritative order and names)
        - Load artifacts/train.csv if available
        - For each expected feature:
            - If present in train.csv: use median (numeric) or mode (categorical)
            - Else: generate a reasonable synthetic default based on name heuristics
        """
        try:
            import pandas as pd

            pre = load_model(filepath=DEFAULT_PREPROCESSOR_PATH)
            feature_info = extract_feature_info(pre)
            expected_names = [f["name"] for f in feature_info]

            train_csv_path = os.path.join(ARTIFACTS_DIR, "train.csv")
            df = None
            if os.path.exists(train_csv_path):
                try:
                    df = pd.read_csv(train_csv_path)
                except Exception:
                    df = None

            def synth_value(name: str, ftype: str):
                n = name.lower()
                if ftype == "numeric":
                    import random
                    def rnd(a, b):
                        return round(random.uniform(a, b), 6)
                    if "inclination" in n:
                        return rnd(0, 180)
                    if "ra_of_asc_node" in n:
                        return rnd(0, 360)
                    if "arg_of_pericenter" in n:
                        return rnd(0, 360)
                    if "mean_anomaly" in n:
                        return rnd(0, 360)
                    if "eccentricity" in n:
                        return rnd(0, 0.99)
                    if "mean_motion_dot" in n:
                        return rnd(-1e-3, 1e-3)
                    if "mean_motion_ddot" in n:
                        return rnd(-1e-6, 1e-6)
                    if n == "bstar" or "bstar" in n:
                        return rnd(0, 0.05)
                    if "mean_motion" in n:
                        return rnd(0.5, 16)
                    if "semimajor" in n or ("semi" in n and "axis" in n):
                        return rnd(6500, 45000)
                    if "period" in n:
                        return rnd(80, 1500)
                    if "apoapsis" in n:
                        return rnd(100, 40000)
                    if "periapsis" in n:
                        return rnd(100, 38000)
                    return rnd(0, 1000)
                # categorical defaults by common domains
                if "time_system" in n:
                    return "UTC"
                if "ref_frame" in n:
                    return "TEME"
                if "center_name" in n:
                    return "EARTH"
                if "originator" in n:
                    return "18 SPCS"
                if "mean_element_theory" in n:
                    return "SGP4"
                if "object_type" in n:
                    return "DEBRIS"
                if "rcs_size" in n:
                    return "SMALL"
                if "country_code" in n:
                    return "US"
                if "classification_type" in n:
                    return "U"
                if "ephemeris_type" in n:
                    return "0"
                if "site" in n:
                    return "FRGUI"
                if "object_name" in n:
                    return "SL-8 DEB"
                if "object_id" in n:
                    return "2001-018A"
                if n.startswith("tle_line"):
                    return "1 25544U 98067A   20344.91667824  .00001264  00000-0  29621-4 0  9990"
                if n == "tle_line0":
                    return "0 EXAMPLE"
                return "UNKNOWN"

            # Prefer sampling a single random row for realism and variability
            example = {}
            sampled = None
            if df is not None:
                try:
                    # Limit to expected columns; missing columns handled later
                    sampled = df.sample(n=1, replace=False, random_state=None)
                except Exception:
                    sampled = None

            for f in feature_info:
                name = f.get("name")
                ftype = f.get("type", "unknown")
                val = None
                if sampled is not None and name in sampled.columns:
                    try:
                        cell = sampled.iloc[0][name]
                        val = None if pd.isna(cell) else cell
                    except Exception:
                        val = None
                if val is None or (hasattr(pd, "isna") and pd.isna(val)):
                    val = synth_value(name, ftype)
                example[name] = val

            return jsonify({
                "ok": True,
                "example": example,
                "features": expected_names,
            })
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "Artifacts missing. Train or provide artifacts."}), 500
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            data = request.get_json(force=True) or {}
            if not isinstance(data, dict):
                return jsonify({"ok": False, "error": "Invalid JSON payload."}), 400

            pre = load_model(filepath=DEFAULT_PREPROCESSOR_PATH)
            feature_info = extract_feature_info(pre)
            expected_names = [f["name"] for f in feature_info]

            # Coerce by declared types
            features = coerce_by_types(data, feature_info)

            # Align to expected columns exactly (order + drop extras + add missing)
            import pandas as pd
            df = pd.DataFrame([features])
            df = df.reindex(columns=expected_names)
            # Force numeric where possible; non-convertible strings become NaN and will
            # be handled by the preprocessor's imputers/scalers. This prevents errors like
            # "could not convert string to float: 'US'" during transform.
            try:
                df = df.apply(pd.to_numeric, errors="coerce")
            except Exception:
                pass

            pipeline = PredictPipeline()
            preds = pipeline.predict(df)

            # Best-effort probabilities if classifier supports it
            proba = None
            try:
                model = load_model(filepath=DEFAULT_MODEL_PATH)
                if hasattr(model, "predict_proba"):
                    proba_arr = model.predict_proba(pre.transform(df))  # type: ignore
                    if proba_arr is not None:
                        proba = proba_arr[0].tolist() if len(proba_arr) else None
            except Exception:
                proba = None

            # Normalize single-value predictions to scalar
            out_pred = None
            try:
                if hasattr(preds, "tolist"):
                    lst = preds.tolist()
                    out_pred = lst[0] if isinstance(lst, list) and len(lst) == 1 else lst
                else:
                    out_pred = preds
            except Exception:
                out_pred = str(preds)

            return jsonify({
                "ok": True,
                "prediction": out_pred,
                "probabilities": proba,
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route('/static/<path:filename>')
    def static_files(filename):
        return send_from_directory(app.static_folder, filename)

    return app


if __name__ == "__main__":
    flask_app = create_app()
    port = int(os.environ.get("PORT", "5000"))
    flask_app.run(host="0.0.0.0", port=port, debug=True)
