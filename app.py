import os, json, time
import numpy as np
import tensorflow as tf
import keras
from PIL import Image, ImageFile
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess
from werkzeug.datastructures import FileStorage
from io import BytesIO

ImageFile.LOAD_TRUNCATED_IMAGES = True  # évite crash sur images incomplètes

# ---------- Config ----------
CLASSES_PATH = os.path.join("models", "classes.json")

# Registre des modèles (adapte les paths si besoin)
MODEL_REGISTRY = {
    # MobileNetV2 fine-tuning (SavedModel -> TFSMLayer en Keras 3)
    "mobilenet_ft": {
        "type": "tfsmlayer",
        "path": os.path.join("models", "mobilenet_v2_finetuned_savedmodel"),
        "img_size": 128,
        "preprocess": "mobilenet",
    },
    # Optionnel : tête gelée (non utilisée dans /predict_all pour éviter des soucis de graphes)
    "mobilenet_head": {
        "type": "keras",
        "path": os.path.join("models", "mobilenet_v2_best.h5"),
        "img_size": 128,
        "preprocess": "mobilenet",
    },
    # CNN from scratch (.h5)
    "cnn": {
        "type": "keras",
        "path": os.path.join("models", "fruit_model.h5"),
        "img_size": 100,
        "preprocess": "cnn",
    },
}

# ---------- App ----------
app = Flask(__name__)
CORS(app)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    idx_to_class = {int(k): v for k, v in json.load(f).items()}

def nice_label(raw_label: str) -> str:
    """Nettoie le label brut (ex: 'apple_red_yellow_1' -> 'Apple Red Yellow')."""
    lab = raw_label.replace("_", " ")
    lab = "".join(ch for ch in lab if not ch.isdigit())
    return lab.strip().title() if lab.strip() else raw_label

# ---------- Cache des runners (chargés une seule fois) ----------
RUNNER_CACHE = {}

def build_runner(entry):
    """Construit un runner(x)->np.ndarray pour un modèle donné et le met en cache."""
    key = (entry["type"], entry["path"])
    if key in RUNNER_CACHE:
        return RUNNER_CACHE[key]

    mtype = entry["type"]
    path = entry["path"]

    if mtype == "tfsmlayer":
        layer = keras.layers.TFSMLayer(path, call_endpoint="serving_default")
        def runner(x):
            y = layer(tf.convert_to_tensor(x))
            if isinstance(y, dict):
                y = next(iter(y.values()))
            elif isinstance(y, (list, tuple)):
                y = y[0]
            return y.numpy()
    elif mtype == "keras":
        mdl = keras.models.load_model(path, compile=False)
        def runner(x):
            return mdl.predict(x)
    else:
        raise ValueError(f"Type modèle non supporté: {mtype}")

    RUNNER_CACHE[key] = runner
    return runner

def get_runner(name: str):
    """Récupère (ou charge) le runner pour un modèle par son nom."""
    entry = MODEL_REGISTRY[name]
    return build_runner(entry)

# ---------- Sélection du modèle courant (lazy) ----------
CURRENT_NAME = None
CURRENT_CONF = None   # {'img_size':..., 'preprocess':...}

def set_active_model(name: str):
    """Sélectionne le modèle courant sans charger le poids tout de suite."""
    global CURRENT_NAME, CURRENT_CONF
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modèle inconnu: {name}")
    conf = MODEL_REGISTRY[name]
    CURRENT_NAME = name
    CURRENT_CONF = {"img_size": conf["img_size"], "preprocess": conf["preprocess"]}
    return name

# modèle par défaut (sélection seulement, chargement à la 1ère requête)
set_active_model("mobilenet_ft")

# ---------- Prétraitements ----------
def preprocess_generic(img, img_size: int, ptype: str):
    img = img.convert("RGB").resize((img_size, img_size))
    x = img_to_array(img)
    if ptype == "mobilenet":
        x = mob_preprocess(x)
    elif ptype == "cnn":
        x = x / 255.0
    else:
        raise ValueError(f"Prétraitement inconnu: {ptype}")
    return np.expand_dims(x, axis=0)

def preprocess_image(file_storage):
    """Prétraitement basé sur le modèle actif."""
    img = Image.open(file_storage.stream)
    return preprocess_generic(img, CURRENT_CONF["img_size"], CURRENT_CONF["preprocess"])

def preprocess_for_entry(file_storage, entry):
    """Prétraitement pour un modèle donné."""
    img = Image.open(file_storage.stream)
    return preprocess_generic(img, entry["img_size"], entry["preprocess"])

# ---------- Utils I/O ----------
def clone_filestorage_from_bytes(payload: bytes, filename: str, content_type: str | None):
    """Crée un FileStorage 'neuf' depuis des bytes (utile pour réutiliser la même image)."""
    return FileStorage(stream=BytesIO(payload), filename=filename, content_type=content_type)

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": CURRENT_NAME})

@app.get("/labels")
def labels():
    return jsonify(idx_to_class)

@app.get("/models")
def models():
    return jsonify({"available": list(MODEL_REGISTRY.keys()), "current": CURRENT_NAME})

@app.post("/set_model")
def set_model():
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    if not name:
        return jsonify({"error": "Missing field 'name'."}), 400
    try:
        t0 = time.time()
        set_active_model(name)   # sélectionne seulement (lazy)
        dt = int((time.time() - t0) * 1000)
        return jsonify({"ok": True, "current": CURRENT_NAME, "switched_in_ms": dt})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded under key 'file'."}), 400
    x = preprocess_image(request.files["file"])
    runner = get_runner(CURRENT_NAME)  # lazy load ici

    t0 = time.time()
    prob = runner(x)[0]
    lat_ms = int((time.time() - t0) * 1000)
    top_idx = int(np.argmax(prob))
    raw = idx_to_class.get(top_idx, str(top_idx))
    return jsonify({
        "model": CURRENT_NAME,
        "label": nice_label(raw),
        "raw_label": raw,
        "confidence": float(np.max(prob)),
        "latency_ms": lat_ms
    })

def run_one(file_storage, name, entry):
    """Exécute une prédiction pour un modèle donné."""
    runner = build_runner(entry)  # utilisera le cache
    x = preprocess_for_entry(file_storage, entry)
    t1 = time.time()
    prob = runner(x)[0]
    lat_ms = int((time.time() - t1) * 1000)
    idx = int(np.argmax(prob))
    raw = idx_to_class.get(idx, str(idx))
    return {
        "model": name,
        "label": nice_label(raw),
        "raw_label": raw,
        "confidence": float(np.max(prob)),
        "latency_ms": lat_ms
    }

# --------- Comparaison : MobileNet FT d'abord, puis CNN (pas de tri par confiance) ---------
@app.post("/predict_all")
def predict_all():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded under key 'file'."}), 400
    f = request.files["file"]

    payload = f.read()
    filename = f.filename
    content_type = f.content_type

    results = []
    t0 = time.time()

    # Ordre d'affichage voulu
    models_to_eval = ["mobilenet_ft", "cnn"]

    for name in models_to_eval:
        entry = MODEL_REGISTRY[name]
        try:
            pseudo = clone_filestorage_from_bytes(payload, filename, content_type)
            r = run_one(pseudo, name, entry)
            results.append(r)
        except Exception as e:
            results.append({"model": name, "error": str(e)})

    total_ms = int((time.time() - t0) * 1000)
    #  PAS DE TRI : on garde l'ordre ci-dessus (FT puis CNN)
    return jsonify({"results": results, "total_ms": total_ms})

# --------- Choix automatique : TOUJOURS préférer MobileNet FT en cas de désaccord ---------
# --------- Choix automatique : TOUJOURS préférer MobileNet FT en cas de désaccord ---------
CHOOSE_THRESH = 0.85  # si FT >= 85% on le choisit direct
MARGIN = 0.05         # (ne sert plus au désaccord ; conservé pour cas "labels identiques")

@app.post("/predict_best")
def predict_best():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded under key 'file'."}), 400
    f = request.files["file"]

    payload = f.read()
    filename = f.filename
    content_type = f.content_type

    def fs():
        return clone_filestorage_from_bytes(payload, filename, content_type)

    # Prédictions des deux modèles
    try:
        ft = run_one(fs(), "mobilenet_ft", MODEL_REGISTRY["mobilenet_ft"])
    except Exception as e:
        ft = {"model": "mobilenet_ft", "error": str(e), "confidence": 0.0}

    try:
        cnn = run_one(fs(), "cnn", MODEL_REGISTRY["cnn"])
    except Exception as e:
        cnn = {"model": "cnn", "error": str(e), "confidence": 0.0}

    # Règles de décision
    reason = ""
    if "error" not in ft and ft["confidence"] >= CHOOSE_THRESH:
        # 1) FT très confiant -> on le prend
        chosen = ft
        reason = f"MobileNet FT >= {int(CHOOSE_THRESH*100)}% de confiance."
    elif "error" not in ft and "error" not in cnn:
        if ft["label"] != cnn["label"]:
            # 2) DÉSACCORD -> on PRÉFÈRE FT quoi qu’il arrive (robustesse observée)
            chosen = ft
            reason = "Désaccord de labels — MobileNet FT préféré (robustesse supérieure observée)."
        else:
            # 3) Même label -> on garde le plus confiant (marge gardée à titre secondaire)
            chosen = ft if ft["confidence"] + MARGIN >= cnn["confidence"] else cnn
            reason = "Labels identiques — modèle à plus forte confiance choisi."
    else:
        # 4) Si un modèle en erreur -> on prend l'autre
        chosen = ft if "error" not in ft else cnn
        reason = "Un des modèles a rencontré une erreur — choix du modèle valide."

    if "error" not in chosen and chosen["confidence"] < 0.50:
        chosen["note"] = "Prédiction incertaine — essayez une image plus nette/centrée."

    return jsonify({
        "chosen": {**chosen, "reason": reason},
        "mobilenet_ft": ft,
        "cnn": cnn
    })



if __name__ == "__main__":
    # Si le port 8000 est occupé, change en 8010 et adapte l'URL dans index.html
    app.run(host="0.0.0.0", port=8000, debug=True)
