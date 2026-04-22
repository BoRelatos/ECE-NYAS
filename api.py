# Unified API Server for ECENYAS Project (TFLITE READY, GUNICORN DEPLOYMENT)

# Core Flask and Utilities
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import time
import traceback
import os
import base64
import json  # <--- Added json import
from pathlib import Path
import concurrent.futures 
from concurrent.futures import ThreadPoolExecutor

# Sign-to-Text Dependencies 
import tensorflow as tf
import numpy as np
from deepface import DeepFace
import cv2
import re
import string

# 1. FLASK APP INITIALIZATION & CONCURRENCY

app = Flask(__name__)
CORS(app) 

# OPTIMIZATION: Increased ThreadPoolExecutor size for parallel model loading and higher concurrency in inference
executor = ThreadPoolExecutor(max_workers=max(32, os.cpu_count() * 2)) 

# 2. MODEL LOADING & ACCELERATION SETUP (TFLITE INTEGRATION)

# GLOBAL ACCELERATION SETTINGS
tf.config.optimizer.set_jit(True) 

# Sign-to-Text (TensorFlow Lite) Helper
def safe_load_tflite(path):
    """Loads a TFLite model and returns the configured interpreter."""
    print(f"[PARALLEL] Loading TFLite model: {path}")
    try:
        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        expected_features = 126
        actual_features = input_details[0]['shape'][2] 
        
        if actual_features != expected_features:
             print(f"CRITICAL WARNING: Loaded model '{path}' expects {actual_features} features. It should be 126.")
        
        print(f"[PARALLEL] TFLite model loaded successfully from {path}")
        return interpreter, input_details[0]['index'], output_details[0]['index']
    except Exception as e:
        print(f"FATAL ERROR loading TFLite model from {path}: {e}")
        traceback.print_exc()
        return None, None, None

# Submit all TFLite model loading tasks to the executor
tflite_futures = {
    "ASL": executor.submit(safe_load_tflite, "asl_model.tflite"),
    "FSL": executor.submit(safe_load_tflite, "fsl_model.tflite")
}

# Process TFLite results
try:
    asl_interpreter, asl_input_index, asl_output_index = tflite_futures["ASL"].result()
    fsl_interpreter, fsl_input_index, fsl_output_index = tflite_futures["FSL"].result()
    print("Finished loading Sign-to-Text models.")
except Exception as e:
    print(f"FATAL ERROR loading TFLite models: {e}")
    traceback.print_exc()


# 3. SINGLE SOURCE OF TRUTH: Load translations from database.json
# ==============================================================================
hardcoded_phrases = {}
raw_database = {} # Store raw data for the /api/get_signs_data endpoint

try:
    with open('database.json', 'r', encoding='utf-8') as f:
        raw_database = json.load(f)
        
    # Flatten the categorized data for the translation logic
    for category, items in raw_database.items():
        for item in items:
            # Create a combined entry
            # Note: We ensure regional translations are at the top level for easy access
            flat_entry = {
                "english": item.get('english', ''),
                "filipino": item.get('filipino', ''),
                "asl_gif": item.get('asl_gif', ''),
                "fsl_gif": item.get('fsl_gif', ''),
                **item.get('translations', {}) 
            }
            
            # Map Keys: English, Filipino, and regional languages
            # This allows the system to find the entry regardless of which language is detected
            keys_to_map = [flat_entry['english'], flat_entry['filipino']]
            # Add regional values to lookup keys if you want reverse lookup for them too
            keys_to_map.extend(item.get('translations', {}).values())

            for k in keys_to_map:
                if k:
                    hardcoded_phrases[k.lower().strip()] = flat_entry

    print(f"Successfully loaded {len(hardcoded_phrases)} phrases from database.json")

except Exception as e:
    print(f"WARNING: Could not load database.json. Hardcoded phrases will be empty. Error: {e}")
    # Fallback to empty if file missing, or define a small default set here if needed
    hardcoded_phrases = {}


# 4. TRANSLATION & UTILITY ROUTES
# ==============================================================================

def normalize_text(text):
    """Removes punctuation, converts text to lowercase, and normalizes spacing."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[' + re.escape(string.punctuation) + r']', '', text)
    text = ' '.join(text.split())
    return text

@app.route('/api/get_signs_data', methods=['GET'])
def get_signs_data():
    """Sends the FULL categorized database to the frontend for the Explore page."""
    return jsonify(raw_database)

@app.route('/translate_text', methods=['POST'])
def translate_text():
    start_time = time.time()
    try:
        data = request.get_json()
        input_text = data.get('text', '').strip()
        if not input_text:
            return jsonify({"error": "No text provided.", "time": "0.00s"}), 400

        # Normalize input for robust checking
        normalized_input = normalize_text(input_text)

        # Check loaded phrases using normalized input
        match = hardcoded_phrases.get(normalized_input)
        
        if match:
            response_data = match.copy()

            # Attempt to preserve the capitalization of the input for the matching language field
            found_lang = None
            # List of keys to check against input for proper case display
            all_langs = ["english", "filipino", "bicolano", "cebuano", "ilocano", "hiligaynon", "kapampangan", "waray"]
            
            for lang in all_langs:
                if lang in match and normalize_text(match[lang]) == normalized_input:
                    found_lang = lang
                    break

            if found_lang:
                response_data[found_lang] = input_text.capitalize()

        else:
            # Fallback if not found in JSON
            response_data = {
                "english": input_text.capitalize(),
                "filipino": f"Hindi matagpuan: '{normalized_input}'",
                "bicolano": "N/A", "cebuano": "N/A", "ilocano": "N/A",
                "hiligaynon": "N/A", "kapampangan": "N/A", "waray": "N/A",
                "asl_gif": "/static/ASL_GIFS/default/default.gif",
                "fsl_gif": "/static/FSL_GIFS/default/default.gif"
            }

        # Time taken
        end_time = time.time()
        response_data['time'] = f"{(end_time - start_time):.2f}s"
        return jsonify(response_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Translation request failed: {str(e)}", "time": "N/A"}), 500

# 5. FLASK PAGE ROUTES
# ==============================================================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/sign-to-text")
def sign_to_text_page():
    return render_template("sign-to-text.html")

@app.route("/speech-to-sign")
def speech_to_sign_page():
    return render_template("speech-to-sign.html")

@app.route("/explore")
def explore_page():
    return render_template("explore.html")

@app.route("/about-us")
def about_us_page():
    return render_template("about-us.html")

# Health Check & Debug Routes 
@app.route("/debug_info")
def debug_info():
    return jsonify({
        "ASL_Loaded": bool(asl_interpreter),
        "FSL_Loaded": bool(fsl_interpreter),
        "Expected_SL_Features": 126,
        "Speech_Models_Loaded": list(speech_models.keys()),
        "Speech_Device": str(device),
        "Database_Size": len(hardcoded_phrases)
    })

@app.route("/health")
def health():
    return "ok", 200

@app.route("/translate", methods=["POST"])
def translate():
    start_time = time.time()
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    target = (data.get("target") or "").strip().lower()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Note: If you want 'database.json' to override the AI models, check 'hardcoded_phrases' here first.
    
    result = {"english": text}
    lang_map = {
        "english": "eng_Latn", "cebuano": "ceb_Latn", "filipino": "fil_Latn",
        "ilokano": "ilo_Latn", "waray": "war_Latn", "bicolano" : "bcl_Latn", "kapampangan":"pam_Latn",
        "hiligaynon": "hil_Latn",
    }
    
    langs_to_generate = [target] if target in lang_map else list(lang_map.keys())
    
    futures = {}
    for lang_name in langs_to_generate:
        lang_code = lang_map.get(lang_name)
        key = lang_name[:3]
        
        if lang_code and key in speech_models:
            future = executor.submit(perform_translation_inference, key, text, lang_code)
            futures[lang_name] = future
        else:
             result[lang_name] = f"[Model for '{lang_name}' not loaded/supported]"

    for lang_name, future in futures.items():
        try:
            result[lang_name] = future.result(timeout=10) # 10s timeout
        except Exception as e:
            result[lang_name] = f"[Error generating {lang_name}: {str(e)}]"

    result["processing_time"] = round(time.time() - start_time, 2)
    return jsonify(result)


# 7. SIGN-TO-TEXT (TFLITE INFERENCE)
# ==============================================================================
def perform_sign_inference(interpreter, input_index, output_index, seq_in):
    """Worker function for TFLite model prediction."""
    interpreter.set_tensor(input_index, seq_in)
    interpreter.invoke()
    raw_pred = interpreter.get_tensor(output_index)
    return raw_pred[0] 

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    t0 = time.time()
    
    if not asl_interpreter or not fsl_interpreter:
        return jsonify({"error": "TFLite interpreters not loaded"}), 500

    data = request.get_json() or {}
    sequence = data.get("sequence")
    selected_lang = (data.get("language") or "").upper()

    if not sequence or selected_lang not in ("ASL", "FSL"):
       return jsonify({"error": "Missing or invalid 'sequence' or 'language'"}), 400

    try:
        seq = np.array(sequence, dtype=np.float32)
        expected_feature_size = 126
        
        if seq.ndim != 2 or seq.shape[1] != expected_feature_size:
            return jsonify({
                "error": "Bad sequence shape or feature size. Expected (N, 126).",
                "received_shape": seq.shape,
                "expected_features": expected_feature_size
            }), 400
            
        seq_in = np.expand_dims(seq, axis=0)

    except Exception as e:
        return jsonify({"error": "Invalid sequence format", "detail": str(e)}), 400

    if selected_lang == "ASL":
        interpreter, input_index, output_index = asl_interpreter, asl_input_index, asl_output_index
        actions = ASL_ACTIONS
    else:
        interpreter, input_index, output_index = fsl_interpreter, fsl_input_index, fsl_output_index
        actions = FSL_ACTIONS

    base_threshold = 0.90

    try:
        # Inference is submitted to the executor
        future = executor.submit(perform_sign_inference, interpreter, input_index, output_index, seq_in)
        raw_pred = future.result(timeout=5)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Model prediction failed or timed out", "detail": str(e)}), 500

    avg_pred = raw_pred
    confidence = float(np.max(avg_pred))
    pred_index = int(np.argmax(avg_pred))
    predicted_sign = str(actions[pred_index]) if confidence > base_threshold and pred_index < len(actions) else "No confident prediction"

    now = time.time()
    cooldown = compute_cooldown(confidence)

    if predicted_sign != "No confident prediction" and \
       predicted_sign == last_prediction_label[selected_lang] and \
       (now - last_prediction_time[selected_lang]) < cooldown:
        predicted_sign = "No confident prediction"
        confidence = 0.0
    elif predicted_sign != "No confident prediction":
        last_prediction_label[selected_lang] = predicted_sign
        last_prediction_time[selected_lang] = now

    top_k = np.argsort(avg_pred)[-3:][::-1]
    top_preds = [{"label": str(actions[i]) if i < len(actions) else "Unknown", "confidence": float(avg_pred[i])} for i in top_k]

    return jsonify({
        "sign": predicted_sign, "confidence": confidence, "top_preds": top_preds,
        "prediction_time": time.time() - t0, "detected_lang": selected_lang
    })

# --- Placeholder Actions Arrays (UPDATE THESE WITH YOUR FINAL SORTED LISTS) ---
ASL_ACTIONS = np.array([
    'A', 'Again', 'Alright', 'April', 'Are you okay', 'August', 'Aunt', 'B', 'Baby', 'Boy', 'Brother', 'Bye', 'C', 'Child', 'Children', 'Congratulations', 'Cousin', 'D', 'Daughter', 'Deaf', 'December', 'Do you understand', 'E', 'F', 'Family', 'Father', 'February', 'Friday', 'G', 'Girl', 'Good afternoon', 'Good evening', 'Good morning', 'Grandchild', 'Grandfather', 'Grandmother', 'H', 'Happy Anniversary', 'Happy New Year', 'Happy birthday', 'Happy to meet you', 'Hard of hearing', 'Hearing', 'Hello', 'How', 'How are you', 'How many', 'How old are you', 'Husband', 'I', 'I am feeling bad', 'I am fine', 'I am from', 'I am learning sign language', 'I am okay', 'I like you', 'I live in', 'I love you', 'J', 'January', 'July', 'June', 'K', 'L', 'M', 'March', 'May', 'Maybe', 'Merry Christmas', 'Monday', 'Mother', 'My name is', 'N', 'Nephew', 'Niece', 'No', 'No problem', 'November', 'O', 'October', 'P', 'Parents', 'Philippines', 'Please', 'Q', 'R', 'S', 'Saturday', 'See you', 'September', 'Sister', 'Son', 'Sorry', 'Sunday', 'T', 'Take care', 'Teenager', 'Thank you', 'Thursday', 'Tuesday', 'U', 'Uncle', 'V', 'W', 'Wednesday', 'What', 'What are you doing', 'What happened', 'What is your name', 'When', 'Where', 'Where are you from', 'Where are you going', 'Where do you live', 'Which', 'Who', 'Why', 'Wife', 'X', 'Y', 'Yes', 'You are welcome', 'Z', 'bad', 'bathe', 'beautiful', 'big', 'bird', 'bitter', 'black', 'bland', 'blue', 'brave', 'bread', 'breakfast', 'brown', 'buy', 'call', 'cat', 'chicken', 'church', 'city', 'clean', 'clinic', 'cold', 'color', 'cook', 'correct', 'cow', 'coward', 'cry', 'dance', 'delicious', 'difficult', 'dinner', 'dirty', 'do', 'do not', 'do not know', 'do not like', 'dog', 'down', 'drink', 'duck', 'early', 'easy', 'eat', 'egg', 'eight', 'eighteen', 'eighty', 'eleven', 'far', 'fast', 'feel', 'few', 'fifteen', 'fifty', 'fish', 'five', 'food', 'four', 'fourteen', 'fourty', 'fruits', 'get', 'give', 'go', 'good', 'gray', 'green', 'healthy', 'heavy', 'help', 'hold', 'horse', 'hospital', 'hot', 'house', 'hungry', 'juice', 'know', 'late', 'learn', 'left', 'light', 'like', 'listen', 'long', 'look', 'lunch', 'many', 'market', 'me', 'meat', 'milk', 'mine', 'monkey', 'near', 'new', 'nine', 'nineteen', 'ninety', 'noisy', 'old', 'one', 'one hundred', 'orange', 'our', 'park', 'pharmacy', 'pig', 'pink', 'play', 'purple', 'quiet', 'rainbow', 'read', 'red', 'rest', 'restaurant', 'rice', 'right', 'run', 'school', 'seven', 'seventeen', 'seventy', 'short', 'short in height', 'shy', 'sick', 'sign', 'sing', 'six', 'sixteen', 'sixty', 'sleep', 'sleepy', 'slow', 'small', 'smell', 'stop', 'strong', 'study', 'sweet', 'talk', 'tall', 'teach', 'tell', 'ten', 'their', 'they', 'thick', 'thin', 'think', 'thirsty', 'thirteen', 'thirty', 'three', 'tired', 'twelve', 'twenty', 'two', 'understand', 'up', 'use', 'vegetables', 'wait', 'walk', 'wash', 'water', 'we', 'weak', 'white', 'write', 'wrong', 'yellow', 'you', 'yours'

])
FSL_ACTIONS = np.array([
    'A', 'Abril', 'Agusto', 'Alin', 'Anak na Babae', 'Anak na Lalaki', 'Ang pangalan ko', 'Ano', 'Ano ang pangalan mo', 'Anong ginagawa mo', 'Anong nangyari', 'Apo', 'Asawang Babae', 'Asawang Lalaki', 'Ate', 'Ayos', 'Ayos ka lang', 'Ayos lang ako', 'B', 'Babae', 'Bakit', 'Bata', 'Biyernes', 'C', 'D', 'Deaf', 'Disyembre', 'E', 'Enero', 'F', 'G', 'Gusto kita', 'H', 'Hard of Hearing', 'Hearing', 'Hello', 'Hindi', 'Hulyo', 'Hunyo', 'Huwebes', 'I', 'Ilan', 'Ilang taon kana', 'Ingat', 'J', 'K', 'Kailan', 'Kamusta ka', 'Kita tayo', 'Kuya', 'L', 'Lalaki', 'Linggo', 'Lola', 'Lolo', 'Lunes', 'M', 'Mabuti naman ako', 'Magandang gabi', 'Magandang hapon', 'Magandang umaga', 'Magkano', 'Magulang', 'Mahal kita', 'Maligayang Anibersaryo', 'Maligayang Bagong Taon', 'Maligayang Kaarawan', 'Maligayang Pasko', 'Marso', 'Martes', 'Masama ang pakiramdam ko', 'Masayang makilala ka', 'Mayo', 'Mga Bata', 'Miyerkules', 'N', 'Nag aaral ako ng sign language', 'Nagmula ako', 'Naiintindihan mo ba', 'Nakatira ako sa', 'Nanay', 'Nobyembre', 'O', 'Oktubre', 'Oo', 'P', 'Paalam', 'Paano', 'Pagbati', 'Pakiulit', 'Pakiusap', 'Pamangkin na Babae', 'Pamangkin na Lalaki', 'Pamilya', 'Paumanhin', 'Pebrero', 'Pilipinas', 'Pinsan', 'Q', 'R', 'S', 'Saan', 'Saan ka nagmula', 'Saan ka nakatira', 'Saan ka papunta', 'Sabado', 'Salamat', 'Sanggol', 'Setyembre', 'Siguro', 'Sino', 'T', 'Tatay', 'Tinedyer', 'Tita', 'Tito', 'U', 'V', 'W', 'Walang anuman', 'Walang problema', 'X', 'Y', 'Z', 'abo', 'akin', 'ako', 'alam', 'almusal', 'amin', 'amoy', 'anim', 'animnapu', 'antok', 'apat', 'apatnapu', 'aral', 'aso', 'asul', 'baba', 'baboy', 'bago', 'bahaghari', 'bahay', 'baka', 'basa', 'berde', 'bibe', 'bigay', 'bili', 'botika', 'dalawa', 'dalawampu', 'dilaw', 'duwag', 'gamitin', 'gatas', 'gawin', 'gulay', 'gutom', 'hapunan', 'hindi alam', 'hindi nais', 'hintay', 'hinto', 'hiya', 'hugas', 'huli', 'huwag', 'ibon', 'ikaw', 'inom', 'isa', 'isang daan', 'isda', 'isip', 'itim', 'itlog', 'iyak', 'iyo', 'juice', 'kabayo', 'kahel', 'kain', 'kakaunti', 'kaliwa', 'kami', 'kanan', 'kanila', 'kanin', 'kanta', 'karne', 'kayumanggi', 'klinik', 'kuha', 'kulay', 'labing anim', 'labing apat', 'labing dalawa', 'labing isa', 'labing lima', 'labing pito', 'labing siyam', 'labing tatlo', 'labing walo', 'lakad', 'ligo', 'lima', 'limampu', 'luma', 'lungsod', 'luto', 'maaga', 'mabagal', 'mabigat', 'mabilis', 'mabuti', 'madali', 'magaan', 'maganda', 'maglaro', 'magusap', 'mahaba', 'mahina', 'mahirap', 'maikli', 'maingay', 'makapal', 'makinig', 'malakas', 'malaki', 'malapit', 'malayo', 'mali', 'maliit', 'malinis', 'malusog', 'manipis', 'manok', 'mapait', 'marami', 'marumi', 'masama', 'masarap', 'matamis', 'matangkad', 'matapang', 'matuto', 'may sakit', 'nais', 'ospital', 'paaralan', 'pagkain', 'pagod', 'pahinga', 'pakiramdam', 'parke', 'pito', 'pitumpu', 'prutas', 'pula', 'punta', 'pusa', 'puti', 'restawran', 'rosas', 'sabihin', 'sampu', 'sayaw', 'senyas', 'sila', 'simbahan', 'siyam', 'siyamnapu', 'sulat', 'taas', 'tahimik', 'takbo', 'tama', 'tanghalian', 'tatlo', 'tatlumpu', 'tawa', 'tawag', 'tignan', 'tinapay', 'tindahan', 'tubig', 'tulog', 'tulong', 'turo', 'ube', 'uhaw', 'unawa', 'unggoy', 'walang lasa', 'walo', 'walumpu'
])



last_prediction_time = {"ASL": 0.0, "FSL": 0.0}
last_prediction_label = {"ASL": None, "FSL": None}

def compute_cooldown(confidence: float) -> float:
    if confidence >= 0.90: return 3.0
    if confidence >= 0.80: return 2.0
    if confidence >= 0.70: return 1.5
    return 0.8

# 8. EMOTION DETECTION (DEEPFACE INFERENCE)
# ==============================================================================
def perform_emotion_inference(img):
    """Worker function for emotion analysis. This part is inherently slow (DeepFace)."""
    return DeepFace.analyze(
        img,
        actions=['emotion'],
        enforce_detection=False,
        # OPTIMIZATION NOTE: 'opencv' is one of the fastest available backends.
        detector_backend='opencv'
    )

@app.route("/emotion", methods=["POST"])
def emotion_endpoint():
    data = request.get_json() or {}
    if "image" not in data:
        return jsonify({"error": "No image data"}), 400

    try:
        img_data = base64.b64decode(data["image"].split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Inference is submitted to the executor
        future = executor.submit(perform_emotion_inference, img)
        emotions_result = future.result(timeout=10)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"emotion": "none", "error": f"DeepFace analysis failed or timed out: {str(e)}"})

    if not emotions_result:
        return jsonify({"emotion": "neutral"})

    all_emotions = emotions_result[0]['emotion']
    all_emotions.pop('fear', None)
    all_emotions.pop('disgust', None)

    if not all_emotions:
        return jsonify({"emotion": "neutral"})

    dominant_emotion = max(all_emotions, key=all_emotions.get)
    return jsonify({"emotion": dominant_emotion})

from flask import Flask

app = Flask(__name__)

# your routes here...

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)