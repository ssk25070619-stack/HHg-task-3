import os
import sys
import json
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import VerificationPipeline
from src.blockchain_verifier import BlockchainVerifier

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('output', exist_ok=True)

pipeline = VerificationPipeline()
blockchain_verifier = BlockchainVerifier()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/samples')
def get_samples():
    assets_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets')
    samples = []
    if os.path.exists(assets_dir):
        for f in os.listdir(assets_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                label = 'Elon Musk' if 'test.jpg' in f else ('Custom Face' if 'test1.jpg' in f else 'Sample Face')
                samples.append({'name': f, 'path': f'/assets/{f}', 'label': label})
    return jsonify({'samples': samples})

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)

@app.route('/uiassets/<path:filename>')
def serve_uiassets(filename):
    return send_from_directory('uiassets', filename)

@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory('output', filename)

@app.route('/api/process', methods=['POST'])
def process_image():
    image_path = None
    if 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_name = f'{timestamp}_{filename}'
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
        file.save(image_path)
    elif request.is_json and request.json.get('sample'):
        image_path = os.path.join('assets', request.json.get('sample'))
    elif request.form.get('sample'):
        image_path = os.path.join('assets', request.form.get('sample'))

    if not image_path or not os.path.exists(image_path):
        return jsonify({'success': False, 'error': 'No valid image provided or file not found.'}), 400

    gemini_key = request.form.get('gemini_api_key') or (request.json.get('gemini_api_key') if request.is_json else None) or request.headers.get('X-Gemini-Key') or os.getenv('GEMINI_API_KEY')
    serpapi_key = request.form.get('serpapi_key') or (request.json.get('serpapi_key') if request.is_json else None) or os.getenv('SERPAPI_KEY')
    google_vision_key = request.form.get('google_vision_key') or (request.json.get('google_vision_key') if request.is_json else None) or os.getenv('GOOGLE_VISION_API_KEY')

    try:
        result = pipeline.run(
            image_path=image_path,
            output_dir='output',
            gemini_api_key=gemini_key,
            serpapi_key=serpapi_key,
            google_vision_key=google_vision_key
        )
        if 'stage1' in result and 'encoding' in result['stage1']:
            enc = result['stage1']['encoding']
            if hasattr(enc, 'tolist'):
                result['stage1']['encoding_preview'] = [round(float(x), 4) for x in enc[:12].tolist()]
                del result['stage1']['encoding']

        with open(image_path, 'rb') as img_f:
            result['input_image_b64'] = 'data:image/jpeg;base64,' + base64.b64encode(img_f.read()).decode('utf-8')

        crop_file = os.path.join('output', 'cropped_face.jpg')
        if os.path.exists(crop_file):
            with open(crop_file, 'rb') as crop_f:
                result['cropped_image_b64'] = 'data:image/jpeg;base64,' + base64.b64encode(crop_f.read()).decode('utf-8')

        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/verify', methods=['POST'])
def verify_record():
    data = request.json or {}
    tx_hash = data.get('tx_hash')
    payload = data.get('payload')
    if not tx_hash:
        return jsonify({'success': False, 'error': 'Transaction hash is required.'}), 400
    try:
        res = blockchain_verifier.verify_record(payload or {}, tx_hash)
        return jsonify(res)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5001)))
    args, _ = parser.parse_known_args()
    port = args.port
    print(f'\n[HH GOA 2026] UI Server running at http://127.0.0.1:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
