from flask import Flask, request, send_file, after_this_request
import subprocess
import os
import sys

app = Flask(__name__)

UPLOAD_FOLDER = './demo_assets'
OUTPUT_FOLDER = './outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

HTML_PAGE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cosmic Handwriting Recognizer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Poppins', 'Segoe UI', 'Arial', sans-serif;
            min-height: 100vh;
            background: linear-gradient(135deg, #0a0e2a 0%, #1a1f3a 50%, #0f1228 100%);
            position: relative;
            overflow-x: hidden;
        }

        /* Animated stars background */
        .stars {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 0;
        }

        .star {
            position: absolute;
            background: white;
            border-radius: 50%;
            animation: twinkle var(--duration) infinite ease-in-out;
            opacity: 0;
        }

        @keyframes twinkle {
            0%, 100% { opacity: 0; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.2); }
        }

        /* Shooting stars - FIXED VERSION */
        .shooting-star {
            position: fixed;
            width: 150px;
            height: 2px;
            background: linear-gradient(90deg, rgba(255,255,255,1) 0%, rgba(255,255,255,0) 100%);
            border-radius: 2px;
            animation: shoot 6s linear infinite;
            z-index: 5;
            pointer-events: none;
        }

        /* Different positions and delays for multiple shooting stars */
        .shooting-star:nth-child(1) { top: 10%; left: -150px; animation-delay: 0s; }
        .shooting-star:nth-child(2) { top: 30%; left: -150px; animation-delay: 3s; }
        .shooting-star:nth-child(3) { top: 50%; left: -150px; animation-delay: 6s; }
        .shooting-star:nth-child(4) { top: 70%; left: -150px; animation-delay: 2s; }
        .shooting-star:nth-child(5) { top: 20%; left: -150px; animation-delay: 5s; }
        .shooting-star:nth-child(6) { top: 80%; left: -150px; animation-delay: 8s; }

        @keyframes shoot {
            0% {
                transform: translateX(0) translateY(0) rotate(25deg);
                opacity: 1;
            }
            80% {
                opacity: 1;
            }
            100% {
                transform: translateX(calc(100vw + 200px)) translateY(-100px) rotate(25deg);
                opacity: 0;
            }
        }

        /* Nebula glow effect */
        .nebula {
            position: fixed;
            width: 600px;
            height: 600px;
            border-radius: 50%;
            filter: blur(100px);
            opacity: 0.3;
            pointer-events: none;
            z-index: 0;
        }

        .nebula-1 {
            background: #6c5ce7;
            top: -200px;
            left: -200px;
            animation: float 20s ease-in-out infinite;
        }

        .nebula-2 {
            background: #00cec9;
            bottom: -200px;
            right: -200px;
            animation: float 25s ease-in-out infinite reverse;
        }

        .nebula-3 {
            background: #fd79a8;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 400px;
            height: 400px;
            filter: blur(80px);
            opacity: 0.15;
        }

        @keyframes float {
            0%, 100% { transform: translate(0, 0); }
            50% { transform: translate(50px, 50px); }
        }

        /* Main container */
        .container {
            position: relative;
            z-index: 10;
            max-width: 600px;
            margin: 60px auto;
            padding: 20px;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            border-radius: 32px;
            padding: 50px 40px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 25px 45px rgba(0, 0, 0, 0.3), 0 0 40px rgba(108, 92, 231, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 35px 55px rgba(0, 0, 0, 0.4), 0 0 60px rgba(108, 92, 231, 0.3);
        }

        h1 {
            font-size: 3em;
            background: linear-gradient(135deg, #ffffff 0%, #a8a8ff 50%, #6c5ce7 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 15px;
            letter-spacing: 2px;
            text-shadow: 0 0 30px rgba(108, 92, 231, 0.5);
        }

        .subtitle {
            color: rgba(255, 255, 255, 0.7);
            margin-bottom: 40px;
            font-size: 1.1em;
        }

        /* File upload area */
        .upload-area {
            border: 2px dashed rgba(108, 92, 231, 0.5);
            border-radius: 24px;
            padding: 40px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.05);
            margin-bottom: 25px;
        }

        .upload-area:hover {
            border-color: #6c5ce7;
            background: rgba(108, 92, 231, 0.1);
            transform: scale(1.02);
        }

        .upload-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }

        .upload-text {
            color: rgba(255, 255, 255, 0.8);
            font-size: 1.1em;
        }

        .upload-hint {
            color: rgba(255, 255, 255, 0.5);
            font-size: 0.85em;
            margin-top: 10px;
        }

        input[type="file"] {
            display: none;
        }

        /* Button */
        .process-btn {
            background: linear-gradient(135deg, #6c5ce7 0%, #a8a8ff 100%);
            color: white;
            border: none;
            padding: 14px 40px;
            font-size: 1.1em;
            font-weight: 600;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 1px;
            position: relative;
            overflow: hidden;
        }

        .process-btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }

        .process-btn:hover::before {
            width: 300px;
            height: 300px;
        }

        .process-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(108, 92, 231, 0.4);
        }

        .process-btn:active {
            transform: translateY(0);
        }

        /* Loading spinner */
        .loading {
            display: none;
            margin-top: 30px;
            text-align: center;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 3px solid rgba(108, 92, 231, 0.3);
            border-top-color: #6c5ce7;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .loading-text {
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9em;
        }

        /* Result messages */
        .error, .success {
            margin-top: 20px;
            padding: 12px 20px;
            border-radius: 12px;
            font-weight: 500;
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .error {
            background: rgba(255, 99, 71, 0.2);
            border-left: 4px solid #ff6b6b;
            color: #ffa8a8;
        }

        .success {
            background: rgba(0, 255, 191, 0.1);
            border-left: 4px solid #00ffbf;
            color: #a8ffe0;
        }

        /* File info */
        .file-info {
            margin-top: 15px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            font-size: 0.85em;
            color: rgba(255, 255, 255, 0.6);
            display: none;
        }

        /* Footer */
        .footer {
            position: relative;
            z-index: 10;
            text-align: center;
            margin-top: 40px;
            color: rgba(255, 255, 255, 0.4);
            font-size: 0.8em;
        }
    </style>
</head>
<body>
    <!-- Stars background -->
    <div class="stars" id="stars"></div>
    <div class="nebula nebula-1"></div>
    <div class="nebula nebula-2"></div>
    <div class="nebula nebula-3"></div>
    <!-- Shooting stars - FIXED VERSION -->
<div class="shooting-star"></div>
<div class="shooting-star"></div>
<div class="shooting-star"></div>
<div class="shooting-star"></div>
<div class="shooting-star"></div>
<div class="shooting-star"></div>

    <div class="container">
        <div class="glass-card">
            <h1>✨ COSMIC HANDWRITING ✨</h1>
            <div class="subtitle">Transform your handwritten notes into digital text</div>
            
            <form id="uploadForm">
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📝</div>
                    <div class="upload-text">Click or drag to upload</div>
                    <div class="upload-hint">JPG, PNG, or JPEG (max 10MB)</div>
                </div>
                <input type="file" id="fileInput" accept=".jpg,.jpeg,.png" required>
                <div class="file-info" id="fileInfo"></div>
                <button type="submit" class="process-btn">Process Image</button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div class="loading-text">Decoding handwriting from the cosmos...</div>
            </div>
            <div id="result"></div>
        </div>
        <div class="footer">
            ✨ Powered by AI & Deep Learning ✨
        </div>
    </div>

    <script>
        // Generate stars
        function createStars() {
            const starsContainer = document.getElementById('stars');
            for (let i = 0; i < 200; i++) {
                const star = document.createElement('div');
                star.className = 'star';
                const size = Math.random() * 3 + 1;
                star.style.width = size + 'px';
                star.style.height = size + 'px';
                star.style.left = Math.random() * 100 + '%';
                star.style.top = Math.random() * 100 + '%';
                star.style.setProperty('--duration', (Math.random() * 3 + 1) + 's');
                star.style.animationDelay = Math.random() * 5 + 's';
                starsContainer.appendChild(star);
            }
        }
        createStars();

        // File input handling
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                fileInfo.innerHTML = `📄 Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
                fileInfo.style.display = 'block';
            } else {
                fileInfo.style.display = 'none';
            }
        });

        // Form submission
        const form = document.getElementById('uploadForm');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const file = fileInput.files[0];
            if (!file) {
                result.innerHTML = '<div class="error">✨ Please select a file first ✨</div>';
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            loading.style.display = 'block';
            result.innerHTML = '';
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(errorText || 'Processing failed');
                }
                
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'predicted_words.txt';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                result.innerHTML = '<div class="success">✅ Success! Your text has been decoded and downloaded.</div>';
                
            } catch (error) {
                result.innerHTML = `<div class="error">❌ Cosmic error: ${error.message}</div>`;
            } finally {
                loading.style.display = 'none';
            }
        });

        // Drag and drop functionality
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#a8a8ff';
            uploadArea.style.background = 'rgba(108, 92, 231, 0.15)';
        });
        
        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'rgba(108, 92, 231, 0.5)';
            uploadArea.style.background = 'rgba(255, 255, 255, 0.05)';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'rgba(108, 92, 231, 0.5)';
            uploadArea.style.background = 'rgba(255, 255, 255, 0.05)';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                const event = new Event('change');
                fileInput.dispatchEvent(event);
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return HTML_PAGE

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    print(f"File saved: {filepath}")
    
    if not os.path.exists('handwriting_pipeline.py'):
        return f"Error: handwriting_pipeline.py not found", 500
    
    cmd = [
        sys.executable, 'handwriting_pipeline.py',
        '--image', filepath,
        '--output-stem', os.path.join(OUTPUT_FOLDER, 'predicted_words')
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(f"Return code: {result.returncode}")
    if result.stdout:
        print(f"STDOUT: {result.stdout}")
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    
    if result.returncode != 0:
        error_msg = f"Pipeline failed: {result.stderr}"
        print(error_msg)
        return error_msg, 500
    
    txt_path = os.path.join(OUTPUT_FOLDER, 'predicted_words.txt')
    if not os.path.exists(txt_path):
        return 'Output file not generated', 500
    
    @after_this_request
    def cleanup(response):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Cleaned up: {filepath}")
        except Exception as e:
            print(f'Cleanup error: {e}')
        return response
    
    return send_file(txt_path, as_attachment=True, download_name='predicted_words.txt')

if __name__ == '__main__':
    print("=" * 50)
    print("✨ COSMIC HANDWRITING RECOGNIZER ✨")
    print("Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
