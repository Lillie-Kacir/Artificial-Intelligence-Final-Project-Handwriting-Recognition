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
<html>
<head>
    <title>Handwriting Recognition</title>
    <style>
        body { font-family: Arial; text-align: center; padding: 50px; background: #f0f0f0; }
        .container { background: white; border-radius: 10px; padding: 40px; max-width: 500px; margin: auto; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        button { background: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin-top: 20px; }
        button:hover { background: #0056b3; }
        .loading { display: none; margin-top: 20px; color: #666; }
        .error { color: red; margin-top: 20px; }
        .success { color: green; margin-top: 20px; }
        input { margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>✍️ Handwriting Recognition</h1>
        <p>Upload a photo of handwritten text and get digital text back</p>
        
        <form id="uploadForm">
            <input type="file" id="fileInput" accept=".jpg,.jpeg,.png" required>
            <br>
            <button type="submit">Process Image</button>
        </form>
        
        <div class="loading" id="loading">
            ⏳ Processing your image. This may take a moment...
        </div>
        <div id="result"></div>
    </div>

    <script>
        const form = document.getElementById('uploadForm');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const file = fileInput.files[0];
            if (!file) {
                result.innerHTML = '<div class="error">Please select a file first</div>';
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
                
                // Get the file blob from the response
                const blob = await response.blob();
                
                // Create download link
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'predicted_words.txt';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                result.innerHTML = '<div class="success">✅ Success! File downloaded.</div>';
                
            } catch (error) {
                result.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
            } finally {
                loading.style.display = 'none';
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
    
    # Check if handwriting_pipeline.py exists
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
    print("Handwriting Recognition Web App")
    print("Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
