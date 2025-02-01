import cv2
import os
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file
import tensorflow as tf

unet = tf.keras.models.load_model("try/unet/best_model.h5")

# Initialize Flask app
app = Flask(__name__)

# Set up file upload configuration
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to process the incoming image for prediction
def process_frame(image):
    IMAGE_SIZE = 512  # Resize frame to 512x512
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    image = image.astype(np.float32) / 127.5 - 1  # Normalize image
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image_tensor = tf.convert_to_tensor(image)

    # Run prediction
    mask, values = unet.predict(np.expand_dims(image_tensor, axis=0), verbose=0)
    mask = np.squeeze(mask)
    mask_npy = np.argmax(mask, axis=2).astype(np.uint8)

    return mask_npy

# Route for the main UI
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read the uploaded image for AI processing
        image = cv2.imread(filepath)

        # Process the image through the model
        prediction_mask = process_frame(image)

        # Save prediction output as an image (for visualization)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction.jpg')
        cv2.imwrite(output_path, prediction_mask)

        return render_template('result.html', raw_image=filepath, predicted_image=output_path)

    return redirect(url_for('index'))

# Route to serve the image after prediction
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

# Start the Flask web app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
