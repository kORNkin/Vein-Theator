from flask import Flask, render_template, jsonify, request
import requests
import io
from PIL import Image
import os
import cv2
import tensorflow as tf
import numpy as np
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# IP address of the Raspberry Pi
raspberry_pi_ip = 'http://172.20.10.7:5000/take_photo'  # Update the path to call the /take_photo route

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/take_photo', methods=['POST'])
def take_photo():
    try:
        # Send a POST request to the Raspberry Pi's /take_photo route
        response = requests.post(raspberry_pi_ip)  # Correct the URL to your Raspberry Pi's endpoint
        
        # If the request is successful
        if response.status_code == 200:
            # Get the image data from the response (it should be in binary)
            img_data = response.content
            
            # Convert the image data into a PIL image
            image = Image.open(io.BytesIO(img_data))
            
            # Save the image to the static folder
            photo_path = 'static/photo.jpg'
            image.save(photo_path)
            
            # Return the URL of the saved photo
            return jsonify({'photo_url': f'/{photo_path}'})
        else:
            print(f"Failed to capture photo on Raspberry Pi, status code: {response.status_code}")
            return jsonify({'error': 'Failed to capture photo on Raspberry Pi'}), 500
    except requests.exceptions.RequestException as e:
        print(f"Error during photo capture request: {e}")
        return jsonify({'error': 'Error connecting to Raspberry Pi'}), 500

SAVED_MODEL = "try/unet/best_model"
unet = tf.keras.models.load_model(SAVED_MODEL)


# ตั้งค่าขนาดภาพ
IMAGE_SIZE = 512

# 📷 **ถ่ายภาพจากกล้อง**
def capture_image():
    cap = cv2.VideoCapture(0)  # เปิดกล้อง (Camera Index = 0)
    cap.set(3, IMAGE_SIZE)  # ตั้งค่าความกว้างของกล้อง
    cap.set(4, IMAGE_SIZE)  # ตั้งค่าความสูงของกล้อง
    ret, frame = cap.read()  # อ่านภาพจากกล้อง
    cap.release()  # ปิดกล้อง
    
    if not ret:
        print("❌ ไม่สามารถถ่ายภาพได้")
        return None
    
    # แปลงภาพเป็นขาวดำ (Grayscale)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # บันทึกไฟล์เป็นภาพชั่วคราว
    image_path = "      "
    cv2.imwrite(image_path, frame_gray)
    
    return image_path

# ฟังก์ชันอ่านภาพ
def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)  # ใช้ 1 channel (Grayscale)
    base_image = image
    image.set_shape([None, None, 1])
    
    if mask:
        image = tf.cast(image, dtype=tf.int32)
        return image
    else:
        image = tf.cast(image, dtype=tf.float32)
        image = image / 127.5 - 1  # Normalization [-1,1]
        return image, base_image

# ฟังก์ชันปรับขนาดภาพให้เป็น 512x512
def normalize(image):
    image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    return image

# ฟังก์ชันพยากรณ์
def infer(model, image_tensor):
    mask, values = model.predict(np.expand_dims(image_tensor, axis=0), verbose=0)
    mask = np.squeeze(mask)
    mask_npy = np.argmax(mask, axis=2).astype(np.uint8)
    mask = tf.convert_to_tensor(mask_npy)

    predictions = {
        "image": image_tensor,
        "mask": mask,
        "x": int(values[0][0] * 1080) + 420,
        "y": int(values[0][1] * 1080),
        "angle": values[0][2] * 180 if len(values[0]) > 2 else 0,
        "type": "predicted"
    }

    return predictions

def plot_visualization(real_data, predicted_data):
    try:
        colors = {
            "background": [59, 82, 139],
            "arm": [3, 31, 254],
            "veins": [253, 231, 37]
        }

        visualization = []
        image = real_data["image"].numpy().astype(np.uint8)

        # Convert grayscale image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize real image to 1080x1920
        image_resized = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)

        for data in [real_data, predicted_data]:
            mask = data["mask"].numpy().astype(np.uint8)

            # Ensure the mask is 3-channel (convert grayscale to RGB)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

            # Assign color to mask classes
            mask[np.where((mask == [0, 0, 0]).all(axis=2))] = colors["background"]
            mask[np.where((mask == [1, 1, 1]).all(axis=2))] = colors["arm"]
            mask[np.where((mask == [2, 2, 2]).all(axis=2))] = colors["veins"]

            if data["type"] == "predicted":
                # Resize mask to 1080x1080
                mask_resized = cv2.resize(mask, (1080, 1080), interpolation=cv2.INTER_AREA)

                # Create blank mask (1080x1920) and place the 1080x1080 mask in the center
                mask_final = np.full((1080, 1920, 3), colors["background"], dtype=np.uint8)
                mask_final[:, 540:1620, :] = mask_resized  # Center the mask
            else:
                mask_final = cv2.resize(mask, (1920, 1080), interpolation=cv2.INTER_AREA)

            # Ensure both images are the same size
            if image_resized.shape != mask_final.shape:
                print("Size mismatch! Resizing mask to match image size.")
                mask_final = cv2.resize(mask_final, (1920, 1080), interpolation=cv2.INTER_AREA)

            # Blend image with mask
            new_image = cv2.addWeighted(image_resized, 0.8, mask_final, 0.5, 0.0)

            # Draw the predicted point and angle on the image
            angle = data["angle"]
            cv2.circle(new_image, (data["x"], data["y"]), radius=10, color=(0, 255, 0), thickness=-1)
            cv2.putText(new_image, f"{angle:.2f}", (data["x"] + 10, data["y"] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), thickness=5)
            visualization.append(new_image)

            # Create second visualization with a bounding box
            new_image2 = new_image.copy()
            size = 100
            start_point = (data["x"] - int(size / 2), data["y"] - int(size / 2))
            end_point = (data["x"] + int(size / 2), data["y"] + int(size / 2))
            cv2.rectangle(new_image2, start_point, end_point, (0, 255, 0), 5)
            visualization.append(new_image2)

        # Save the processed images
        output_path1 = "static/predicted_image1.jpg"
        output_path2 = "static/predicted_image2.jpg"
        output_path3 = "static/predicted_image3.jpg"

        cv2.imwrite(output_path1, visualization[0])  # First visualization
        cv2.imwrite(output_path2, visualization[1])  # Second visualization (Bounding box)
        cv2.imwrite(output_path3, visualization[2])  # Third visualization (Optional)

        return [output_path1, output_path2, output_path3]
    
    except Exception as e:
        print(f'Error during AI processing: {e}')
        return None

# ฟังก์ชันพยากรณ์โดยใช้ภาพจากกล้อง
def make_prediction(model):
    image_path = capture_image()  # 📷 ถ่ายภาพใหม่จากกล้อง
    if image_path is None:
        return
    
    image_tensor, base_image_tensor = read_image(image_path)

    mask_tensor = tf.zeros_like(base_image_tensor)  # สร้าง mask เปล่า

    real_data = {
        "image": base_image_tensor,
        "mask": mask_tensor,
        "x": 0,
        "y": 0,
        "angle": 0,
        "type": "real"
    }

    image_tensor = normalize(image_tensor)
    predicted_data = infer(model=model, image_tensor=image_tensor)
    plot_visualization(real_data, predicted_data)

    print("🔍 ใช้ภาพที่ถ่ายจากกล้อง:", image_path)

@app.route('/process_ai', methods=['POST'])
def process_ai():
    try:
        image_path = 'static/photo.jpg' # Modify this to capture image or use an existing image
        if image_path is None:
            return
        
        image_tensor, base_image_tensor = read_image(image_path)
        print("1")

        mask_tensor = tf.zeros_like(base_image_tensor)  # สร้าง mask เปล่า
        print("2")
        real_data = {
            "image": base_image_tensor,
            "mask": mask_tensor,
            "x": 0,
            "y": 0,
            "angle": 0,
            "type": "real"
        }
        print("3")
        image_tensor = normalize(image_tensor)
        print("4")
        predicted_data = infer(model=unet, image_tensor=image_tensor)
        print("5")
        predicted_image_path = plot_visualization(real_data, predicted_data)
        print("6")

        return jsonify({'mask_url': f'/{predicted_image_path[2]}'}), 200
    except Exception as e:
        return jsonify({'error': f'Error during AI processing: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
