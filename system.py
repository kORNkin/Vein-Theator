from flask import Flask, render_template, request, jsonify
import os
import cv2
import tensorflow as tf
import numpy as np
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt

# โหลดโมเดล UNet
SAVED_MODEL = "C:/Users/noppa/Desktop/I-NewGen/System/Project/try/unet/best_model"
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
    image_path = "C:/Users/noppa/Desktop/I-NewGen/System/Project/image_path/captured_image.jpg"
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

# ฟังก์ชันแสดงผล
def plot_visualization(real_data, predicted_data):
    plt.figure(figsize=(6, 6))
    plt.imshow(real_data["image"].numpy().astype(np.uint8), cmap="gray")
    plt.title("Captured Image")
    plt.axis("off")
    plt.show()

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