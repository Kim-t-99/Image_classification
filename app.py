import tkinter as tk
from tkinter import filedialog, Label
from tensorflow.keras.models import load_model
import joblib
import cv2
from PIL import Image, ImageTk
import numpy as np


best_model = load_model("best_model.h5")
label_enc = joblib.load("label_encoder.pkl")

def preprocess_img(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.reshape(1, 64, 64, 1)

    return img

def classify_img():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    img_arr = preprocess_img(file_path)
    predict = best_model.predict(img_arr)
    class_index = np.argmax(predict)
    predicted_class = label_enc.inverse_transform([class_index])[0]
    result_label.config(text=f"Predicted class: {predicted_class}")


    img = Image.open(file_path)
    img.thumbnail((200, 200))
    tk_img = ImageTk.PhotoImage(img)
    image_label.config(image=tk_img)
    image_label.image = tk_img


root = tk.Tk()
root.title("Image Classification")
root.geometry("500x300")

button = tk.Button(root, text="Classify Image", command=classify_img)
button.pack(pady=20)

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="")
result_label.pack()

root.mainloop()
