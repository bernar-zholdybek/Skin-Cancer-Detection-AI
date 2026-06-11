import os
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image
import tensorflow as tf
import numpy as np

MODEL_PATH = "ham10000_cnn_transfer_learning.h5"


CLASSES = {
    0: "Actinic Keratoses and Intraepithelial Carcinoma",
    1: "Basal Cell Carcinoma",
    2: "Benign Keratosis-like Lesions",
    3: "Dermatofibroma",
    4: "Melanocytic Nevi",
    5: "Melanoma",
    6: "Vascular Lesions"
}
CANCEROUS_CLASSES = {
    0: True,   # Actinic Keratoses and Intraepithelial Carcinoma
    1: True,   # Basal Cell Carcinoma
    2: False,  # Benign Keratosis-like Lesions
    3: False,  # Dermatofibroma
    4: False,  # Melanocytic Nevi
    5: True,   # Melanoma
    6: False   # Vascular Lesions
}


def build_model():
    # Creating the architecture again to upload weights 

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax")
    ])
    return model



def load_model_safely():
    # Checking what's inside the .h5 and uploading properly

    if not os.path.exists(MODEL_PATH):
        print("Model file did not find:", MODEL_PATH)
        return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model is uploaded!")
        return model
    except Exception as e:
        print("Could not upload the file:", e)
        print("Trying to upload the weights...")
        try:
            model = build_model()
            model.load_weights(MODEL_PATH)
            print("Only weights are uploaded, archirecture is uploaded again.")
            return model
        except Exception as e2:
            print("Error trying to upload the:", e2)
            return None



class SkinAIApp(ctk.CTk):


    def __init__(self):
        # Main app of the skin cancer analyzisis

        super().__init__()

        # Window config
        self.title("XMUM AIT102 - Python and TensorFlow Project")
        self.geometry("900x700")
        ctk.set_appearance_mode("dark")

        # App state
        self.current_image_path = None
        self.model = load_model_safely()

        # UI
        self.setup_layout()



    def setup_layout(self):
        # Creating UI 

        self.header = ctk.CTkLabel(
            self,
            text="Skin Cancer Analysis",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.header.pack(pady=30)

        self.display_box = ctk.CTkFrame(self, width=500, height=400)
        self.display_box.pack(pady=10)

        self.img_label = ctk.CTkLabel(self.display_box, text="Please upload a JPG/PNG image")
        self.img_label.pack(expand=True, padx=20, pady=20)

        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(pady=30)

        self.upload_btn = ctk.CTkButton(
            self.btn_frame,
            text="Upload Image",
            command=self.handle_upload
        )
        self.upload_btn.grid(row=0, column=0, padx=10)

        self.analyze_btn = ctk.CTkButton(
            self.btn_frame,
            text="Analyze with AI",
            fg_color="#2fa572",
            command=self.run_inference
        )
        self.analyze_btn.grid(row=0, column=1, padx=10)



    def handle_upload(self):
        # Image uploading

        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if file_path:
            self.current_image_path = file_path
            raw_img = Image.open(file_path)
            display_img = ctk.CTkImage(light_image=raw_img, dark_image=raw_img, size=(400, 300))
            self.img_label.configure(image=display_img, text="")



    def run_inference(self):
        if not self.current_image_path:
            messagebox.showwarning("Input Required", "No image selected for analysis.")
            return

        if not self.model:
            messagebox.showerror("System Error", "Model could not be loaded.")
            return

        # ===== Image preprocessing =====
        img = Image.open(self.current_image_path).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ===== Prediction =====
        predictions = self.model.predict(img_array)[0]

        # ===== Top 3 results =====
        top_indices = predictions.argsort()[-3:][::-1]

        result_lines = []
        for i, idx in enumerate(top_indices, start=1):
            diagnosis = CLASSES.get(idx, "Unknown")
            confidence = float(predictions[idx] * 100)

            cancer_status = (
                "Cancerous" if CANCEROUS_CLASSES.get(idx, False)
                else "Non-cancerous"
            )

            result_lines.append(
                f"{i}) {cancer_status} — {diagnosis} ({confidence:.2f}%)"
            )

        result_text = "Top-3 Diagnoses:\n" + "\n".join(result_lines)
        messagebox.showinfo("AI Result", result_text)



if __name__ == "__main__":
    app = SkinAIApp()
    app.mainloop()