import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from PIL import ImageTk, Image

#%% Load the model
model = load_model('C:/Users/Elif/Desktop/Plant_diseaseDetection/eski/cnn_best_model.keras')

#%% Load the CSV data
disease_data = pd.read_csv('C:/Users/Elif/Desktop/Plant_diseaseDetection/Plant_diseases.csv')

#%% Define the class labels
class_labels = [
    'Aloevera_Healthy', 'Aloevera_LeafSpot', 'Aloevera_Rust', 
    'Apple_BlackRot', 'Apple_CedarRust', 'Apple_Healthy', 'Apple_Scab', 
    'Banana_Anthracnose', 'Banana_BlackLeaf', 'Banana_BunchyTop', 
    'Banana_CigarAndRot', 'Banana_CordanaLeafSpot', 'Banana_Panama', 
    'Blueberry_AnthracnoseLeaf', 'Blueberry_ExobasidiumLeafSpot', 
    'Blueberry_Healthy', 'Blueberry_SeptoriaLeafSpot', 'Cherry_Healthy', 
    'Cherry_PowderyMildew', 'Coffee_BerryBlotch', 'Coffee_BlackRot', 
    'Coffee_BrownEyeSpot', 'Coffee_CercosporaLeafSpot', 'Coffee_Healthy', 
    'Coffee_Rust', 'Corn_CercosporaAndGrayLeafSpot', 'Corn_CommonRust', 
    'Corn_Healthy', 'Corn_NorthernLeafBlight', 'Cotton_Anthracnose', 
    'Cotton_Aphids', 'Cotton_BacterialBlight', 'Cotton_BollRot', 
    'Cotton_CurlVirus', 'Cotton_FusariumWilt', 'Cotton_Healthy', 
    'Cotton_PowderyMildew', 'Cotton_TargetSpot', 'Cucumber_AngularLeafSpot', 
    'Cucumber_BacterialWilt', 'Cucumber_PowderyMildew', 'Grape_BlackMeasles', 
    'Grape_BlackRot', 'Grape_Healthy', 'Grape_LeafBlight', 'Lentil_AscochytaBlight', 
    'Lentil_Healthy', 'Lentil_PowderyMildew', 'Lentil_Rust', 'Orange_Haunglongbing', 
    'Peach_BacterialSpot', 'Peach_Healthy', 'PepperBell_BacterialSpot', 
    'PepperBell_Healthy', 'Potato_EarlyBlight', 'Potato_Healthy', 'Potato_LateBlight', 
    'Raspberry_Healthy', 'Rice_BacterialBlight', 'Rice_Blast', 'Rice_BrownSpot', 
    'Rice_Healthy', 'Rice_LeafBlast', 'Rice_LeafSmut', 'Rice_SheathBlight', 'Rice_Tungro', 
    'Rose_BlackSpot', 'Rose_DownyMildew', 'Rose_Healthy', 'Soybean_Healthy', 
    'Squash_PowderyMildew', 'Strawberry_Anthracnose', 'Strawberry_Healthy', 
    'Strawberry_LeafScorch', 'Sugarcane_BacterialBlight', 'Sugarcane_Healthy', 
    'Sugarcane_Mosaic', 'Sugarcane_RedRot', 'Sugarcane_RedRust', 'Sugarcane_RedStripe', 
    'Sugarcane_Rust', 'Sugarcane_Yellowing', 'Sunflower_DownyMildew', 'Sunflower_GrayMold', 
    'Sunflower_Healthy', 'Sunflower_LeafScars', 'Tea_AlgalLeaf', 'Tea_Anthracnose', 
    'Tea_BirdEyeSpot', 'Tea_BrownBlight', 'Tea_Healthy', 'Tea_RedLeafSpot', 
    'Tomato_BacterialSpot', 'Tomato_EarlyBlight', 'Tomato_Healthy', 'Tomato_LateBlight', 
    'Tomato_LeafMold', 'Tomato_MosaicVirus', 'Tomato_SeptoriaLeafSpot', 
    'Tomato_SpiderMites', 'Tomato_TargetSpot', 'Tomato_YellowLeafCurlVirus', 
    'Wheat_Aphid', 'Wheat_BacterialLeafStreak', 'Wheat_BlackRust', 'Wheat_BrownRust', 
    'Wheat_FlagSmut', 'Wheat_Healthy', 'Wheat_LeafBlight', 'Wheat_LooseSmut', 
    'Wheat_PowderyMildew', 'Wheat_Scab', 'Wheat_Septoria', 'Wheat_SeptoriaBlotch', 
    'Wheat_StemRust', 'Wheat_StripeRust'
]


#%% Function to predict the image
def predict_image(image_path):
    try:
        # Load the image and resize it to the model input size
        img = image.load_img(image_path, target_size=(254, 254))
        img_array = image.img_to_array(img)  # Convert the image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch size (1, 224, 224, 3)
        img_array = img_array / 255.0  # Normalize the image

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Get the class index
        predicted_class_label = class_labels[predicted_class]  # Get the label
        confidence = np.max(predictions) * 100  # Get confidence (probability)

        # Return prediction and confidence
        return predicted_class_label, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

#%% Function to get disease details from CSV
def get_disease_details(predicted_class_label):
    # Split class label into plant and disease parts
    plant, disease = predicted_class_label.split('_', 1)
    
    # Search for the disease in the CSV and get details
    disease_info = disease_data[disease_data['Disease Name'] == disease]
    
    if not disease_info.empty:
        cause = disease_info['Cause'].values[0]
        return f"Disease Cause: {cause}"
    else:
        return "No additional details available."

#%% Function to open a file dialog and select an image
def open_image():
    # Kullanıcıdan bir görüntü seçmesini iste
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        display_image(file_path)  # Seçilen görüntüyü ekranda göster
        predicted_class, confidence = predict_image(file_path)  # Tahmin yap

        if predicted_class and confidence is not None:
            # Tahmin sonuçlarını ekranda göster
            result_label.config(text=f"Predicted Class: {predicted_class}")
            confidence_label.config(text=f"Confidence: {confidence:.2f}%")

            # Bitki ve hastalık adını ayır (_ karakterine göre split)
            plant, disease = predicted_class.split('_', 1)  # İlk '_' karakterine göre ayır

            # Eğer 'Healthy' ise durumunu belirt
            if disease == "Healthy":
                details_label.config(text=f"The plant '{plant}' is healthy.")
            else:
                details_label.config(text=f"The plant '{plant}' has the disease '{disease}'.")
                
            # CSV'den hastalık bilgilerini al
            disease_details = get_disease_details(predicted_class)
            disease_details_label.config(text=disease_details)
        else:
            result_label.config(text="Error during prediction!")

#%% Function to display the selected image
def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((250, 250))
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img


# Create the Tkinter window
root = tk.Tk()
root.title("Plant Disease Classifier")
root.geometry("700x800")
root.config(bg="#f5f5f5")

# Styling for the widgets
bg_color = "#f5f5f5"  # Light grey background color
button_color = "#007BFF"  # Blue button color
button_hover_color = "#0056b3"  # Darker blue for hover
font_style = ("Arial", 14)
title_font = ("Arial", 18, "bold")

# Create GUI elements
img_label = tk.Label(root, bg=bg_color)
img_label.grid(row=0, column=0, columnspan=2, pady=20)

open_button = tk.Button(root, text="Open Image", command=open_image, bg=button_color, fg="white", font=("Arial", 16, "bold"), relief="flat", bd=2)
open_button.grid(row=1, column=0, columnspan=2, pady=20)

result_label = tk.Label(root, text="Predicted Class: Awaiting prediction...", font=font_style, fg="blue", bg=bg_color)
result_label.grid(row=2, column=0, columnspan=2, sticky="ew", padx=20, pady=10)

confidence_label = tk.Label(root, text="Confidence: Awaiting prediction...", font=font_style, fg="green", bg=bg_color)
confidence_label.grid(row=3, column=0, columnspan=2, sticky="ew", padx=20, pady=10)

details_label = tk.Label(root, text="Details will appear here.", font=font_style, fg="black", bg=bg_color, justify="left")
details_label.grid(row=5, column=0, columnspan=2, sticky="w", padx=20, pady=10)

disease_details_label = tk.Label(root, text="Disease details will appear here.", font=font_style, fg="red", bg=bg_color)
disease_details_label.grid(row=4, column=0, columnspan=2, sticky="ew", padx=20, pady=10)

# Add some space at the bottom for a more balanced look
bottom_label = tk.Label(root, text="Plant Disease Detection", font=("Arial", 12), fg="#888", bg=bg_color)
bottom_label.grid(row=6, column=0, columnspan=2, pady=20)

# Run the application
root.mainloop()
