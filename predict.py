import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ==========================================
# LOAD MODEL
# ==========================================

model = load_model("model/plant_model.h5")

# ==========================================
# CLASS NAMES
# ==========================================

class_names = [

    "aloe_vera",
    "amla",
    "bael",
    "catnip",
    "lemon_balm",
    "mint",
    "neem",
    "peppermint",
    "stevia",
    "tulsi"

]

# ==========================================
# MEDICINAL INFO DATABASE
# ==========================================

plant_info = {

    "tulsi":
        "Used for cough, cold, immunity and anti-inflammatory treatment.",

    "neem":
        "Used for skin diseases and antibacterial treatment.",

    "mint":
        "Used for digestion and cooling effect.",

    "aloe_vera":
        "Used for skin healing and digestion.",

    "amla":
        "Rich in Vitamin C and improves immunity.",

    "bael":
        "Used for digestive disorders and stomach health.",

    "catnip":
        "Used for relaxation and herbal remedies.",

    "lemon_balm":
        "Used for stress relief and better sleep.",

    "peppermint":
        "Used for digestion and headache relief.",

    "stevia":
        "Natural sugar substitute used for diabetes management."
}

# ==========================================
# IMAGE PATH
# ==========================================

img_path = "test_leaf.png"

# ==========================================
# PREPROCESS IMAGE
# ==========================================

img = image.load_img(img_path, target_size=(224,224))

img_array = image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)

predicted_class = class_names[predicted_index]

confidence = np.max(prediction) * 100

# ==========================================
# OUTPUT
# ==========================================

print("\n===================================")

print("Predicted Plant :", predicted_class)

print("Confidence Score: {:.2f}%".format(confidence))

print("\nMedicinal Uses:")

print(plant_info[predicted_class])

print("===================================")