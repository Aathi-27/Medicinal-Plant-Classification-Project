from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os

input_dir = "E:/MIni project/medicinal_plant_dataset/dataset"
output_dir = "augmented_dataset"

datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    fill_mode='nearest'
)

for class_name in os.listdir(input_dir):

    class_input = os.path.join(input_dir, class_name)
    class_output = os.path.join(output_dir, class_name)

    os.makedirs(class_output, exist_ok=True)

    for img_name in os.listdir(class_input):

        img_path = os.path.join(class_input, img_name)

        img = image.load_img(img_path, target_size=(224,224))
        x = image.img_to_array(img)
        x = x.reshape((1,) + x.shape)

        count = 0

        for batch in datagen.flow(
            x,
            batch_size=1,
            save_to_dir=class_output,
            save_prefix=class_name,
            save_format='png'
        ):

            count += 1

            if count >= 20:
                break

print("Augmentation completed.")