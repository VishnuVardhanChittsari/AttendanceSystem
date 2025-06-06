import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, normalize
from mtcnn import MTCNN
from keras_facenet import FaceNet
from tensorflow import keras

def create_folder(base_dir, folder_name):
    """
    Create a folder if it doesn't exist.
    """
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.exists(folder_path):
        raise FileNotFoundError("No such file")
    return folder_path

def load_images_and_labels(data_dir):
    """
    Load images and labels from the specified directory.
    """
    images = []
    labels = []
    for label in os.listdir(data_dir):
        student_dir = os.path.join(data_dir, label)
        if os.path.isdir(student_dir):
            for image_name in os.listdir(student_dir):
                image_path = os.path.join(student_dir, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (224, 224))  # Resize image for consistency
                    images.append(image)
                    labels.append(label)
    return np.array(images), np.array(labels)

def extract_faces_and_embeddings(images, labels, detector, embedder):
    """
    Extract faces and embeddings from images.
    """
    embeddings = []
    associated_labels = []
    
    for image, label in zip(images, labels):
        detections = detector.detect_faces(image)
        
        if len(detections) == 0:
            print(f"No face detected in image with label: {label}")
            continue
        
        for detection in detections:
            x, y, width, height = detection['box']
            face = image[y:y+height, x:x+width]
            face_embedding = embedder.embeddings([face])[0]
            embeddings.append(face_embedding)
            associated_labels.append(label)
    
    return np.array(embeddings), np.array(associated_labels)

def augment_images_on_the_fly(image_array, num_augmented_images=10):
    """
    Generate augmented images on the fly without saving them.
    """
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=[0.8, 1.0],  # Zoom-out
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],  # Color space augmentation
        channel_shift_range=50.0,  # Color space augmentation
        preprocessing_function=lambda x: x + 0.05 * tf.random.normal(tf.shape(x)),  # Noise injection
    )

    augmented_images = []
    for batch in datagen.flow(image_array, batch_size=1):
        augmented_images.append(batch[0])
        if len(augmented_images) >= num_augmented_images:
            break
    
    return np.array(augmented_images)

def process_images(data_dir, num_augmented_images=10):
    """
    Process images by loading, augmenting, and extracting embeddings.
    """
    images, labels = load_images_and_labels(data_dir)
    total_images = len(images)
    
    detector = MTCNN()
    embedder = FaceNet()
    all_embeddings = []
    all_labels = []

    for image, label in zip(images, labels):
        # Include the original image
        image_array = image.reshape((1,) + image.shape)
        embeddings, associated_labels = extract_faces_and_embeddings([image], [label], detector, embedder)
        all_embeddings.extend(embeddings)
        all_labels.extend(associated_labels)

        # Augment and process the images
        augmented_images = augment_images_on_the_fly(image_array, num_augmented_images)
        aug_embeddings, aug_labels = extract_faces_and_embeddings(augmented_images, [label] * num_augmented_images, detector, embedder)
        all_embeddings.extend(aug_embeddings)
        all_labels.extend(aug_labels)

    all_embeddings = normalize(np.array(all_embeddings))
    return total_images, len(all_embeddings), np.array(all_embeddings), np.array(all_labels)

# Main Execution
if __name__ == "__main__":
    base_dir = r"M:\FinalYearProject\project\students"  # Replace with your base directory
    
    # Step 1: Labeling
    images, labels = load_images_and_labels(base_dir)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    np.save("labels.npy", encoded_labels)
    np.save("label_names.npy", label_encoder.classes_)
    np.save("images.npy", images)
    print("Labeling complete")
    
    # Step 2: Augmentation and Embeddings Extraction
    num_augmented_images = 100 # Define how many augmentations per image
    total_images, total_processed_images, embeddings, associated_labels = process_images(base_dir, num_augmented_images)
    np.save('embeddings.npy', embeddings)
    np.save('associated_labels.npy', label_encoder.transform(associated_labels))
    print("Processing complete")
    print(f"Total original images: {total_images}")
    print(f"Total processed images (including augmentations): {total_processed_images}")
    





