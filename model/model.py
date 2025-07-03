import pandas as pd
import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def load_metadata(metadata_path):
    return pd.read_csv(metadata_path)

def load_images(image_dir, df, size=(64,64)):
    images = []
    labels = []
    for i, row in df.iterrows():
        img_path = os.path.join(image_dir, row['image_id'] + ".jpg")
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, size)
            images.append(img / 255.0)
            labels.append(row['dx'])
    return np.array(images), labels

def preprocess_labels(labels):
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)
    return labels_categorical, label_encoder

def build_simple_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_transfer_model(input_shape, num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_confusion_matrix(y_true, y_pred_classes, class_names):
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def show_sample_predictions(X_test, y_true, y_pred_classes, class_names, n=5):
    for i in range(n):
        plt.imshow(X_test[i])
        plt.title(f"True: {class_names[y_true[i]]}, Pred: {class_names[y_pred_classes[i]]}")
        plt.axis('off')
        plt.show()

def main():
    # Load data
    df = load_metadata("data/HAM10000_metadata.csv")
    images, labels = load_images("data/HAM10000_images_part_1", df, size=(64,64))
    labels_categorical, label_encoder = preprocess_labels(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # Simple CNN
    model = build_simple_model((64, 64, 3), y_train.shape[1])
    print("Training simple CNN model...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    model.save('model.keras')

    # Augmented training
    print("Training with data augmentation...")
    aug_history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=10,
        validation_data=(X_test, y_test)
    )

   

    # Evaluation
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))
    plot_confusion_matrix(y_true, y_pred_classes, label_encoder.classes_)
    show_sample_predictions(X_test, y_true, y_pred_classes, label_encoder.classes_)

if __name__ == "__main__":
    main()
