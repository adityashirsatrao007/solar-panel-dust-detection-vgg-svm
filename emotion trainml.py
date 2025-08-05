# emotion_training.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# --- Configuration ---
DATASET_PATH = r'D:\Emotion detection'
IMG_WIDTH, IMG_HEIGHT = 48, 48
BATCH_SIZE = 32
EPOCHS = 100
NUM_CLASSES = 7
CLASS_NAMES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
MODEL_SAVE_PATH = 'emotion_detection_model.h5'
CHECKPOINT_SAVE_PATH = 'emotion_detection_model_checkpoint.h5'

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as training_history.png")

def plot_confusion_matrix(y_true, y_pred_classes, class_names_list):
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_list, yticklabels=class_names_list)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix plot saved as confusion_matrix.png")

def calculate_steps(num_samples):
    return (num_samples + BATCH_SIZE - 1) // BATCH_SIZE

def main():
    global CLASS_NAMES
    global NUM_CLASSES

    # --- Data Preparation ---
    train_dir = os.path.join(DATASET_PATH, 'train')
    test_dir = os.path.join(DATASET_PATH, 'test')

    if not os.path.exists(DATASET_PATH):
        print(f"Error: The base DATASET_PATH '{DATASET_PATH}' does not exist.")
        return
    if not os.path.exists(train_dir):
        print(f"Error: Training directory '{train_dir}' not found.")
        print(f"Please ensure your dataset is structured correctly within '{DATASET_PATH}'.")
        print("Expected: .../Emotion detection/train/angry/img1.jpg")
        return
    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' not found.")
        print(f"Please ensure your dataset is structured correctly within '{DATASET_PATH}'.")
        print("Expected: .../Emotion detection/test/angry/img1.jpg")
        return

    # Data Augmentation with validation split
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Train generator (80% of training data)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        subset='training'
    )

    # Validation generator (20% of training data)
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False,
        subset='validation'
    )

    # Test generator (completely separate)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )

    # Verify class indices
    print("Class indices from training generator:", train_generator.class_indices)
    
    # Dynamically get ordered class names
    ordered_class_names = [None] * len(train_generator.class_indices)
    for class_name, index in train_generator.class_indices.items():
        ordered_class_names[index] = class_name
    
    # Update global variables
    if CLASS_NAMES != ordered_class_names:
        print(f"Updating global CLASS_NAMES to: {ordered_class_names}")
        CLASS_NAMES = ordered_class_names
        
    if NUM_CLASSES != len(CLASS_NAMES):
        print(f"Updating global NUM_CLASSES to: {len(CLASS_NAMES)}")
        NUM_CLASSES = len(CLASS_NAMES)

    # --- Class Weighting for Imbalanced Data ---
    print("Calculating class weights...")
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)

    # --- Model Building ---
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
    if train_generator.color_mode == 'grayscale':
        input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)

    model = build_model(input_shape, NUM_CLASSES)
    model.summary()

    # --- Callbacks ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
    
    callbacks_list = [early_stopping, reduce_lr]

    try:
        checkpoint_dir = os.path.dirname(CHECKPOINT_SAVE_PATH)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        model_checkpoint = ModelCheckpoint(
            CHECKPOINT_SAVE_PATH, 
            monitor='val_accuracy', 
            save_best_only=True, 
            verbose=1
        )
        callbacks_list.append(model_checkpoint)
    except OSError as e:
        print(f"Warning: Could not set up ModelCheckpoint. Error: {e}")

    # --- Model Training ---
    print("Starting model training...")
    if train_generator.samples == 0:
        print(f"Error: No images found in training directory")
        return
        
    train_steps = calculate_steps(train_generator.samples)
    val_steps = calculate_steps(validation_generator.samples)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps, 
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=val_steps, 
        callbacks=callbacks_list,
        class_weight=class_weights  # Add class weighting
    )

    # --- Save the final model ---
    print(f"Saving final model to {MODEL_SAVE_PATH}...")
    try:
        model_dir = os.path.dirname(MODEL_SAVE_PATH)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        model.save(MODEL_SAVE_PATH)
        print("Model saved.")
    except OSError as e:
        print(f"Error: Could not save model. Error: {e}")

    # --- Evaluation ---
    print("\nEvaluating model on TEST data...")
    best_model = model
    if 'model_checkpoint' in locals() and os.path.exists(CHECKPOINT_SAVE_PATH):
        print(f"Loading best model from checkpoint {CHECKPOINT_SAVE_PATH}...")
        try:
            best_model = tf.keras.models.load_model(CHECKPOINT_SAVE_PATH)
        except Exception as e:
            print(f"Warning: Could not load checkpoint. Using final model. Error: {e}")
            
    test_steps = calculate_steps(test_generator.samples)
    loss, accuracy = best_model.evaluate(test_generator, steps=test_steps)
    print(f"TEST Loss: {loss:.4f}")
    print(f"TEST Accuracy: {accuracy:.4f}")

    # --- Performance Metrics and Graphs ---
    plot_training_history(history)

    print("\nGenerating predictions for confusion matrix...")
    test_generator.reset()
    y_pred_probs = best_model.predict(test_generator, steps=test_steps)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    # Handle case where predictions might be more than samples
    y_true = test_generator.classes
    if len(y_pred_classes) > len(y_true):
        y_pred_classes = y_pred_classes[:len(y_true)]

    print("\nClassification Report:")
    # FIXED: Added missing closing parenthesis
    print(classification_report(
        y_true, 
        y_pred_classes, 
        target_names=CLASS_NAMES, 
        labels=np.arange(NUM_CLASSES), 
        zero_division=0
    ))

    plot_confusion_matrix(y_true, y_pred_classes, class_names_list=CLASS_NAMES)

    # --- Additional Analysis ---
    print("\nClass Distribution:")
    for i, class_name in enumerate(CLASS_NAMES):
        count = np.sum(y_true == i)
        print(f"{class_name}: {count} samples")
        
    print("\nTraining complete!")


if __name__ == '__main__':
    try:
        import tensorflow
    except ImportError:
        print("ERROR: TensorFlow is not installed. Please install it by running: pip install tensorflow")
        exit()

    if DATASET_PATH == r'path\to\your\dataset' or DATASET_PATH == 'path/to/your/dataset':
        print("ERROR: Please update the 'DATASET_PATH' variable in the script to your actual dataset directory.")
    else:
        main()