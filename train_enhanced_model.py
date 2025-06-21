#!/usr/bin/env python3
"""
Quick Training Script for Enhanced Garbage Classification Model
This script provides a streamlined version for testing the enhanced model.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall, TopKCategoricalAccuracy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
import time
import random

# Try to import optional dependencies
try:
    import tensorflow_addons as tfa
    TFA_AVAILABLE = True
    print("✅ tensorflow-addons available")
except ImportError:
    print("⚠️  tensorflow-addons not available, using standard Adam optimizer")
    TFA_AVAILABLE = False

# Set mixed precision for performance (only if supported)
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("✅ Mixed precision enabled")
except:
    print("⚠️  Mixed precision not supported, using float32")

# Constants
IMG_SIZE = (380, 380)
BATCH_SIZE = 8  # Smaller for quick testing
EPOCHS = 10  # Reduced for quick testing
FINE_TUNE_EPOCHS = 5
SEED = 42
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
NUM_CLASSES = len(CLASS_NAMES)

# Path configuration
dataset_dir = "TrashType_Image_Dataset"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")

# Set random seeds
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

def create_metadata(dataset_dir):
    """Create metadata dataframe with enhanced features."""
    metadata = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(dataset_dir, "train", class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️  {class_dir} not found")
            continue
        for img_name in os.listdir(class_dir)[:100]:  # Limit for quick testing
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(class_dir, img_name)
            try:
                file_size = os.path.getsize(img_path)
                file_number = int(img_name.split("_")[-1].split(".")[0]) if "_" in img_name else 0
                
                metadata.append({
                    "path": img_path,
                    "class": class_name,
                    "size": file_size,
                    "size_log": np.log(file_size + 1),
                    "size_normalized": file_size / 1e6,
                    "file_number": file_number
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    return pd.DataFrame(metadata)

def simple_augmentation(img):
    """Simple augmentation for quick testing."""
    # Basic augmentations
    if np.random.random() < 0.5:
        img = np.fliplr(img)
    if np.random.random() < 0.3:
        img = np.flipud(img)
    if np.random.random() < 0.3:
        # Random brightness
        img = np.clip(img * (0.8 + 0.4 * np.random.random()), 0, 1)
    return img

def create_data_generators(metadata, test_dir):
    """Create data generators with simple augmentation."""
    if len(metadata) == 0:
        raise ValueError("No metadata found. Please check your dataset structure.")
    
    # Split metadata
    train_meta, val_meta = train_test_split(
        metadata, test_size=0.2, stratify=metadata['class'], random_state=SEED
    )
    
    # Metadata features
    meta_features = ['size_normalized', 'size_log', 'file_number']
    
    # Create test metadata
    test_meta = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️  {class_dir} not found")
            continue
        for img_name in os.listdir(class_dir)[:50]:  # Limit for quick testing
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(class_dir, img_name)
            try:
                file_size = os.path.getsize(img_path)
                file_number = int(img_name.split("_")[-1].split(".")[0]) if "_" in img_name else 0
                
                test_meta.append({
                    "path": img_path,
                    "class": class_name,
                    "size": file_size,
                    "size_log": np.log(file_size + 1),
                    "size_normalized": file_size / 1e6,
                    "file_number": file_number
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    test_meta = pd.DataFrame(test_meta)
    
    def generator(meta_data, batch_size, shuffle=False, augment=False):
        while True:
            if shuffle:
                meta_data = meta_data.sample(frac=1, random_state=SEED).reset_index(drop=True)
            
            for i in range(0, len(meta_data), batch_size):
                batch_meta = meta_data.iloc[i:i+batch_size]
                batch_images = []
                batch_labels = []
                batch_meta_features = []
                
                for _, row in batch_meta.iterrows():
                    try:
                        # Load image
                        img = cv2.imread(row['path'])
                        if img is None:
                            print(f"Warning: Could not load image {row['path']}")
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, IMG_SIZE)
                        img = img / 255.0
                        
                        # Apply augmentation
                        if augment:
                            img = simple_augmentation(img)
                        
                        batch_images.append(img)
                        
                        # Create label
                        label = np.zeros(NUM_CLASSES)
                        label[CLASS_NAMES.index(row['class'])] = 1
                        batch_labels.append(label)
                        
                        # Metadata features
                        batch_meta_features.append([row['size_normalized'], row['size_log'], row['file_number']])
                    except Exception as e:
                        print(f"Error processing {row['path']}: {e}")
                        continue
                
                if len(batch_images) == 0:
                    continue
                
                batch_images = np.array(batch_images)
                batch_labels = np.array(batch_labels)
                batch_meta_features = np.array(batch_meta_features)
                
                yield [batch_images, batch_meta_features], batch_labels
    
    train_gen = generator(train_meta, BATCH_SIZE, shuffle=True, augment=True)
    val_gen = generator(val_meta, BATCH_SIZE, shuffle=False, augment=False)
    test_gen = generator(test_meta, BATCH_SIZE, shuffle=False, augment=False)
    
    return train_gen, val_gen, test_gen, len(train_meta), len(val_meta), len(test_meta), len(meta_features)

# Attention mechanism (simplified)
class ChannelAttention(layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        
        channels = input_shape[-1]
        self.fc1 = layers.Dense(channels // self.reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
        
    def call(self, x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        out = avg_out + max_out
        return x * tf.expand_dims(tf.expand_dims(out, 1), 1)

def build_model(meta_dim):
    """Build enhanced model with attention mechanisms."""
    # Image input
    img_input = layers.Input(shape=(*IMG_SIZE, 3), name='image_input')
    
    # EfficientNetV2L backbone
    base_model = EfficientNetV2L(
        include_top=False,
        weights='imagenet',
        input_tensor=img_input,
        pooling=None
    )
    base_model.trainable = False
    
    # Add attention
    x = base_model.output
    x = ChannelAttention(reduction_ratio=16)(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Metadata branch
    meta_input = layers.Input(shape=(meta_dim,), name='meta_input')
    meta_branch = layers.Dense(64, activation='swish')(meta_input)
    meta_branch = layers.BatchNormalization()(meta_branch)
    meta_branch = layers.Dropout(0.3)(meta_branch)
    
    # Combine
    x = layers.Concatenate()([x, meta_branch])
    
    # Classification head
    x = layers.Dense(512, activation='swish', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, activation='swish', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='output', dtype=tf.float32)(x)
    
    model = models.Model(inputs=[img_input, meta_input], outputs=outputs)
    
    # Compile
    if TFA_AVAILABLE:
        optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
    else:
        optimizer = Adam(learning_rate=0.001)
    
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy', TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    
    return model, base_model

def train_model(model, train_gen, val_gen, train_steps, val_steps):
    """Train the model."""
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint('quick_best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    ]
    
    history = model.fit(
        train_gen,
        steps_per_epoch=max(1, train_steps // BATCH_SIZE),
        validation_data=val_gen,
        validation_steps=max(1, val_steps // BATCH_SIZE),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def fine_tune_model(model, base_model, train_gen, val_gen, train_steps, val_steps):
    """Fine-tune the model."""
    base_model.trainable = True
    for layer in base_model.layers[:100]:  # Freeze fewer layers for quick testing
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy', 'top3_accuracy']
    )
    
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-8, verbose=1),
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint('quick_best_finetuned_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    ]
    
    history_fine = model.fit(
        train_gen,
        steps_per_epoch=max(1, train_steps // BATCH_SIZE),
        validation_data=val_gen,
        validation_steps=max(1, val_steps // BATCH_SIZE),
        epochs=EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return history_fine

def evaluate_model(model, test_gen, test_steps):
    """Evaluate the model."""
    test_loss, test_acc, test_top3 = model.evaluate(
        test_gen, steps=max(1, test_steps // BATCH_SIZE)
    )
    print(f"\nTest Accuracy: {test_acc:.2%}")
    print(f"Test Top-3 Accuracy: {test_top3:.2%}")
    
    # Generate predictions
    y_true, y_pred = [], []
    for _ in range(max(1, test_steps // BATCH_SIZE)):
        try:
            ([images, meta_features], labels) = next(test_gen)
            preds = model.predict([images, meta_features], verbose=0)
            y_true.extend(np.argmax(labels, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
        except Exception as e:
            print(f"Error during evaluation: {e}")
            continue
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Quick Test)')
    plt.savefig('confusion_matrix_quick.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return test_acc, test_top3

def main():
    """Main training function."""
    print("🚀 Quick Training for Enhanced Garbage Classification Model")
    print("This is a simplified version for testing the enhancements.")
    start_time = time.time()
    
    # Check if dataset exists
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory '{dataset_dir}' not found!")
        print("Please ensure the dataset is in the correct location.")
        return
    
    # Create metadata
    print("\n📊 Creating metadata...")
    metadata = create_metadata(dataset_dir)
    print(f"Total samples: {len(metadata)}")
    if len(metadata) > 0:
        print(f"Class distribution:\n{metadata['class'].value_counts()}")
    else:
        print("❌ No training data found!")
        return
    
    # Create data generators
    print("\n🔄 Creating data generators...")
    try:
        train_gen, val_gen, test_gen, train_steps, val_steps, test_steps, meta_dim = create_data_generators(
            metadata, test_dir
        )
        print(f"Train samples: {train_steps}")
        print(f"Validation samples: {val_steps}")
        print(f"Test samples: {test_steps}")
    except Exception as e:
        print(f"❌ Error creating data generators: {e}")
        return
    
    # Build model
    print("\n🏗️ Building model...")
    try:
        model, base_model = build_model(meta_dim=meta_dim)
        print(f"Model parameters: {model.count_params():,}")
    except Exception as e:
        print(f"❌ Error building model: {e}")
        return
    
    # Train model
    print("\n🎯 Starting training...")
    try:
        history = train_model(model, train_gen, val_gen, train_steps, val_steps)
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return
    
    # Fine-tune
    print("\n🔧 Starting fine-tuning...")
    try:
        history_fine = fine_tune_model(
            model, base_model, train_gen, val_gen, train_steps, val_steps
        )
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        return
    
    # Evaluate
    print("\n📈 Evaluating model...")
    try:
        test_acc, test_top3 = evaluate_model(model, test_gen, test_steps)
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return
    
    # Save final model
    try:
        model.save('garbage_classifier_quick_test.keras')
        print(f"\n💾 Model saved with accuracy: {test_acc:.2%}, Top-3: {test_top3:.2%}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
    
    print(f"\n⏱️ Total execution time: {time.time() - start_time:.2f} seconds")
    print("\n✅ Quick training completed! Run the full training script for best results.")

if __name__ == "__main__":
    main() 