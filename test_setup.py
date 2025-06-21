#!/usr/bin/env python3
"""
Test Setup Script
This script tests the basic setup and dataset structure.
"""

import os
import sys

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing imports...")
    
    # Required packages
    required = {
        'tensorflow': 'tf',
        'numpy': 'np',
        'pandas': 'pd',
        'matplotlib.pyplot': 'plt',
        'seaborn': 'sns',
        'sklearn': 'sklearn',
        'cv2': 'cv2',
        'PIL': 'PIL'
    }
    
    # Optional packages
    optional = {
        'tensorflow_addons': 'tfa',
        'albumentations': 'A',
        'gradio': 'gr'
    }
    
    failed_required = []
    failed_optional = []
    
    for package, alias in required.items():
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_required.append(package)
    
    for package, alias in optional.items():
        try:
            __import__(package)
            print(f"✅ {package} (optional)")
        except ImportError:
            print(f"⚠️  {package} (optional)")
            failed_optional.append(package)
    
    return failed_required, failed_optional

def test_dataset():
    """Test dataset structure."""
    print("\n📁 Testing dataset structure...")
    
    dataset_dir = "TrashType_Image_Dataset"
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory '{dataset_dir}' not found!")
        return False
    
    print(f"✅ Dataset directory found: {dataset_dir}")
    
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    splits = ['train', 'test']
    
    total_images = 0
    
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            print(f"❌ {split} directory not found")
            continue
        
        print(f"\n{split.upper()}:")
        split_total = 0
        
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"  ❌ {class_name}: Not found")
                continue
            
            # Count images
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            count = len(images)
            split_total += count
            
            if count > 0:
                print(f"  ✅ {class_name}: {count} images")
            else:
                print(f"  ⚠️  {class_name}: No images found")
        
        print(f"  Total {split}: {split_total} images")
        total_images += split_total
    
    print(f"\n📊 Total images: {total_images}")
    return total_images > 0

def test_tensorflow():
    """Test TensorFlow setup."""
    print("\n🤖 Testing TensorFlow...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU available: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"   - {gpu.name}")
        else:
            print("⚠️  GPU not available, will use CPU")
        
        # Test basic operations
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = a + b
        print(f"✅ Basic TensorFlow operations work: {c.numpy()}")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow error: {e}")
        return False

def test_model_building():
    """Test if we can build the model."""
    print("\n🏗️ Testing model building...")
    
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import EfficientNetV2L
        
        # Test building a simple model
        img_input = layers.Input(shape=(224, 224, 3))
        base_model = EfficientNetV2L(
            include_top=False,
            weights='imagenet',
            input_tensor=img_input,
            pooling='avg'
        )
        
        x = base_model.output
        x = layers.Dense(6, activation='softmax')(x)
        model = models.Model(inputs=img_input, outputs=x)
        
        print(f"✅ Model built successfully")
        print(f"   Parameters: {model.count_params():,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model building error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Enhanced Garbage Classification Setup")
    print("=" * 50)
    
    # Test imports
    failed_required, failed_optional = test_imports()
    
    if failed_required:
        print(f"\n❌ Required packages missing: {', '.join(failed_required)}")
        print("Please run: python install_dependencies.py")
        return False
    
    if failed_optional:
        print(f"\n⚠️  Optional packages missing: {', '.join(failed_optional)}")
        print("The model will work with basic features.")
    
    # Test dataset
    dataset_ok = test_dataset()
    
    # Test TensorFlow
    tf_ok = test_tensorflow()
    
    # Test model building
    model_ok = test_model_building()
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Imports: {'✅' if not failed_required else '❌'}")
    print(f"   Dataset: {'✅' if dataset_ok else '❌'}")
    print(f"   TensorFlow: {'✅' if tf_ok else '❌'}")
    print(f"   Model Building: {'✅' if model_ok else '❌'}")
    
    if not failed_required and dataset_ok and tf_ok and model_ok:
        print("\n🎉 All tests passed! You're ready to train the model.")
        print("\nNext steps:")
        print("1. Run: python garbage_classification_fixed.py")
        print("2. Or run: python train_enhanced_model.py (for quick testing)")
        return True
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 