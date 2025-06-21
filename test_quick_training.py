#!/usr/bin/env python3
"""
Test Quick Training Script
This script tests the quick training functionality.
"""

import os
import sys

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing imports for quick training...")
    
    required = {
        'tensorflow': 'tf',
        'numpy': 'np',
        'pandas': 'pd',
        'matplotlib.pyplot': 'plt',
        'seaborn': 'sns',
        'sklearn.metrics': 'sklearn.metrics',
        'sklearn.model_selection': 'sklearn.model_selection',
        'cv2': 'cv2'
    }
    
    optional = {
        'tensorflow_addons': 'tfa'
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

def test_dataset_structure():
    """Test if dataset structure is correct for quick training."""
    print("\n📁 Testing dataset structure...")
    
    dataset_dir = "TrashType_Image_Dataset"
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory '{dataset_dir}' not found!")
        return False
    
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    splits = ['train', 'test']
    
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            print(f"❌ {split} directory not found")
            return False
        
        print(f"\n{split.upper()}:")
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"  ❌ {class_name}: Not found")
                return False
            
            # Count images
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            count = len(images)
            
            if count > 0:
                print(f"  ✅ {class_name}: {count} images")
            else:
                print(f"  ⚠️  {class_name}: No images found")
                return False
    
    return True

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
    print("🧪 Testing Quick Training Setup")
    print("=" * 40)
    
    # Test imports
    failed_required, failed_optional = test_imports()
    
    if failed_required:
        print(f"\n❌ Required packages missing: {', '.join(failed_required)}")
        print("Please run: python install_dependencies.py")
        return False
    
    # Test dataset
    dataset_ok = test_dataset_structure()
    
    # Test model building
    model_ok = test_model_building()
    
    print("\n" + "=" * 40)
    print("📋 Test Summary:")
    print(f"   Imports: {'✅' if not failed_required else '❌'}")
    print(f"   Dataset: {'✅' if dataset_ok else '❌'}")
    print(f"   Model Building: {'✅' if model_ok else '❌'}")
    
    if not failed_required and dataset_ok and model_ok:
        print("\n🎉 All tests passed! You can run quick training.")
        print("\nNext step:")
        print("Run: python train_enhanced_model.py")
        return True
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 