#!/usr/bin/env python3
"""
Dependency Installation and Environment Check Script
This script helps install required dependencies and verify the setup.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package_name):
    """Check if a package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    print("🔧 Setting up Enhanced Garbage Classification Environment")
    print("=" * 60)
    
    # Required packages (core dependencies)
    required_packages = [
        "tensorflow>=2.10.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0"
    ]
    
    # Optional packages (enhanced features)
    optional_packages = [
        "tensorflow-addons>=0.19.0",
        "albumentations>=1.3.0",
        "gradio>=3.20.0"
    ]
    
    print("\n📦 Installing required packages...")
    failed_required = []
    
    for package in required_packages:
        package_name = package.split(">=")[0].split("==")[0]
        print(f"Installing {package}...", end=" ")
        
        if check_package(package_name):
            print("✅ Already installed")
        else:
            if install_package(package):
                print("✅ Installed successfully")
            else:
                print("❌ Failed to install")
                failed_required.append(package)
    
    if failed_required:
        print(f"\n❌ Failed to install required packages: {', '.join(failed_required)}")
        print("Please install them manually using: pip install <package_name>")
        return
    
    print("\n📦 Installing optional packages...")
    failed_optional = []
    
    for package in optional_packages:
        package_name = package.split(">=")[0].split("==")[0]
        print(f"Installing {package}...", end=" ")
        
        if check_package(package_name):
            print("✅ Already installed")
        else:
            if install_package(package):
                print("✅ Installed successfully")
            else:
                print("⚠️  Failed to install (optional)")
                failed_optional.append(package)
    
    print("\n🔍 Checking environment...")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   GPU available: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"   - {gpu.name}")
        else:
            print("   GPU not available, will use CPU")
    except ImportError:
        print("❌ TensorFlow not available")
    
    # Check OpenCV
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not available")
    
    # Check Albumentations
    try:
        import albumentations as A
        print(f"✅ Albumentations {A.__version__}")
    except ImportError:
        print("⚠️  Albumentations not available (will use basic augmentation)")
    
    # Check TensorFlow Addons
    try:
        import tensorflow_addons as tfa
        print(f"✅ TensorFlow Addons {tfa.__version__}")
    except ImportError:
        print("⚠️  TensorFlow Addons not available (will use standard optimizer)")
    
    # Check Gradio
    try:
        import gradio as gr
        print(f"✅ Gradio {gr.__version__}")
    except ImportError:
        print("⚠️  Gradio not available (will skip web interface)")
    
    # Check dataset
    dataset_dir = "TrashType_Image_Dataset"
    if os.path.exists(dataset_dir):
        print(f"✅ Dataset found at {dataset_dir}")
        
        # Check class directories
        classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        for split in ['train', 'test']:
            split_dir = os.path.join(dataset_dir, split)
            if os.path.exists(split_dir):
                print(f"   {split}: ", end="")
                for class_name in classes:
                    class_dir = os.path.join(split_dir, class_name)
                    if os.path.exists(class_dir):
                        count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
                        print(f"{class_name}({count}) ", end="")
                print()
            else:
                print(f"   {split}: Not found")
    else:
        print(f"❌ Dataset not found at {dataset_dir}")
        print("   Please ensure the dataset is in the correct location")
    
    print("\n🚀 Environment setup complete!")
    print("\nNext steps:")
    print("1. Run 'python garbage_classification_fixed.py' for the enhanced model")
    print("2. Run 'python train_enhanced_model.py' for quick testing")
    print("3. Check README.md for detailed instructions")
    
    if failed_optional:
        print(f"\n⚠️  Optional packages not installed: {', '.join(failed_optional)}")
        print("The model will work with basic features. Install these for enhanced functionality.")

if __name__ == "__main__":
    main() 