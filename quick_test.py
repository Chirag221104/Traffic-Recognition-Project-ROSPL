"""
Quick Test Script for Enhanced Traffic Sign Recognition
- Verifies NumPy/OpenCV/TensorFlow/Keras versions
- Runs enhancement functions (CLAHE, edge, combined) on a synthetic image
- Builds baseline and enhanced CNNs and prints parameter counts
- Saves a 4-panel visualization as 'quick_test_results.png'
"""

import sys
import platform
import numpy as np
import matplotlib.pyplot as plt

def print_env_info():
    print("=== Environment Info ===")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except Exception as e:
        print(f"NumPy import failed: {e}")
    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except Exception as e:
        print(f"OpenCV import failed: {e}")
    try:
        import tensorflow as tf
        print(f"TensorFlow: {tf.__version__}")
    except Exception as e:
        print(f"TensorFlow import failed: {e}")
    try:
        import keras
        print(f"Keras: {keras.__version__}")
    except Exception as e:
        print(f"Keras import failed: {e}")
    print("========================\n")

def test_enhancements_and_models():
    from enhanced_preprocessing import ImageEnhancer
    from model_comparison import TrafficSignCNN

    print("=== Enhancement Sanity Check ===")
    # Make a synthetic 32x32 RGB image that vaguely resembles sign-like content
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)

    # Add a red circle to simulate a sign border
    try:
        import cv2
        cv2.circle(img, (16, 16), 10, (255, 0, 0), 2)
    except Exception:
        pass

    enhancer = ImageEnhancer()
    img_clahe = enhancer.apply_clahe_rgb(img, clip_limit=0.01)
    img_edge  = enhancer.apply_edge_enhancement(img, method='unsharp_mask')
    img_comb  = enhancer.apply_combined_enhancement(img, ['clahe', 'edge_enhancement'])
    print("✓ CLAHE, Edge, and Combined enhancements computed")

    print("\n=== CNN Build Sanity Check ===")
    cnn = TrafficSignCNN(num_classes=43, input_shape=(32, 32, 3))
    baseline_model = cnn.build_baseline_cnn()
    print(f"✓ Baseline CNN params: {baseline_model.count_params():,}")
    enhanced_model = cnn.build_enhanced_cnn()
    print(f"✓ Enhanced CNN params: {enhanced_model.count_params():,}")

    # Save visualization
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes.imshow(img);       axes.set_title("Original");         axes.axis("off")
    axes[21].imshow(img_clahe); axes[21].set_title("CLAHE");            axes[21].axis("off")
    axes[22].imshow(img_edge);  axes[22].set_title("Edge Enhanced");    axes[22].axis("off")
    axes[23].imshow(img_comb);  axes[23].set_title("Combined");         axes[23].axis("off")
    plt.tight_layout()
    out_path = "quick_test_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved enhancement visualization to {out_path}")

if __name__ == "__main__":
    print_env_info()
    try:
        test_enhancements_and_models()
        print("\nAll quick checks passed. You're ready to run the enhanced pipeline.")
        print("Try: python main_enhanced.py --data_dir data --demo_only")
    except Exception as e:
        print(f"\nQuick test failed: {e}")
        print("Check that enhanced_preprocessing.py and model_comparison.py are in this folder and imports succeed.")
