"""
Enhanced Traffic Sign Recognition - Main Execution Script
Integrates enhanced preprocessing with existing CNN architecture
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Import our enhanced modules
from enhanced_preprocessing import ImageEnhancer
from model_comparison import ModelComparator, TrafficSignCNN

class EnhancedTrafficSignRecognition:
    """
    Main class for enhanced traffic sign recognition system
    """
    
    def __init__(self, data_dir="data", output_dir="outputs"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.enhancer = ImageEnhancer()
        self.comparator = ModelComparator()
        
        # Data containers
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        
    def load_gtsrb_data(self):
        """
        Load GTSRB dataset from pickle files or CSV format
        """
        print("Loading GTSRB dataset...")
        
        # Try to load pickle files first (original format)
        train_file = self.data_dir / "train.p"
        test_file = self.data_dir / "test.p"
        
        if train_file.exists() and test_file.exists():
            print("Loading from pickle files...")
            with open(train_file, 'rb') as f:
                train_data = pickle.load(f)
            with open(test_file, 'rb') as f:
                test_data = pickle.load(f)
            
            x_train_full = train_data['features']
            y_train_full = train_data['labels']
            self.x_test = test_data['features']
            self.y_test = test_data['labels']
            
            # Split training data into train and validation
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
                x_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
            )
            
        else:
            print("Loading from CSV format...")
            self._load_from_csv()
        
        print(f"Dataset loaded successfully!")
        print(f"Training set: {self.x_train.shape}")
        print(f"Validation set: {self.x_val.shape}")
        print(f"Test set: {self.x_test.shape}")
        print(f"Number of classes: {len(np.unique(self.y_train))}")
        
    def _load_from_csv(self):
        """
        Load dataset from CSV files and image directories
        """
        train_csv = self.data_dir / "Train.csv"
        test_csv = self.data_dir / "Test.csv"
        
        if not train_csv.exists():
            raise FileNotFoundError("Neither pickle files nor CSV files found. Please check data directory.")
        
        # Load CSV files
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv) if test_csv.exists() else None
        
        # Load training images
        train_images = []
        train_labels = []
        
        print("Loading training images...")
        for idx, row in train_df.iterrows():
            img_path = self.data_dir / row['Path']
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (32, 32))  # Resize to standard size
                train_images.append(img)
                train_labels.append(row['ClassId'])
            
            if idx % 1000 == 0:
                print(f"Loaded {idx} training images...")
        
        x_train_full = np.array(train_images)
        y_train_full = np.array(train_labels)
        
        # Split training data
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )
        
        # Load test images if available
        if test_df is not None:
            test_images = []
            test_labels = []
            
            print("Loading test images...")
            for idx, row in test_df.iterrows():
                img_path = self.data_dir / row['Path']
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (32, 32))
                    test_images.append(img)
                    test_labels.append(row['ClassId'])
            
            self.x_test = np.array(test_images)
            self.y_test = np.array(test_labels)
        else:
            # Use a portion of training data as test set
            print("No separate test set found, using 10% of training data...")
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_train, self.y_train, test_size=0.1, random_state=42, stratify=self.y_train
            )
    
    def train_denoising_autoencoder(self, epochs=50):
        """
        Train denoising autoencoder for image enhancement
        """
        print("Training denoising autoencoder...")
        
        # Build and train autoencoder
        self.enhancer.build_denoising_autoencoder()
        history = self.enhancer.train_denoising_autoencoder(
            self.x_train, epochs=epochs, batch_size=32
        )
        
        # Save autoencoder model
        model_path = self.output_dir / "denoising_autoencoder.h5"
        self.enhancer.denoising_autoencoder.save(str(model_path))
        print(f"Denoising autoencoder saved to {model_path}")
        
        # Plot training history
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Denoising Autoencoder Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Show sample denoised images
        plt.subplot(1, 2, 2)
        sample_idx = np.random.randint(0, len(self.x_train), 1)[0]
        sample_img = self.x_train[sample_idx]
        denoised_img = self.enhancer.apply_denoising(sample_img)
        
        # Create comparison
        comparison = np.hstack([sample_img, denoised_img])
        plt.imshow(comparison)
        plt.title('Original (Left) vs Denoised (Right)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "autoencoder_training.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return history
    
    def demonstrate_enhancements(self, num_samples=5):
        """
        Create visual demonstration of different enhancement techniques
        """
        print("Creating enhancement demonstration...")
        
        # Select random samples
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        methods = ['Original', 'CLAHE', 'Edge Enhanced', 'Denoised', 'Combined']
        
        for i, idx in enumerate(indices):
            img = self.x_test[idx]
            
            # Apply different enhancements
            enhanced_images = {
                'Original': img,
                'CLAHE': self.enhancer.apply_clahe_rgb(img),
                'Edge Enhanced': self.enhancer.apply_edge_enhancement(img, 'unsharp_mask'),
                'Denoised': self.enhancer.apply_denoising(img) if self.enhancer.denoising_autoencoder else img,
                'Combined': self.enhancer.apply_combined_enhancement(img, ['clahe', 'edge_enhancement'])
            }
            
            for j, (method, enhanced_img) in enumerate(enhanced_images.items()):
                axes[i, j].imshow(enhanced_img)
                axes[i, j].set_title(f'{method}\nClass: {self.y_test[idx]}')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "enhancement_demonstration.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_comparison(self, epochs=20, enhancement_techniques=['clahe', 'edge_enhancement']):
        """
        Run complete comparison between baseline and enhanced models
        """
        print("\n" + "="*60)
        print("RUNNING COMPLETE MODEL COMPARISON")
        print("="*60)
        
        # Run comparison
        results = self.comparator.run_comparison(
            self.x_train, self.y_train,
            self.x_val, self.y_val,
            self.x_test, self.y_test,
            epochs=epochs,
            enhancement_techniques=enhancement_techniques
        )
        
        # Generate and display comparison report
        report = results['comparison_report']
        
        print("\n" + "="*40)
        print("FINAL RESULTS SUMMARY")
        print("="*40)
        print(f"Baseline CNN Accuracy: {report['accuracy_comparison']['baseline_accuracy']:.4f}")
        print(f"Enhanced CNN Accuracy: {report['accuracy_comparison']['enhanced_accuracy']:.4f}")
        print(f"Improvement: +{report['accuracy_comparison']['improvement']:.4f}")
        print(f"Improvement Percentage: +{report['accuracy_comparison']['improvement_percentage']:.2f}%")
        print("\nEnhancement Techniques Used:")
        for technique in enhancement_techniques:
            print(f"  - {technique}")
        
        # Create visualization
        fig = self.comparator.plot_comparison_results(results)
        plt.savefig(self.output_dir / "model_comparison_results.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save results
        self.comparator.save_comparison_results(results, str(self.output_dir / "comparison"))
        
        return results
    
    def generate_research_report(self, results):
        """
        Generate comprehensive research report for ROSPL submission
        """
        report_path = self.output_dir / "ROSPL_Research_Report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced Traffic Sign Recognition using Image Preprocessing\n")
            f.write("## Research Oriented Student Project Learning (ROSPL) Report\n\n")
            
            f.write("### Project Overview\n")
            f.write("This project enhances an existing open-source traffic sign recognition system ")
            f.write("by implementing advanced image preprocessing techniques before CNN classification.\n\n")
            
            f.write("### Base Project\n")
            f.write("- **Original Repository**: Traffic Sign Recognition using CNN (Keras)\n")
            f.write("- **Dataset**: German Traffic Sign Recognition Benchmark (GTSRB)\n")
            f.write("- **Base Architecture**: Convolutional Neural Network\n\n")
            
            f.write("### Enhancement Contributions\n")
            techniques_used = results['enhanced']['techniques']
            for i, technique in enumerate(techniques_used, 1):
                f.write(f"{i}. **{technique.upper()}**: ")
                
                if technique == 'clahe':
                    f.write("Contrast Limited Adaptive Histogram Equalization for improving low-light/low-contrast images\n")
                elif technique == 'denoising':
                    f.write("Denoising Autoencoder for noise reduction in blurry or corrupted images\n")
                elif technique == 'super_resolution':
                    f.write("Super-Resolution GAN (SRGAN) for enhancing small or low-resolution signs\n")
                elif technique == 'edge_enhancement':
                    f.write("Edge detection and sharpening filters for clearer symbol/text recognition\n")
            
            f.write("\n### Experimental Results\n")
            report_data = results['comparison_report']
            f.write(f"- **Baseline CNN Accuracy**: {report_data['accuracy_comparison']['baseline_accuracy']:.4f}\n")
            f.write(f"- **Enhanced CNN Accuracy**: {report_data['accuracy_comparison']['enhanced_accuracy']:.4f}\n")
            f.write(f"- **Accuracy Improvement**: +{report_data['accuracy_comparison']['improvement']:.4f} ")
            f.write(f"({report_data['accuracy_comparison']['improvement_percentage']:.2f}%)\n\n")
            
            f.write("### Technical Implementation\n")
            f.write("1. **Enhanced Preprocessing Pipeline**: Implemented modular enhancement system\n")
            f.write("2. **Model Architecture**: Enhanced CNN with additional preprocessing layers\n")
            f.write("3. **Comparison Framework**: Systematic evaluation of baseline vs enhanced models\n")
            f.write("4. **Performance Metrics**: Comprehensive accuracy, timing, and visual comparisons\n\n")
            
            f.write("### Conclusion\n")
            if report_data['accuracy_comparison']['improvement'] > 0:
                f.write("The enhanced preprocessing pipeline successfully improved traffic sign recognition accuracy. ")
                f.write("The image enhancement techniques help the CNN model better recognize traffic signs ")
                f.write("in challenging conditions such as low contrast, noise, or small size.\n\n")
            else:
                f.write("While the enhanced pipeline did not show significant improvement in this experiment, ")
                f.write("the framework provides a solid foundation for further research and optimization. ")
                f.write("Different enhancement parameters or additional techniques may yield better results.\n\n")
            
            f.write("### Future Work\n")
            f.write("1. Parameter optimization for enhancement techniques\n")
            f.write("2. Integration with real-time traffic sign detection systems\n")
            f.write("3. Evaluation on additional datasets and challenging conditions\n")
            f.write("4. Investigation of other advanced preprocessing techniques\n")
        
        print(f"Research report generated: {report_path}")
        return report_path

def main():
    """
    Main execution function with command-line arguments
    """
    parser = argparse.ArgumentParser(description='Enhanced Traffic Sign Recognition')
    parser.add_argument('--data_dir', default='data', help='Path to data directory')
    parser.add_argument('--output_dir', default='outputs', help='Path to output directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--train_autoencoder', action='store_true', help='Train denoising autoencoder')
    parser.add_argument('--autoencoder_epochs', type=int, default=50, help='Autoencoder training epochs')
    parser.add_argument('--techniques', nargs='+', default=['clahe', 'edge_enhancement'],
                       choices=['clahe', 'denoising', 'super_resolution', 'edge_enhancement'],
                       help='Enhancement techniques to use')
    parser.add_argument('--demo_only', action='store_true', help='Run enhancement demonstration only')
    
    args = parser.parse_args()
    
    # Initialize system
    system = EnhancedTrafficSignRecognition(args.data_dir, args.output_dir)
    
    try:
        # Load data
        system.load_gtsrb_data()
        
        # Train autoencoder if requested
        if args.train_autoencoder and 'denoising' in args.techniques:
            system.train_denoising_autoencoder(args.autoencoder_epochs)
        
        # Demonstration of enhancement techniques
        system.demonstrate_enhancements()
        
        if not args.demo_only:
            # Run complete comparison
            results = system.run_complete_comparison(args.epochs, args.techniques)
            
            # Generate research report
            system.generate_research_report(results)
            
            print("\n" + "="*60)
            print("ENHANCED TRAFFIC SIGN RECOGNITION COMPLETED!")
            print("="*60)
            print(f"Results saved in: {args.output_dir}")
            print("Files generated:")
            print("  - model_comparison_results.png")
            print("  - comparison_summary.csv")
            print("  - comparison_detailed_report.txt")
            print("  - enhancement_demonstration.png")
            print("  - ROSPL_Research_Report.md")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check your data directory and file paths.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())