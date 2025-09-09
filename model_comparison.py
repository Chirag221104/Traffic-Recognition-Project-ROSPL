"""
Model Comparison Module for Traffic Sign Recognition
Compares baseline CNN vs enhanced preprocessing pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import pandas as pd
from enhanced_preprocessing import ImageEnhancer

class TrafficSignCNN:
    """
    CNN Architecture for Traffic Sign Classification
    """
    
    def __init__(self, num_classes=43, input_shape=(32, 32, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        
    def build_baseline_cnn(self):
        """
        Build baseline CNN architecture (similar to original project)
        """
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_enhanced_cnn(self):
        """
        Build enhanced CNN architecture with additional layers for processed images
        """
        model = keras.Sequential([
            # Additional preprocessing layers
            layers.Conv2D(16, (1, 1), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            
            # Main CNN architecture
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, x_train, y_train, x_val, y_val, epochs=20, batch_size=64):
        """
        Train the CNN model
        """
        if self.model is None:
            raise ValueError("Model not built yet! Call build_baseline_cnn() or build_enhanced_cnn() first.")
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001
        )
        
        # Train model
        start_time = time.time()
        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        training_time = time.time() - start_time
        
        return history, training_time
    
    def evaluate_model(self, x_test, y_test):
        """
        Evaluate model performance
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Predictions
        start_time = time.time()
        y_pred_proba = self.model.predict(x_test)
        inference_time = time.time() - start_time
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'inference_time': inference_time
        }
        
        return results

class ModelComparator:
    """
    Compare baseline and enhanced model performance
    """
    
    def __init__(self, num_classes=43):
        self.num_classes = num_classes
        self.baseline_model = TrafficSignCNN(num_classes)
        self.enhanced_model = TrafficSignCNN(num_classes)
        self.enhancer = ImageEnhancer()
        
    def preprocess_data_baseline(self, images):
        """
        Basic preprocessing for baseline model (normalization only)
        """
        processed = images.astype('float32') / 255.0
        return processed
    
    def preprocess_data_enhanced(self, images, enhancement_techniques=['clahe', 'edge_enhancement']):
        """
        Enhanced preprocessing pipeline
        """
        enhanced_images = []
        
        for img in images:
            # Convert to proper format if needed
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            # Apply enhancement techniques
            enhanced = self.enhancer.apply_combined_enhancement(img, enhancement_techniques)
            
            # Normalize
            enhanced = enhanced.astype('float32') / 255.0
            enhanced_images.append(enhanced)
        
        return np.array(enhanced_images)
    
    def run_comparison(self, x_train, y_train, x_val, y_val, x_test, y_test, 
                      epochs=20, enhancement_techniques=['clahe', 'edge_enhancement']):
        """
        Run complete comparison between baseline and enhanced models
        """
        results = {}
        
        print("=== BASELINE MODEL TRAINING ===")
        # Baseline model
        self.baseline_model.build_baseline_cnn()
        
        # Preprocess data for baseline
        x_train_baseline = self.preprocess_data_baseline(x_train)
        x_val_baseline = self.preprocess_data_baseline(x_val)
        x_test_baseline = self.preprocess_data_baseline(x_test)
        
        # Train baseline
        baseline_history, baseline_train_time = self.baseline_model.train_model(
            x_train_baseline, y_train, x_val_baseline, y_val, epochs=epochs
        )
        
        # Evaluate baseline
        baseline_results = self.baseline_model.evaluate_model(x_test_baseline, y_test)
        results['baseline'] = {
            'model': self.baseline_model,
            'history': baseline_history,
            'results': baseline_results,
            'training_time': baseline_train_time,
            'techniques': ['normalization_only']
        }
        
        print("\n=== ENHANCED MODEL TRAINING ===")
        # Enhanced model
        self.enhanced_model.build_enhanced_cnn()
        
        # Preprocess data for enhanced model
        print("Applying image enhancement techniques...")
        x_train_enhanced = self.preprocess_data_enhanced(x_train, enhancement_techniques)
        x_val_enhanced = self.preprocess_data_enhanced(x_val, enhancement_techniques)
        x_test_enhanced = self.preprocess_data_enhanced(x_test, enhancement_techniques)
        
        # Train enhanced model
        enhanced_history, enhanced_train_time = self.enhanced_model.train_model(
            x_train_enhanced, y_train, x_val_enhanced, y_val, epochs=epochs
        )
        
        # Evaluate enhanced model
        enhanced_results = self.enhanced_model.evaluate_model(x_test_enhanced, y_test)
        results['enhanced'] = {
            'model': self.enhanced_model,
            'history': enhanced_history,
            'results': enhanced_results,
            'training_time': enhanced_train_time,
            'techniques': enhancement_techniques
        }
        
        # Generate comparison report
        comparison_report = self.generate_comparison_report(results)
        results['comparison_report'] = comparison_report
        
        return results
    
    def generate_comparison_report(self, results):
        """
        Generate detailed comparison report
        """
        baseline_acc = results['baseline']['results']['accuracy']
        enhanced_acc = results['enhanced']['results']['accuracy']
        
        improvement = enhanced_acc - baseline_acc
        improvement_pct = (improvement / baseline_acc) * 100
        
        baseline_time = results['baseline']['training_time']
        enhanced_time = results['enhanced']['training_time']
        
        baseline_inference = results['baseline']['results']['inference_time']
        enhanced_inference = results['enhanced']['results']['inference_time']
        
        report = {
            'accuracy_comparison': {
                'baseline_accuracy': baseline_acc,
                'enhanced_accuracy': enhanced_acc,
                'improvement': improvement,
                'improvement_percentage': improvement_pct
            },
            'timing_comparison': {
                'baseline_training_time': baseline_time,
                'enhanced_training_time': enhanced_time,
                'baseline_inference_time': baseline_inference,
                'enhanced_inference_time': enhanced_inference
            },
            'techniques_used': {
                'baseline': results['baseline']['techniques'],
                'enhanced': results['enhanced']['techniques']
            }
        }
        
        return report
    
    def plot_comparison_results(self, results, save_path=None):
        """
        Create visualization comparing model performances
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training history comparison
        baseline_history = results['baseline']['history']
        enhanced_history = results['enhanced']['history']
        
        # Accuracy comparison
        axes[0, 0].plot(baseline_history.history['accuracy'], label='Baseline Train', color='blue')
        axes[0, 0].plot(baseline_history.history['val_accuracy'], label='Baseline Val', color='blue', linestyle='--')
        axes[0, 0].plot(enhanced_history.history['accuracy'], label='Enhanced Train', color='red')
        axes[0, 0].plot(enhanced_history.history['val_accuracy'], label='Enhanced Val', color='red', linestyle='--')
        axes[0, 0].set_title('Training Accuracy Comparison')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss comparison
        axes[0, 1].plot(baseline_history.history['loss'], label='Baseline Train', color='blue')
        axes[0, 1].plot(baseline_history.history['val_loss'], label='Baseline Val', color='blue', linestyle='--')
        axes[0, 1].plot(enhanced_history.history['loss'], label='Enhanced Train', color='red')
        axes[0, 1].plot(enhanced_history.history['val_loss'], label='Enhanced Val', color='red', linestyle='--')
        axes[0, 1].set_title('Training Loss Comparison')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Final accuracy comparison bar chart
        accuracies = [
            results['baseline']['results']['accuracy'],
            results['enhanced']['results']['accuracy']
        ]
        models = ['Baseline CNN', 'Enhanced CNN']
        colors = ['skyblue', 'lightcoral']
        
        bars = axes[1, 0].bar(models, accuracies, color=colors)
        axes[1, 0].set_title('Final Test Accuracy Comparison')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.4f}', ha='center', va='bottom')
        
        # Improvement visualization
        improvement = results['comparison_report']['accuracy_comparison']['improvement']
        improvement_pct = results['comparison_report']['accuracy_comparison']['improvement_percentage']
        
        axes[1, 1].bar(['Improvement'], [improvement], color='green', alpha=0.7)
        axes[1, 1].set_title(f'Accuracy Improvement: +{improvement:.4f} ({improvement_pct:.2f}%)')
        axes[1, 1].set_ylabel('Improvement')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_comparison_results(self, results, save_path):
        """
        Save comparison results to CSV and text files
        """
        # Create summary DataFrame
        summary_data = {
            'Model': ['Baseline CNN', 'Enhanced CNN'],
            'Test_Accuracy': [
                results['baseline']['results']['accuracy'],
                results['enhanced']['results']['accuracy']
            ],
            'Training_Time_Seconds': [
                results['baseline']['training_time'],
                results['enhanced']['training_time']
            ],
            'Inference_Time_Seconds': [
                results['baseline']['results']['inference_time'],
                results['enhanced']['results']['inference_time']
            ],
            'Enhancement_Techniques': [
                str(results['baseline']['techniques']),
                str(results['enhanced']['techniques'])
            ]
        }
        
        df = pd.DataFrame(summary_data)
        df.to_csv(f"{save_path}_summary.csv", index=False)
        
        # Save detailed report
        report = results['comparison_report']
        with open(f"{save_path}_detailed_report.txt", 'w') as f:
            f.write("TRAFFIC SIGN RECOGNITION - ENHANCEMENT COMPARISON REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("ACCURACY COMPARISON:\n")
            f.write(f"Baseline CNN Accuracy: {report['accuracy_comparison']['baseline_accuracy']:.4f}\n")
            f.write(f"Enhanced CNN Accuracy: {report['accuracy_comparison']['enhanced_accuracy']:.4f}\n")
            f.write(f"Improvement: +{report['accuracy_comparison']['improvement']:.4f}\n")
            f.write(f"Improvement Percentage: +{report['accuracy_comparison']['improvement_percentage']:.2f}%\n\n")
            
            f.write("TIMING COMPARISON:\n")
            f.write(f"Baseline Training Time: {report['timing_comparison']['baseline_training_time']:.2f} seconds\n")
            f.write(f"Enhanced Training Time: {report['timing_comparison']['enhanced_training_time']:.2f} seconds\n")
            f.write(f"Baseline Inference Time: {report['timing_comparison']['baseline_inference_time']:.4f} seconds\n")
            f.write(f"Enhanced Inference Time: {report['timing_comparison']['enhanced_inference_time']:.4f} seconds\n\n")
            
            f.write("ENHANCEMENT TECHNIQUES:\n")
            f.write(f"Baseline: {report['techniques_used']['baseline']}\n")
            f.write(f"Enhanced: {report['techniques_used']['enhanced']}\n")
        
        print(f"Results saved to {save_path}_summary.csv and {save_path}_detailed_report.txt")

# Example usage function
def demo_comparison():
    """
    Demonstration of the comparison functionality with synthetic data
    """
    # Create synthetic data for demonstration
    num_samples = 1000
    x_train = np.random.randint(0, 255, (num_samples, 32, 32, 3), dtype=np.uint8)
    y_train = np.random.randint(0, 43, num_samples)
    x_val = np.random.randint(0, 255, (200, 32, 32, 3), dtype=np.uint8)
    y_val = np.random.randint(0, 43, 200)
    x_test = np.random.randint(0, 255, (200, 32, 32, 3), dtype=np.uint8)
    y_test = np.random.randint(0, 43, 200)
    
    # Run comparison
    comparator = ModelComparator()
    results = comparator.run_comparison(
        x_train, y_train, x_val, y_val, x_test, y_test, 
        epochs=3,  # Small number for demo
        enhancement_techniques=['clahe', 'edge_enhancement']
    )
    
    return comparator, results

if __name__ == "__main__":
    comparator, results = demo_comparison()
    print("Demo comparison completed!")