"""
Enhanced Preprocessing Module for Traffic Sign Recognition
Implements CLAHE, Denoising Autoencoder, Super-Resolution, Edge Detection
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.cluster import KMeans
from skimage import exposure, filters, morphology
from scipy import ndimage
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ImageEnhancer:
    """
    Comprehensive image enhancement class for traffic sign preprocessing
    """
    
    def __init__(self):
        self.denoising_autoencoder = None
        self.srgan_generator = None
        
    def apply_clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Apply Contrast Limited Adaptive Histogram Equalization
        
        Args:
            image: Input image (BGR or RGB)
            clip_limit: Clipping limit for histogram equalization
            tile_grid_size: Size of the neighborhood area
        
        Returns:
            Enhanced image with improved contrast
        """
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel only
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def apply_clahe_rgb(self, image, clip_limit=0.01):
        """
        Apply CLAHE using scikit-image (RGB version)
        """
        # Normalize to [0, 1] range
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
            
        # Apply adaptive histogram equalization
        enhanced = exposure.equalize_adapthist(image, clip_limit=clip_limit)
        
        # Convert back to [0, 255] range
        enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced
    
    def build_denoising_autoencoder(self, input_shape=(32, 32, 3)):
        """
        Build and compile denoising autoencoder architecture
        """
        # Encoder
        input_img = keras.Input(shape=input_shape)
        
        # Encoder layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Decoder layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        
        # Create and compile autoencoder
        autoencoder = keras.Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        
        self.denoising_autoencoder = autoencoder
        return autoencoder
    
    def train_denoising_autoencoder(self, x_train, epochs=50, batch_size=32, 
                                  noise_factor=0.1, validation_split=0.2):
        """
        Train the denoising autoencoder on traffic sign data
        """
        if self.denoising_autoencoder is None:
            self.build_denoising_autoencoder()
        
        # Normalize data
        x_train = x_train.astype('float32') / 255.0
        
        # Add noise to training data
        x_train_noisy = x_train + noise_factor * np.random.normal(0.0, 1.0, x_train.shape)
        x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
        
        # Train autoencoder
        history = self.denoising_autoencoder.fit(
            x_train_noisy, x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True,
            verbose=1
        )
        
        return history
    
    def apply_denoising(self, image):
        """
        Apply trained denoising autoencoder to image
        """
        if self.denoising_autoencoder is None:
            raise ValueError("Denoising autoencoder not trained yet!")
        
        # Preprocess image
        if len(image.shape) == 3:
            image_input = np.expand_dims(image.astype('float32') / 255.0, axis=0)
        else:
            image_input = image.astype('float32') / 255.0
        
        # Apply denoising
        denoised = self.denoising_autoencoder.predict(image_input, verbose=0)
        
        # Post-process
        denoised = (denoised[0] * 255).astype(np.uint8)
        
        return denoised
    
    def build_srgan_generator(self, input_shape=(32, 32, 3), upscale_factor=2):
        """
        Build simplified SRGAN generator for super-resolution
        """
        def residual_block(x, filters=64):
            """Residual block for generator"""
            shortcut = x
            
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.Add()([x, shortcut])
            return x
        
        def upsampling_block(x, filters=256):
            """Upsampling block"""
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.UpSampling2D(2)(x)
            x = layers.Activation('relu')(x)
            return x
        
        # Input
        input_layer = keras.Input(shape=input_shape)
        
        # Pre-residual block
        x = layers.Conv2D(64, 9, padding='same')(input_layer)
        x = layers.Activation('relu')(x)
        
        # Residual blocks
        for _ in range(5):
            x = residual_block(x)
        
        # Upsampling blocks
        for _ in range(int(np.log2(upscale_factor))):
            x = upsampling_block(x)
        
        # Final output
        output = layers.Conv2D(3, 9, padding='same', activation='tanh')(x)
        
        generator = keras.Model(input_layer, output)
        self.srgan_generator = generator
        
        return generator
    
    def apply_super_resolution(self, image, upscale_factor=2):
        """
        Apply super-resolution using simple interpolation or trained SRGAN
        """
        if self.srgan_generator is not None:
            # Use trained SRGAN
            if len(image.shape) == 3:
                image_input = np.expand_dims(image.astype('float32') / 127.5 - 1, axis=0)
            else:
                image_input = image.astype('float32') / 127.5 - 1
            
            sr_image = self.srgan_generator.predict(image_input, verbose=0)
            sr_image = (sr_image[0] + 1) * 127.5
            sr_image = np.clip(sr_image, 0, 255).astype(np.uint8)
            
            return sr_image
        else:
            # Use bicubic interpolation as fallback
            h, w = image.shape[:2]
            new_h, new_w = h * upscale_factor, w * upscale_factor
            sr_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            return sr_image
    
    def apply_edge_enhancement(self, image, method='unsharp_mask'):
        """
        Apply edge detection and sharpening filters
        
        Args:
            image: Input image
            method: Edge detection method ('laplacian', 'sobel', 'canny', 'unsharp_mask')
        """
        if method == 'laplacian':
            # Laplacian edge detection and sharpening
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            
            # Combine with original for sharpening
            enhanced = cv2.addWeighted(image, 0.8, 
                                     cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB), 0.2, 0)
            
        elif method == 'sobel':
            # Sobel edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = np.sqrt(sobelx**2 + sobely**2)
            sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
            
            # Combine with original
            enhanced = cv2.addWeighted(image, 0.8,
                                     cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2RGB), 0.2, 0)
            
        elif method == 'canny':
            # Canny edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Combine with original
            enhanced = cv2.addWeighted(image, 0.8,
                                     cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 0.2, 0)
            
        elif method == 'unsharp_mask':
            # Unsharp masking for sharpening
            gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
            enhanced = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
            
        else:
            raise ValueError(f"Unknown edge enhancement method: {method}")
        
        return enhanced
    
    def apply_combined_enhancement(self, image, techniques=['clahe', 'edge_enhancement']):
        """
        Apply multiple enhancement techniques in sequence
        
        Args:
            image: Input image
            techniques: List of techniques to apply in order
        """
        enhanced = image.copy()
        
        for technique in techniques:
            if technique == 'clahe':
                enhanced = self.apply_clahe_rgb(enhanced)
            elif technique == 'denoising' and self.denoising_autoencoder is not None:
                enhanced = self.apply_denoising(enhanced)
            elif technique == 'super_resolution':
                enhanced = self.apply_super_resolution(enhanced)
            elif technique == 'edge_enhancement':
                enhanced = self.apply_edge_enhancement(enhanced, method='unsharp_mask')
            
        return enhanced

def compare_enhancement_methods(image, enhancer):
    """
    Compare different enhancement methods visually
    """
    methods = {
        'Original': image,
        'CLAHE': enhancer.apply_clahe_rgb(image),
        'Edge Enhanced': enhancer.apply_edge_enhancement(image, 'unsharp_mask'),
        'Combined': enhancer.apply_combined_enhancement(image, ['clahe', 'edge_enhancement'])
    }
    
    if enhancer.denoising_autoencoder is not None:
        methods['Denoised'] = enhancer.apply_denoising(image)
        methods['Full Enhanced'] = enhancer.apply_combined_enhancement(
            image, ['denoising', 'clahe', 'edge_enhancement']
        )
    
    return methods

# Example usage and testing functions
def test_enhancement_pipeline():
    """
    Test the enhancement pipeline with sample data
    """
    # Create sample traffic sign image (simulate)
    sample_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    
    # Initialize enhancer
    enhancer = ImageEnhancer()
    
    # Test CLAHE
    clahe_enhanced = enhancer.apply_clahe_rgb(sample_image)
    print("CLAHE enhancement completed")
    
    # Test edge enhancement
    edge_enhanced = enhancer.apply_edge_enhancement(sample_image)
    print("Edge enhancement completed")
    
    # Test combined enhancement
    combined = enhancer.apply_combined_enhancement(sample_image)
    print("Combined enhancement completed")
    
    return enhancer, {
        'original': sample_image,
        'clahe': clahe_enhanced,
        'edge': edge_enhanced,
        'combined': combined
    }

if __name__ == "__main__":
    test_enhancement_pipeline()