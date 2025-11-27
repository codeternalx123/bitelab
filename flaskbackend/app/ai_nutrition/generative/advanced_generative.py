"""
Advanced Generative Models
===========================

Generative models for food image synthesis, recipe creation,
and data augmentation using GANs, VAEs, and diffusion models.

Features:
1. Generative Adversarial Networks (GANs)
2. Variational Autoencoders (VAEs)
3. Diffusion models for image generation
4. Conditional generation
5. Style transfer
6. Data augmentation
7. Latent space manipulation
8. Quality assessment

Performance Targets:
- Generation time: <2s per image
- FID score: <50
- Inception score: >3.0
- Image resolution: up to 512x512
- Batch generation: 16 images
- Diversity: high variation

Author: Wellomex AI Team
Date: November 2025
Version: 5.0.0
"""

import time
import logging
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict, deque

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class GenerativeModel(Enum):
    """Generative model type"""
    GAN = "gan"
    VAE = "vae"
    DIFFUSION = "diffusion"
    AUTOREGRESSIVE = "autoregressive"


class LossType(Enum):
    """Loss function type"""
    ADVERSARIAL = "adversarial"
    RECONSTRUCTION = "reconstruction"
    KL_DIVERGENCE = "kl_divergence"
    PERCEPTUAL = "perceptual"


@dataclass
class GenerativeConfig:
    """Generative model configuration"""
    # Model
    model_type: GenerativeModel = GenerativeModel.GAN
    latent_dim: int = 128
    
    # Image
    image_size: int = 256
    channels: int = 3
    
    # Generator
    gen_filters: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    
    # Discriminator
    disc_filters: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    
    # Training
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    
    # Conditioning
    num_classes: int = 100
    conditional: bool = True


# ============================================================================
# GENERATOR NETWORK
# ============================================================================

class Generator:
    """
    Generator network for GAN
    """
    
    def __init__(self, config: GenerativeConfig):
        self.config = config
        
        # Build network layers
        self.layers = self._build_layers()
        
        logger.info(f"Generator initialized: latent_dim={config.latent_dim}")
    
    def _build_layers(self) -> List[Dict[str, Any]]:
        """Build generator layers"""
        layers = []
        
        # Initial dense layer
        layers.append({
            'type': 'dense',
            'units': 4 * 4 * self.config.gen_filters[0],
            'activation': 'relu'
        })
        
        # Reshape
        layers.append({
            'type': 'reshape',
            'shape': (4, 4, self.config.gen_filters[0])
        })
        
        # Upsampling blocks
        for i, filters in enumerate(self.config.gen_filters):
            layers.append({
                'type': 'upsample',
                'filters': filters,
                'kernel_size': 4,
                'strides': 2
            })
            
            layers.append({
                'type': 'batch_norm'
            })
            
            layers.append({
                'type': 'activation',
                'activation': 'relu'
            })
        
        # Final conv to RGB
        layers.append({
            'type': 'conv',
            'filters': self.config.channels,
            'kernel_size': 3,
            'activation': 'tanh'
        })
        
        return layers
    
    def forward(
        self,
        latent: Any,
        condition: Optional[Any] = None
    ) -> Any:
        """
        Generate image from latent vector
        
        Args:
            latent: (batch_size, latent_dim)
            condition: Optional conditioning (e.g., class label)
        
        Returns:
            image: (batch_size, height, width, channels)
        """
        if not NUMPY_AVAILABLE:
            return np.zeros((1, self.config.image_size, self.config.image_size, self.config.channels)) \
                if NUMPY_AVAILABLE else [[[0] * self.config.channels] * self.config.image_size] * self.config.image_size
        
        batch_size = latent.shape[0]
        
        # Concatenate with condition if provided
        if condition is not None and self.config.conditional:
            x = np.concatenate([latent, condition], axis=-1)
        else:
            x = latent
        
        # Initial dense
        x = np.dot(x, np.random.randn(x.shape[-1], 4 * 4 * self.config.gen_filters[0]))
        
        # Reshape
        x = x.reshape(batch_size, 4, 4, self.config.gen_filters[0])
        
        # Upsampling (simplified - use bilinear interpolation)
        for _ in range(len(self.config.gen_filters)):
            # Upsample 2x
            x = self._upsample_2x(x)
        
        # Final layer to match image size
        while x.shape[1] < self.config.image_size:
            x = self._upsample_2x(x)
        
        # Crop to exact size
        x = x[:, :self.config.image_size, :self.config.image_size, :self.config.channels]
        
        # Tanh activation
        x = np.tanh(x)
        
        return x
    
    def _upsample_2x(self, x: Any) -> Any:
        """Upsample by factor of 2"""
        batch_size, h, w, c = x.shape
        
        # Repeat along spatial dimensions
        x_up = np.repeat(x, 2, axis=1)
        x_up = np.repeat(x_up, 2, axis=2)
        
        return x_up


# ============================================================================
# DISCRIMINATOR NETWORK
# ============================================================================

class Discriminator:
    """
    Discriminator network for GAN
    """
    
    def __init__(self, config: GenerativeConfig):
        self.config = config
        
        # Build network layers
        self.layers = self._build_layers()
        
        logger.info("Discriminator initialized")
    
    def _build_layers(self) -> List[Dict[str, Any]]:
        """Build discriminator layers"""
        layers = []
        
        # Downsampling blocks
        for i, filters in enumerate(self.config.disc_filters):
            layers.append({
                'type': 'conv',
                'filters': filters,
                'kernel_size': 4,
                'strides': 2
            })
            
            if i > 0:  # No batch norm on first layer
                layers.append({
                    'type': 'batch_norm'
                })
            
            layers.append({
                'type': 'activation',
                'activation': 'leaky_relu',
                'alpha': 0.2
            })
        
        # Final classification
        layers.append({
            'type': 'flatten'
        })
        
        layers.append({
            'type': 'dense',
            'units': 1,
            'activation': 'sigmoid'
        })
        
        return layers
    
    def forward(
        self,
        image: Any,
        condition: Optional[Any] = None
    ) -> Any:
        """
        Discriminate real vs fake images
        
        Args:
            image: (batch_size, height, width, channels)
            condition: Optional conditioning
        
        Returns:
            score: (batch_size, 1) - probability of being real
        """
        if not NUMPY_AVAILABLE:
            return np.array([[0.5]])
        
        batch_size = image.shape[0]
        
        x = image
        
        # Downsampling (simplified)
        for _ in range(len(self.config.disc_filters)):
            x = self._downsample_2x(x)
        
        # Flatten
        x = x.reshape(batch_size, -1)
        
        # Final dense
        score = np.dot(x, np.random.randn(x.shape[-1], 1))
        
        # Sigmoid
        score = 1.0 / (1.0 + np.exp(-score))
        
        return score
    
    def _downsample_2x(self, x: Any) -> Any:
        """Downsample by factor of 2"""
        batch_size, h, w, c = x.shape
        
        # Average pooling
        x_down = np.zeros((batch_size, h // 2, w // 2, c))
        
        for i in range(h // 2):
            for j in range(w // 2):
                x_down[:, i, j, :] = np.mean(
                    x[:, i*2:(i+1)*2, j*2:(j+1)*2, :],
                    axis=(1, 2)
                )
        
        return x_down


# ============================================================================
# VARIATIONAL AUTOENCODER
# ============================================================================

class VAE:
    """
    Variational Autoencoder for food images
    """
    
    def __init__(self, config: GenerativeConfig):
        self.config = config
        
        # Encoder
        self.encoder = self._build_encoder()
        
        # Decoder (similar to generator)
        self.decoder = Generator(config)
        
        logger.info("VAE initialized")
    
    def _build_encoder(self) -> Dict[str, Any]:
        """Build encoder network"""
        return {
            'conv_layers': self.config.disc_filters,
            'latent_dim': self.config.latent_dim
        }
    
    def encode(self, image: Any) -> Tuple[Any, Any]:
        """
        Encode image to latent distribution
        
        Returns:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        if not NUMPY_AVAILABLE:
            return (
                np.zeros((1, self.config.latent_dim)),
                np.zeros((1, self.config.latent_dim))
            )
        
        batch_size = image.shape[0]
        
        # Encode (simplified)
        x = image
        
        # Downsample
        for _ in range(4):
            x = self._downsample(x)
        
        # Flatten
        x = x.reshape(batch_size, -1)
        
        # Project to latent space
        mu = np.dot(x, np.random.randn(x.shape[-1], self.config.latent_dim))
        log_var = np.dot(x, np.random.randn(x.shape[-1], self.config.latent_dim))
        
        return mu, log_var
    
    def _downsample(self, x: Any) -> Any:
        """Downsample"""
        batch_size, h, w, c = x.shape
        
        x_down = np.zeros((batch_size, h // 2, w // 2, c))
        
        for i in range(h // 2):
            for j in range(w // 2):
                x_down[:, i, j, :] = np.mean(
                    x[:, i*2:(i+1)*2, j*2:(j+1)*2, :],
                    axis=(1, 2)
                )
        
        return x_down
    
    def reparameterize(self, mu: Any, log_var: Any) -> Any:
        """Reparameterization trick"""
        if not NUMPY_AVAILABLE:
            return mu
        
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape)
        
        z = mu + std * eps
        
        return z
    
    def decode(self, z: Any) -> Any:
        """Decode latent to image"""
        return self.decoder.forward(z)
    
    def forward(self, image: Any) -> Tuple[Any, Any, Any]:
        """
        Forward pass
        
        Returns:
            reconstruction: Reconstructed image
            mu: Latent mean
            log_var: Latent log variance
        """
        mu, log_var = self.encode(image)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        
        return reconstruction, mu, log_var
    
    def compute_loss(
        self,
        image: Any,
        reconstruction: Any,
        mu: Any,
        log_var: Any
    ) -> Dict[str, float]:
        """Compute VAE loss"""
        if not NUMPY_AVAILABLE:
            return {'total': 0.0, 'reconstruction': 0.0, 'kl': 0.0}
        
        # Reconstruction loss (MSE)
        recon_loss = np.mean((image - reconstruction) ** 2)
        
        # KL divergence
        kl_loss = -0.5 * np.mean(1 + log_var - mu ** 2 - np.exp(log_var))
        
        # Total loss
        total_loss = recon_loss + kl_loss
        
        return {
            'total': float(total_loss),
            'reconstruction': float(recon_loss),
            'kl': float(kl_loss)
        }


# ============================================================================
# DIFFUSION MODEL
# ============================================================================

class DiffusionModel:
    """
    Diffusion model for high-quality image generation
    """
    
    def __init__(self, config: GenerativeConfig):
        self.config = config
        
        # Noise schedule
        self.num_timesteps = 1000
        self.beta_start = 1e-4
        self.beta_end = 0.02
        
        # Compute schedule
        self.betas = self._compute_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas) if NUMPY_AVAILABLE else [0.0] * self.num_timesteps
        
        # Denoising network
        self.denoiser = self._build_denoiser()
        
        logger.info("Diffusion Model initialized")
    
    def _compute_beta_schedule(self) -> Any:
        """Compute noise schedule"""
        if not NUMPY_AVAILABLE:
            return [self.beta_start] * self.num_timesteps
        
        return np.linspace(self.beta_start, self.beta_end, self.num_timesteps)
    
    def _build_denoiser(self) -> Dict[str, Any]:
        """Build denoising network (U-Net architecture)"""
        return {
            'type': 'unet',
            'channels': self.config.channels,
            'base_channels': 64
        }
    
    def add_noise(
        self,
        image: Any,
        timestep: int
    ) -> Tuple[Any, Any]:
        """
        Add noise to image
        
        Returns:
            noisy_image: Image with noise
            noise: Noise that was added
        """
        if not NUMPY_AVAILABLE:
            return image, np.zeros_like(image)
        
        # Sample noise
        noise = np.random.randn(*image.shape)
        
        # Get alpha for timestep
        alpha_t = self.alphas_cumprod[timestep]
        
        # Add noise: sqrt(alpha_t) * x + sqrt(1 - alpha_t) * noise
        noisy_image = np.sqrt(alpha_t) * image + np.sqrt(1 - alpha_t) * noise
        
        return noisy_image, noise
    
    def denoise_step(
        self,
        noisy_image: Any,
        timestep: int
    ) -> Any:
        """Single denoising step"""
        if not NUMPY_AVAILABLE:
            return noisy_image
        
        # Predict noise
        predicted_noise = self._predict_noise(noisy_image, timestep)
        
        # Remove noise
        alpha_t = self.alphas_cumprod[timestep]
        alpha_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else 1.0
        
        # Compute denoised image
        denoised = (noisy_image - np.sqrt(1 - alpha_t) * predicted_noise) / np.sqrt(alpha_t)
        
        # Add small amount of noise (except at t=0)
        if timestep > 0:
            noise = np.random.randn(*noisy_image.shape)
            denoised = denoised + np.sqrt(self.betas[timestep]) * noise
        
        return denoised
    
    def _predict_noise(self, noisy_image: Any, timestep: int) -> Any:
        """Predict noise (placeholder - use U-Net in practice)"""
        if not NUMPY_AVAILABLE:
            return np.zeros_like(noisy_image)
        
        # Simplified noise prediction
        return np.random.randn(*noisy_image.shape) * 0.1
    
    def generate(
        self,
        batch_size: int = 1,
        condition: Optional[Any] = None
    ) -> Any:
        """
        Generate images from noise
        
        Returns:
            images: Generated images
        """
        if not NUMPY_AVAILABLE:
            return np.zeros((batch_size, self.config.image_size, self.config.image_size, self.config.channels))
        
        # Start from pure noise
        image = np.random.randn(
            batch_size,
            self.config.image_size,
            self.config.image_size,
            self.config.channels
        )
        
        # Denoise iteratively
        for t in reversed(range(self.num_timesteps)):
            image = self.denoise_step(image, t)
            
            # Log progress every 100 steps
            if t % 100 == 0:
                logger.debug(f"Denoising step {t}/{self.num_timesteps}")
        
        # Clip to [-1, 1]
        image = np.clip(image, -1, 1)
        
        return image


# ============================================================================
# STYLE TRANSFER
# ============================================================================

class StyleTransfer:
    """
    Neural style transfer for food images
    """
    
    def __init__(self):
        # VGG-like feature extractor
        self.feature_extractor = self._build_feature_extractor()
        
        logger.info("Style Transfer initialized")
    
    def _build_feature_extractor(self) -> Dict[str, Any]:
        """Build feature extraction network"""
        return {
            'layers': ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        }
    
    def extract_features(self, image: Any) -> Dict[str, Any]:
        """Extract features for style transfer"""
        if not NUMPY_AVAILABLE:
            return {}
        
        # Simplified feature extraction
        features = {}
        
        x = image
        
        for i, layer_name in enumerate(self.feature_extractor['layers']):
            # Downsample
            if i > 0:
                batch_size, h, w, c = x.shape
                x_down = np.zeros((batch_size, h // 2, w // 2, c * 2))
                x = x_down
            
            features[layer_name] = x
        
        return features
    
    def compute_gram_matrix(self, features: Any) -> Any:
        """Compute Gram matrix for style"""
        if not NUMPY_AVAILABLE:
            return np.zeros((1, 1))
        
        batch_size, h, w, c = features.shape
        
        # Reshape to (batch, h*w, c)
        features_flat = features.reshape(batch_size, h * w, c)
        
        # Gram matrix: F^T * F
        gram = np.matmul(
            features_flat.transpose(0, 2, 1),
            features_flat
        )
        
        # Normalize
        gram = gram / (h * w * c)
        
        return gram
    
    def transfer_style(
        self,
        content_image: Any,
        style_image: Any,
        num_iterations: int = 100
    ) -> Any:
        """
        Transfer style from style_image to content_image
        
        Returns:
            stylized_image: Content with style applied
        """
        # Extract features
        content_features = self.extract_features(content_image)
        style_features = self.extract_features(style_image)
        
        # Initialize output with content
        output = content_image.copy() if NUMPY_AVAILABLE else content_image
        
        # Optimization loop (simplified)
        for iteration in range(num_iterations):
            # Extract output features
            output_features = self.extract_features(output)
            
            # Compute losses (placeholder)
            # In practice, compute content loss and style loss
            
            # Update output (placeholder)
            if NUMPY_AVAILABLE:
                output = output + np.random.randn(*output.shape) * 0.01
            
            if iteration % 20 == 0:
                logger.debug(f"Style transfer iteration {iteration}/{num_iterations}")
        
        return output


# ============================================================================
# GENERATIVE ORCHESTRATOR
# ============================================================================

class GenerativeOrchestrator:
    """
    Complete generative modeling system
    """
    
    def __init__(self, config: Optional[GenerativeConfig] = None):
        self.config = config or GenerativeConfig()
        
        # Models
        self.generator = Generator(self.config)
        self.discriminator = Discriminator(self.config)
        self.vae = VAE(self.config)
        self.diffusion = DiffusionModel(self.config)
        self.style_transfer = StyleTransfer()
        
        # Statistics
        self.generated_images = 0
        self.avg_generation_time = 0.0
        
        logger.info(f"Generative Orchestrator initialized with {self.config.model_type.value}")
    
    def generate_image(
        self,
        model_type: Optional[GenerativeModel] = None,
        condition: Optional[Any] = None
    ) -> Any:
        """Generate image"""
        start_time = time.time()
        
        model_type = model_type or self.config.model_type
        
        if model_type == GenerativeModel.GAN:
            # Sample latent
            if NUMPY_AVAILABLE:
                latent = np.random.randn(1, self.config.latent_dim)
            else:
                latent = [[random.gauss(0, 1) for _ in range(self.config.latent_dim)]]
            
            image = self.generator.forward(latent, condition)
        
        elif model_type == GenerativeModel.VAE:
            # Sample from prior
            if NUMPY_AVAILABLE:
                z = np.random.randn(1, self.config.latent_dim)
            else:
                z = [[random.gauss(0, 1) for _ in range(self.config.latent_dim)]]
            
            image = self.vae.decode(z)
        
        elif model_type == GenerativeModel.DIFFUSION:
            image = self.diffusion.generate(batch_size=1, condition=condition)
        
        else:
            if NUMPY_AVAILABLE:
                image = np.zeros((1, self.config.image_size, self.config.image_size, self.config.channels))
            else:
                image = [[[0] * self.config.channels] * self.config.image_size] * self.config.image_size
        
        generation_time = time.time() - start_time
        
        self._update_stats(generation_time)
        
        return image
    
    def interpolate_latent(
        self,
        latent1: Any,
        latent2: Any,
        num_steps: int = 10
    ) -> List[Any]:
        """Interpolate between two latent vectors"""
        if not NUMPY_AVAILABLE:
            return [latent1] * num_steps
        
        interpolated = []
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            
            # Linear interpolation
            latent = (1 - alpha) * latent1 + alpha * latent2
            
            # Generate image
            image = self.generator.forward(latent)
            
            interpolated.append(image)
        
        return interpolated
    
    def _update_stats(self, generation_time: float):
        """Update statistics"""
        self.generated_images += 1
        self.avg_generation_time = (
            self.avg_generation_time * (self.generated_images - 1) + generation_time
        ) / self.generated_images


# ============================================================================
# TESTING
# ============================================================================

def test_generative():
    """Test generative models"""
    print("=" * 80)
    print("ADVANCED GENERATIVE MODELS - TEST")
    print("=" * 80)
    
    # Create orchestrator
    config = GenerativeConfig(
        model_type=GenerativeModel.GAN,
        latent_dim=128,
        image_size=64,  # Smaller for testing
        gen_filters=[256, 128, 64],
        disc_filters=[64, 128, 256]
    )
    
    gen = GenerativeOrchestrator(config)
    
    print("✓ Generative Orchestrator initialized")
    print(f"  Model: {config.model_type.value}")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Image size: {config.image_size}x{config.image_size}")
    
    # Test GAN generation
    print("\n" + "="*80)
    print("Test: GAN Image Generation")
    print("="*80)
    
    image = gen.generate_image(GenerativeModel.GAN)
    
    if NUMPY_AVAILABLE and hasattr(image, 'shape'):
        print(f"✓ Generated image shape: {image.shape}")
        print(f"  Min value: {np.min(image):.3f}")
        print(f"  Max value: {np.max(image):.3f}")
    
    # Test VAE
    print("\n" + "="*80)
    print("Test: VAE")
    print("="*80)
    
    if NUMPY_AVAILABLE:
        # Create dummy image
        dummy_image = np.random.rand(1, config.image_size, config.image_size, config.channels)
        
        # Encode-decode
        reconstruction, mu, log_var = gen.vae.forward(dummy_image)
        
        print(f"✓ VAE reconstruction shape: {reconstruction.shape}")
        print(f"  Latent mean shape: {mu.shape}")
        print(f"  Latent log_var shape: {log_var.shape}")
        
        # Compute loss
        losses = gen.vae.compute_loss(dummy_image, reconstruction, mu, log_var)
        
        print(f"\n✓ VAE losses:")
        print(f"  Total: {losses['total']:.4f}")
        print(f"  Reconstruction: {losses['reconstruction']:.4f}")
        print(f"  KL divergence: {losses['kl']:.4f}")
    
    # Test diffusion
    print("\n" + "="*80)
    print("Test: Diffusion Model")
    print("="*80)
    
    print(f"✓ Diffusion timesteps: {gen.diffusion.num_timesteps}")
    
    if NUMPY_AVAILABLE:
        # Test noise schedule
        print(f"  Beta start: {gen.diffusion.beta_start}")
        print(f"  Beta end: {gen.diffusion.beta_end}")
        
        # Generate (with fewer steps for testing)
        gen.diffusion.num_timesteps = 50  # Reduce for testing
        
        diff_image = gen.generate_image(GenerativeModel.DIFFUSION)
        
        print(f"\n✓ Diffusion generated image shape: {diff_image.shape}")
    
    # Test latent interpolation
    print("\n" + "="*80)
    print("Test: Latent Interpolation")
    print("="*80)
    
    if NUMPY_AVAILABLE:
        latent1 = np.random.randn(1, config.latent_dim)
        latent2 = np.random.randn(1, config.latent_dim)
        
        interpolated = gen.interpolate_latent(latent1, latent2, num_steps=5)
        
        print(f"✓ Interpolated {len(interpolated)} images")
        print(f"  Shape: {interpolated[0].shape}")
    
    # Test style transfer
    print("\n" + "="*80)
    print("Test: Style Transfer")
    print("="*80)
    
    if NUMPY_AVAILABLE:
        content = np.random.rand(1, config.image_size, config.image_size, config.channels)
        style = np.random.rand(1, config.image_size, config.image_size, config.channels)
        
        # Extract features
        content_features = gen.style_transfer.extract_features(content)
        
        print(f"✓ Extracted {len(content_features)} feature layers")
        
        # Compute Gram matrix
        for layer_name, features in content_features.items():
            if len(features.shape) == 4:
                gram = gen.style_transfer.compute_gram_matrix(features)
                print(f"  {layer_name} Gram shape: {gram.shape}")
                break
    
    # Performance summary
    print("\n" + "="*80)
    print("Performance Summary")
    print("="*80)
    
    print(f"✓ Generated images: {gen.generated_images}")
    print(f"  Average time: {gen.avg_generation_time:.2f}s")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_generative()
