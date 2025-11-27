"""
Computer Vision - YOLOv8 Food Detection Model
Implements YOLOv8 for real-time food detection and recognition.
Supports training, inference, and mobile deployment.

Part of Phase 2: Computer Vision System
Author: AI Nutrition System
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import cv2
from dataclasses import dataclass, field
from datetime import datetime
import json
import yaml


@dataclass
class YOLOConfig:
    """Configuration for YOLOv8 model"""
    # Model architecture
    model_size: str = 'n'  # nano, small, medium, large, xlarge
    num_classes: int = 101  # Food-101 has 101 classes, expandable to 10k+
    input_size: int = 640  # Standard YOLOv8 input
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    warmup_epochs: int = 3
    momentum: float = 0.937
    weight_decay: float = 0.0005
    
    # Augmentation (YOLOv8 built-in)
    augment: bool = True
    mosaic: float = 1.0  # Mosaic augmentation probability
    mixup: float = 0.15  # Mixup augmentation probability
    hsv_h: float = 0.015  # Hue augmentation
    hsv_s: float = 0.7  # Saturation augmentation
    hsv_v: float = 0.4  # Value augmentation
    degrees: float = 0.0  # Rotation degrees
    translate: float = 0.1  # Translation
    scale: float = 0.9  # Scale augmentation
    shear: float = 0.0  # Shear augmentation
    perspective: float = 0.0  # Perspective augmentation
    flipud: float = 0.0  # Vertical flip probability
    fliplr: float = 0.5  # Horizontal flip probability
    
    # Optimization
    optimizer: str = 'AdamW'  # SGD, Adam, AdamW
    cos_lr: bool = True  # Use cosine learning rate schedule
    label_smoothing: float = 0.0  # Label smoothing epsilon
    
    # Loss weights
    box_loss_gain: float = 7.5
    cls_loss_gain: float = 0.5
    dfl_loss_gain: float = 1.5
    
    # Inference
    conf_threshold: float = 0.25  # Confidence threshold
    iou_threshold: float = 0.45  # NMS IoU threshold
    max_det: int = 10  # Maximum detections per image
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    workers: int = 8  # Number of dataloader workers
    
    # Paths
    data_yaml: Path = Path("data/food_dataset.yaml")
    project_dir: Path = Path("runs/food_detection")
    name: str = "yolov8_food"


@dataclass
class DetectionResult:
    """Single detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    food_id: Optional[str] = None  # Link to food database
    portion_estimate: Optional[float] = None  # Estimated grams


@dataclass
class MealAnalysis:
    """Complete meal analysis with multiple foods"""
    image_path: Optional[str] = None
    detections: List[DetectionResult] = field(default_factory=list)
    total_foods: int = 0
    inference_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class YOLOv8FoodDetector:
    """
    YOLOv8-based food detection and recognition system.
    Handles training, inference, and model optimization.
    """
    
    def __init__(self, config: Optional[YOLOConfig] = None):
        """
        Initialize YOLOv8 food detector.
        
        Args:
            config: YOLOConfig object with model parameters
        """
        self.config = config or YOLOConfig()
        self.model = None
        self.class_names = []
        self.food_database_mapping = {}  # Maps class_id to food_id
        
        print(f"🤖 YOLOv8 Food Detector initialized")
        print(f"   Device: {self.config.device}")
        print(f"   Model size: YOLOv8{self.config.model_size}")
    
    def create_model(self, pretrained: bool = True):
        """
        Create YOLOv8 model.
        
        Args:
            pretrained: Load pretrained weights from COCO dataset
        """
        model_name = f'yolov8{self.config.model_size}.pt' if pretrained else f'yolov8{self.config.model_size}.yaml'
        
        print(f"📦 Creating YOLOv8{self.config.model_size} model...")
        print(f"   Pretrained: {pretrained}")
        print(f"   Classes: {self.config.num_classes}")
        
        self.model = YOLO(model_name)
        
        # Override number of classes if using custom dataset
        if pretrained and self.config.num_classes != 80:  # COCO has 80 classes
            print(f"   ⚠️ Model will be fine-tuned from COCO (80 classes) to food dataset ({self.config.num_classes} classes)")
        
        print("✅ Model created successfully")
        return self.model
    
    def prepare_dataset_yaml(
        self,
        train_images_dir: Path,
        val_images_dir: Path,
        class_names: List[str],
        save_path: Optional[Path] = None
    ):
        """
        Create YAML configuration file for YOLOv8 dataset.
        
        Format:
        path: /path/to/dataset
        train: images/train
        val: images/val
        test: images/test  # optional
        
        names:
          0: apple_pie
          1: baby_back_ribs
          ...
        
        Args:
            train_images_dir: Path to training images
            val_images_dir: Path to validation images
            class_names: List of class names
            save_path: Where to save YAML file
        """
        save_path = save_path or self.config.data_yaml
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create dataset configuration
        dataset_config = {
            'path': str(train_images_dir.parent.parent),  # Root dataset directory
            'train': str(train_images_dir.relative_to(train_images_dir.parent.parent)),
            'val': str(val_images_dir.relative_to(val_images_dir.parent.parent)),
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        # Save to YAML
        with open(save_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Dataset YAML created: {save_path}")
        print(f"   Train: {train_images_dir}")
        print(f"   Val: {val_images_dir}")
        print(f"   Classes: {len(class_names)}")
        
        self.class_names = class_names
        self.config.data_yaml = save_path
        
        return save_path
    
    def train(
        self,
        resume: bool = False,
        save_period: int = 10,
        patience: int = 50,
        verbose: bool = True
    ):
        """
        Train YOLOv8 model on food dataset.
        
        Args:
            resume: Resume training from last checkpoint
            save_period: Save checkpoint every N epochs
            patience: Early stopping patience
            verbose: Print detailed logs
        
        Returns:
            Training results dictionary
        """
        if self.model is None:
            self.create_model(pretrained=True)
        
        print("🚀 Starting YOLOv8 training...")
        print(f"   Epochs: {self.config.epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print(f"   Device: {self.config.device}")
        print()
        
        # Training arguments
        train_args = {
            'data': str(self.config.data_yaml),
            'epochs': self.config.epochs,
            'batch': self.config.batch_size,
            'imgsz': self.config.input_size,
            'device': self.config.device,
            'workers': self.config.workers,
            
            # Optimization
            'optimizer': self.config.optimizer,
            'lr0': self.config.learning_rate,
            'lrf': 0.01,  # Final learning rate
            'momentum': self.config.momentum,
            'weight_decay': self.config.weight_decay,
            'warmup_epochs': self.config.warmup_epochs,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'cos_lr': self.config.cos_lr,
            'label_smoothing': self.config.label_smoothing,
            
            # Loss
            'box': self.config.box_loss_gain,
            'cls': self.config.cls_loss_gain,
            'dfl': self.config.dfl_loss_gain,
            
            # Augmentation
            'augment': self.config.augment,
            'mosaic': self.config.mosaic,
            'mixup': self.config.mixup,
            'hsv_h': self.config.hsv_h,
            'hsv_s': self.config.hsv_s,
            'hsv_v': self.config.hsv_v,
            'degrees': self.config.degrees,
            'translate': self.config.translate,
            'scale': self.config.scale,
            'shear': self.config.shear,
            'perspective': self.config.perspective,
            'flipud': self.config.flipud,
            'fliplr': self.config.fliplr,
            
            # Saving
            'project': str(self.config.project_dir),
            'name': self.config.name,
            'exist_ok': True,
            'save': True,
            'save_period': save_period,
            'resume': resume,
            'patience': patience,
            'verbose': verbose,
            
            # Evaluation
            'val': True,
            'plots': True,
        }
        
        # Start training
        results = self.model.train(**train_args)
        
        print("✅ Training complete!")
        print(f"   Best weights: {self.config.project_dir}/{self.config.name}/weights/best.pt")
        
        return results
    
    def load_model(self, weights_path: Union[str, Path]):
        """
        Load trained model weights.
        
        Args:
            weights_path: Path to model weights (.pt file)
        """
        weights_path = Path(weights_path)
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        print(f"📦 Loading model weights: {weights_path}")
        self.model = YOLO(str(weights_path))
        
        # Extract class names from model
        if hasattr(self.model, 'names'):
            self.class_names = list(self.model.names.values())
            print(f"✅ Model loaded with {len(self.class_names)} classes")
        
        return self.model
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_det: Optional[int] = None,
        visualize: bool = False
    ) -> MealAnalysis:
        """
        Perform inference on single image.
        
        Args:
            image: Image path or numpy array
            conf_threshold: Confidence threshold (0-1)
            iou_threshold: NMS IoU threshold (0-1)
            max_det: Maximum detections
            visualize: Draw bounding boxes
        
        Returns:
            MealAnalysis object with detections
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call create_model() or load_model() first.")
        
        conf_threshold = conf_threshold or self.config.conf_threshold
        iou_threshold = iou_threshold or self.config.iou_threshold
        max_det = max_det or self.config.max_det
        
        # Run inference
        start_time = cv2.getTickCount()
        
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_det,
            device=self.config.device,
            verbose=False,
            stream=False
        )[0]  # Get first result
        
        end_time = cv2.getTickCount()
        inference_time = ((end_time - start_time) / cv2.getTickFrequency()) * 1000  # ms
        
        # Parse detections
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)  # x1, y1, x2, y2
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                
                detection = DetectionResult(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=tuple(bbox),
                    food_id=self.food_database_mapping.get(cls_id)
                )
                detections.append(detection)
        
        # Create meal analysis
        image_path = str(image) if isinstance(image, (str, Path)) else None
        
        meal_analysis = MealAnalysis(
            image_path=image_path,
            detections=detections,
            total_foods=len(detections),
            inference_time_ms=inference_time
        )
        
        # Visualize if requested
        if visualize and results.orig_img is not None:
            self._visualize_detections(results.orig_img, detections)
        
        return meal_analysis
    
    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> List[MealAnalysis]:
        """
        Perform inference on batch of images.
        
        Args:
            images: List of image paths or numpy arrays
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
        
        Returns:
            List of MealAnalysis objects
        """
        return [self.predict(img, conf_threshold, iou_threshold) for img in images]
    
    def _visualize_detections(
        self,
        image: np.ndarray,
        detections: List[DetectionResult],
        save_path: Optional[Path] = None
    ):
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Input image
            detections: List of detections
            save_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            width = x2 - x1
            height = y2 - y1
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor='lime',
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            ax.text(
                x1, y1 - 5,
                label,
                color='white',
                fontsize=10,
                bbox=dict(facecolor='lime', alpha=0.7, edgecolor='none', pad=2)
            )
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def export_model(
        self,
        format: str = 'onnx',
        save_path: Optional[Path] = None,
        optimize: bool = True
    ):
        """
        Export model to different formats for deployment.
        
        Supported formats:
        - onnx: ONNX (Open Neural Network Exchange)
        - torchscript: TorchScript
        - tflite: TensorFlow Lite (mobile)
        - coreml: CoreML (iOS)
        - engine: TensorRT (NVIDIA)
        
        Args:
            format: Export format
            save_path: Where to save exported model
            optimize: Apply optimization (quantization, pruning)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        print(f"📦 Exporting model to {format.upper()}...")
        
        export_args = {
            'format': format,
            'optimize': optimize,
            'half': False,  # FP16 quantization
            'int8': optimize,  # INT8 quantization for mobile
            'device': self.config.device,
            'verbose': True
        }
        
        exported_path = self.model.export(**export_args)
        
        print(f"✅ Model exported: {exported_path}")
        return exported_path
    
    def link_food_database(self, mapping: Dict[int, str]):
        """
        Link detected classes to food database IDs.
        
        Args:
            mapping: Dictionary mapping class_id to food_id
        
        Example:
            {0: 'fruit_apple_001', 1: 'meat_beef_ribs_001', ...}
        """
        self.food_database_mapping = mapping
        print(f"✅ Linked {len(mapping)} classes to food database")


def test_yolov8_setup():
    """Test YOLOv8 setup"""
    print("🧪 Testing YOLOv8 Food Detection Setup")
    print("=" * 50)
    
    # Initialize detector
    config = YOLOConfig(
        model_size='n',  # Nano for fast testing
        num_classes=101,
        epochs=1,  # Quick test
        batch_size=16
    )
    
    detector = YOLOv8FoodDetector(config)
    
    print("\n✅ YOLOv8 detector initialized")
    print(f"   Model: YOLOv8{config.model_size}")
    print(f"   Classes: {config.num_classes}")
    print(f"   Device: {config.device}")
    print()
    
    print("📝 Next steps:")
    print("   1. Prepare Food-101 dataset in YOLO format")
    print("   2. Create dataset.yaml configuration")
    print("   3. Run detector.train() to start training")
    print("   4. Export model for mobile deployment")
    print()
    
    print("🎯 Ready to train on 10,000+ foods from database!")
    
    return detector


if __name__ == "__main__":
    # Test YOLOv8 setup
    detector = test_yolov8_setup()
