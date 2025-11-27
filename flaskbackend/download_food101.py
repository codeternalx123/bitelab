"""
Download and Prepare Food-101 Dataset for YOLOv8 Training
Food-101 contains 101k images across 101 food categories.

This script:
1. Downloads Food-101 dataset (~5GB)
2. Converts to YOLO format
3. Creates train/val splits
4. Generates dataset.yaml configuration

Dataset info: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
"""

import os
import tarfile
import urllib.request
from pathlib import Path
import shutil
from tqdm import tqdm
import json
from PIL import Image
import random
from typing import Dict, List, Tuple


class Food101Downloader:
    """Download and prepare Food-101 dataset for YOLOv8"""
    
    DATASET_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    DATASET_SIZE_GB = 4.65
    
    def __init__(self, root_dir: Path = Path("data/food101")):
        """
        Initialize downloader.
        
        Args:
            root_dir: Directory to save dataset
        """
        self.root_dir = Path(root_dir)
        self.download_dir = self.root_dir / "raw"
        self.yolo_dir = self.root_dir / "yolo"
        
        self.class_names = []
        self.class_to_idx = {}
    
    def download(self, skip_if_exists: bool = True) -> bool:
        """
        Download Food-101 dataset.
        
        Args:
            skip_if_exists: Skip download if already exists
        
        Returns:
            True if successful
        """
        tar_path = self.download_dir / "food-101.tar.gz"
        extract_path = self.download_dir / "food-101"
        
        # Check if already downloaded
        if skip_if_exists and extract_path.exists():
            print(f"✅ Dataset already exists: {extract_path}")
            return True
        
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Downloading Food-101 dataset (~{self.DATASET_SIZE_GB}GB)")
        print(f"   URL: {self.DATASET_URL}")
        print(f"   Destination: {tar_path}")
        print()
        
        # Download with progress bar
        try:
            def _progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                print(f"\r   Progress: {percent:.1f}% ({downloaded / (1024**3):.2f}GB / {total_size / (1024**3):.2f}GB)", end='')
            
            urllib.request.urlretrieve(
                self.DATASET_URL,
                tar_path,
                reporthook=_progress_hook
            )
            print()  # New line after progress
            print("✅ Download complete!")
            
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            return False
        
        # Extract tarball
        print(f"\n📦 Extracting dataset...")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(self.download_dir)
            print("✅ Extraction complete!")
            
            # Remove tarball to save space
            if tar_path.exists():
                tar_path.unlink()
                print(f"🗑️ Removed tarball (saved {self.DATASET_SIZE_GB:.1f}GB)")
            
            return True
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    def load_class_names(self) -> List[str]:
        """
        Load class names from Food-101 meta data.
        
        Returns:
            List of class names (101 food categories)
        """
        meta_dir = self.download_dir / "food-101" / "meta"
        classes_file = meta_dir / "classes.txt"
        
        if not classes_file.exists():
            print(f"❌ Classes file not found: {classes_file}")
            return []
        
        with open(classes_file, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        print(f"✅ Loaded {len(self.class_names)} classes")
        return self.class_names
    
    def convert_to_yolo_format(
        self,
        train_split: float = 0.8,
        create_labels: bool = True
    ) -> bool:
        """
        Convert Food-101 to YOLO format.
        
        YOLO format:
        - images/train/*.jpg
        - images/val/*.jpg
        - labels/train/*.txt
        - labels/val/*.txt
        
        Each label file contains:
        <class_id> <x_center> <y_center> <width> <height>
        (all normalized to 0-1)
        
        Args:
            train_split: Fraction for training set
            create_labels: Generate label files (single class per image)
        
        Returns:
            True if successful
        """
        if not self.class_names:
            self.load_class_names()
        
        source_dir = self.download_dir / "food-101"
        images_dir = source_dir / "images"
        meta_dir = source_dir / "meta"
        
        # Create YOLO directory structure
        yolo_images_train = self.yolo_dir / "images" / "train"
        yolo_images_val = self.yolo_dir / "images" / "val"
        yolo_labels_train = self.yolo_dir / "labels" / "train"
        yolo_labels_val = self.yolo_dir / "labels" / "val"
        
        for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔄 Converting to YOLO format...")
        print(f"   Train split: {train_split * 100:.0f}%")
        print(f"   Val split: {(1 - train_split) * 100:.0f}%")
        print()
        
        # Load official train/test splits from Food-101
        train_file = meta_dir / "train.txt"
        test_file = meta_dir / "test.txt"
        
        train_images = []
        val_images = []
        
        if train_file.exists() and test_file.exists():
            # Use official splits
            with open(train_file, 'r') as f:
                train_images = [line.strip() for line in f.readlines()]
            
            with open(test_file, 'r') as f:
                val_images = [line.strip() for line in f.readlines()]
            
            print(f"✅ Using official Food-101 splits")
        else:
            # Create custom splits
            print(f"⚠️ Official splits not found, creating custom splits")
            
            for class_name in self.class_names:
                class_dir = images_dir / class_name
                if not class_dir.exists():
                    continue
                
                image_files = list(class_dir.glob("*.jpg"))
                random.shuffle(image_files)
                
                split_idx = int(len(image_files) * train_split)
                train_images.extend([f"{class_name}/{img.stem}" for img in image_files[:split_idx]])
                val_images.extend([f"{class_name}/{img.stem}" for img in image_files[split_idx:]])
        
        print(f"   Train images: {len(train_images)}")
        print(f"   Val images: {len(val_images)}")
        print()
        
        # Copy and convert images
        print("📋 Processing training set...")
        self._process_split(images_dir, train_images, yolo_images_train, yolo_labels_train, create_labels)
        
        print("\n📋 Processing validation set...")
        self._process_split(images_dir, val_images, yolo_images_val, yolo_labels_val, create_labels)
        
        print("\n✅ Conversion complete!")
        print(f"   YOLO dataset saved to: {self.yolo_dir}")
        
        return True
    
    def _process_split(
        self,
        source_images_dir: Path,
        image_list: List[str],
        target_images_dir: Path,
        target_labels_dir: Path,
        create_labels: bool
    ):
        """Process a single split (train or val)"""
        for img_path in tqdm(image_list, desc="   Converting"):
            # Parse class and filename
            class_name, filename = img_path.split('/')
            class_idx = self.class_to_idx.get(class_name, 0)
            
            # Source and target paths
            source_image = source_images_dir / class_name / f"{filename}.jpg"
            target_image = target_images_dir / f"{class_name}_{filename}.jpg"
            target_label = target_labels_dir / f"{class_name}_{filename}.txt"
            
            # Copy image
            if source_image.exists():
                shutil.copy2(source_image, target_image)
                
                # Create label file (single class, full image)
                # For Food-101, each image contains one centered food item
                if create_labels:
                    with open(target_label, 'w') as f:
                        # YOLO format: <class_id> <x_center> <y_center> <width> <height>
                        # Assuming food occupies ~80% of image, centered
                        f.write(f"{class_idx} 0.5 0.5 0.8 0.8\n")
    
    def create_dataset_yaml(self) -> Path:
        """
        Create dataset.yaml for YOLOv8 training.
        
        Returns:
            Path to created YAML file
        """
        yaml_path = self.yolo_dir / "dataset.yaml"
        
        yaml_content = f"""# Food-101 Dataset Configuration for YOLOv8
# Generated by Food101Downloader

path: {self.yolo_dir.absolute()}
train: images/train
val: images/val

# Number of classes
nc: {len(self.class_names)}

# Class names
names:
"""
        
        for idx, name in enumerate(self.class_names):
            yaml_content += f"  {idx}: {name}\n"
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✅ Dataset YAML created: {yaml_path}")
        return yaml_path
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset stats
        """
        stats = {
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'splits': {}
        }
        
        for split in ['train', 'val']:
            images_dir = self.yolo_dir / "images" / split
            labels_dir = self.yolo_dir / "labels" / split
            
            if images_dir.exists():
                num_images = len(list(images_dir.glob("*.jpg")))
                num_labels = len(list(labels_dir.glob("*.txt")))
                
                stats['splits'][split] = {
                    'num_images': num_images,
                    'num_labels': num_labels
                }
        
        return stats
    
    def print_statistics(self):
        """Print dataset statistics"""
        stats = self.get_statistics()
        
        print("\n📊 Dataset Statistics")
        print("=" * 50)
        print(f"Classes: {stats['num_classes']}")
        print()
        
        for split, split_stats in stats['splits'].items():
            print(f"{split.upper()} Split:")
            print(f"  Images: {split_stats['num_images']:,}")
            print(f"  Labels: {split_stats['num_labels']:,}")
            print()
        
        print("Sample classes:")
        for i, name in enumerate(stats['class_names'][:10]):
            print(f"  {i}: {name}")
        if len(stats['class_names']) > 10:
            print(f"  ... and {len(stats['class_names']) - 10} more")


def main():
    """Main execution"""
    print("🍕 Food-101 Dataset Preparation for YOLOv8")
    print("=" * 60)
    print()
    
    # Initialize downloader
    downloader = Food101Downloader(root_dir=Path("data/food101"))
    
    # Download dataset
    print("Step 1: Download Food-101 dataset")
    if not downloader.download(skip_if_exists=True):
        print("❌ Download failed")
        return
    
    # Load class names
    print("\nStep 2: Load class names")
    downloader.load_class_names()
    
    # Convert to YOLO format
    print("\nStep 3: Convert to YOLO format")
    if not downloader.convert_to_yolo_format(train_split=0.8, create_labels=True):
        print("❌ Conversion failed")
        return
    
    # Create dataset.yaml
    print("\nStep 4: Create dataset.yaml")
    yaml_path = downloader.create_dataset_yaml()
    
    # Print statistics
    print("\nStep 5: Dataset statistics")
    downloader.print_statistics()
    
    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print()
    print("Next steps:")
    print("1. Review dataset structure in: data/food101/yolo/")
    print(f"2. Use dataset.yaml in training: {yaml_path}")
    print("3. Train YOLOv8 model:")
    print("   from app.ai_nutrition.cv.cv_yolov8_model import YOLOv8FoodDetector")
    print("   detector = YOLOv8FoodDetector()")
    print("   detector.create_model(pretrained=True)")
    print("   detector.train()")
    print()
    print("⏱️ Estimated training time:")
    print("   - RTX 3060 (12GB): 6-8 hours for 100 epochs")
    print("   - RTX 4090 (24GB): 3-4 hours for 100 epochs")
    print("   - Google Colab (T4): 10-12 hours for 100 epochs")


if __name__ == "__main__":
    main()
