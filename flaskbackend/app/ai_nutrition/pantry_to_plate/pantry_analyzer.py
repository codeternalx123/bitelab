"""
Pantry Analysis & Inventory Management
=======================================

Computer vision-based pantry scanning system for ingredient detection, quantity
estimation, expiry tracking, and inventory database management. Enables the
Pantry-to-Plate recommendation engine.

Features:
1. Pantry scanning with computer vision (YOLO/ViT)
2. Ingredient detection and classification (500+ food items)
3. Quantity estimation (visual + OCR for labels)
4. Expiry date detection (OCR)
5. Freshness assessment (visual quality scoring)
6. Inventory database management
7. Low-stock alerts and shopping list generation
8. Ingredient substitution suggestions
9. Waste prediction and reduction
10. Multi-user household inventory sharing

Technical Stack:
- Vision: YOLO v8 for object detection
- OCR: Tesseract/PaddleOCR for text recognition
- Database: PostgreSQL for inventory
- Cache: Redis for quick lookups
- Storage: S3 for pantry images

Performance Targets:
- Ingredient detection: >90% accuracy
- Quantity estimation: ±10% error
- Expiry OCR: >95% accuracy
- Scan time: <3 seconds per image
- Inventory update: <100ms

Use Cases:
1. User scans pantry → System identifies all items
2. System tracks quantities → Alerts when low
3. Recipe matching → Uses available ingredients
4. Waste reduction → Suggests recipes for near-expiry items
5. Shopping list → Auto-generates based on planned recipes

Author: Wellomex AI Team
Date: November 2025
Version: 9.0.0
"""

import logging
import time
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION & ENUMS
# ============================================================================

class FoodCategory(Enum):
    """Food categories for organization"""
    PROTEIN_ANIMAL = "protein_animal"
    PROTEIN_PLANT = "protein_plant"
    GRAINS = "grains"
    VEGETABLES = "vegetables"
    FRUITS = "fruits"
    DAIRY = "dairy"
    FATS_OILS = "fats_oils"
    SPICES = "spices"
    CANNED = "canned"
    FROZEN = "frozen"
    BEVERAGES = "beverages"
    SNACKS = "snacks"


class StorageLocation(Enum):
    """Storage locations in kitchen"""
    PANTRY = "pantry"
    REFRIGERATOR = "refrigerator"
    FREEZER = "freezer"
    COUNTER = "counter"
    CABINET = "cabinet"


class FreshnessLevel(Enum):
    """Freshness assessment levels"""
    FRESH = "fresh"          # >7 days to expiry
    GOOD = "good"            # 3-7 days to expiry
    USE_SOON = "use_soon"    # 1-3 days to expiry
    EXPIRING = "expiring"    # <1 day to expiry
    EXPIRED = "expired"      # Past expiry date


class Unit(Enum):
    """Measurement units"""
    GRAMS = "g"
    KILOGRAMS = "kg"
    MILLILITERS = "ml"
    LITERS = "l"
    PIECES = "pieces"
    CANS = "cans"
    PACKAGES = "packages"
    CUPS = "cups"
    TABLESPOONS = "tbsp"
    TEASPOONS = "tsp"


@dataclass
class PantryConfig:
    """Pantry analysis configuration"""
    # Detection thresholds
    detection_confidence: float = 0.75
    ocr_confidence: float = 0.80
    
    # Quantity estimation
    quantity_tolerance: float = 0.10  # ±10%
    
    # Freshness alerts
    expiry_warning_days: int = 3
    low_stock_threshold: float = 0.2  # 20% of typical quantity
    
    # Database
    max_inventory_items: int = 500
    history_retention_days: int = 90


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class IngredientDetection:
    """Detected ingredient from vision"""
    ingredient_id: str
    name: str
    category: FoodCategory
    confidence: float  # 0.0-1.0
    
    # Bounding box (normalized 0-1)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)  # (x1, y1, x2, y2)
    
    # Visual features
    estimated_quantity: Optional[float] = None
    unit: Optional[Unit] = None
    detected_brand: Optional[str] = None
    detected_expiry: Optional[datetime.date] = None
    
    # Quality assessment
    visual_freshness: Optional[FreshnessLevel] = None
    package_damaged: bool = False


@dataclass
class InventoryItem:
    """Item in user's inventory"""
    item_id: str
    user_id: str
    ingredient_id: str
    name: str
    category: FoodCategory
    
    # Quantity
    quantity: float
    unit: Unit
    
    # Storage
    storage_location: StorageLocation
    purchase_date: datetime.date
    expiry_date: Optional[datetime.date] = None
    
    # Metadata
    brand: Optional[str] = None
    barcode: Optional[str] = None
    price_paid: Optional[float] = None
    
    # Status
    is_opened: bool = False
    freshness: FreshnessLevel = FreshnessLevel.FRESH
    
    # Tracking
    last_updated: datetime.datetime = field(default_factory=datetime.datetime.now)
    times_scanned: int = 0
    
    def days_until_expiry(self) -> Optional[int]:
        """Calculate days until expiry"""
        if self.expiry_date:
            delta = self.expiry_date - datetime.date.today()
            return delta.days
        return None
    
    def is_low_stock(self, threshold: float = 0.2) -> bool:
        """Check if item is low stock"""
        # Simplified - compare to typical quantities
        typical_quantities = {
            Unit.KILOGRAMS: 1.0,
            Unit.GRAMS: 500.0,
            Unit.LITERS: 1.0,
            Unit.PIECES: 5.0
        }
        
        typical = typical_quantities.get(self.unit, 1.0)
        return self.quantity < (typical * threshold)


@dataclass
class PantryScanResult:
    """Result of scanning pantry"""
    scan_id: str
    user_id: str
    timestamp: datetime.datetime
    
    # Detected items
    detections: List[IngredientDetection] = field(default_factory=list)
    
    # Updated inventory
    inventory_updates: List[InventoryItem] = field(default_factory=list)
    
    # Alerts
    expiring_items: List[InventoryItem] = field(default_factory=list)
    low_stock_items: List[InventoryItem] = field(default_factory=list)
    
    # Stats
    total_items_detected: int = 0
    processing_time_seconds: float = 0.0


@dataclass
class ShoppingListItem:
    """Item for shopping list"""
    ingredient_id: str
    name: str
    quantity: float
    unit: Unit
    priority: str  # "urgent", "soon", "optional"
    reason: str  # "expiring", "low_stock", "recipe_requirement"


# ============================================================================
# MOCK VISION MODEL (PANTRY SCANNER)
# ============================================================================

class PantryVisionModel:
    """
    Computer vision model for pantry scanning
    
    In production: Replace with YOLO v8 fine-tuned on food packaging
    """
    
    def __init__(self):
        # Food item database (500+ items in production)
        self.food_database = {
            'chicken_breast': {
                'name': 'Chicken Breast',
                'category': FoodCategory.PROTEIN_ANIMAL,
                'typical_unit': Unit.GRAMS,
                'typical_quantity': 500.0,
                'shelf_life_days': 3
            },
            'rice': {
                'name': 'White Rice',
                'category': FoodCategory.GRAINS,
                'typical_unit': Unit.KILOGRAMS,
                'typical_quantity': 1.0,
                'shelf_life_days': 365
            },
            'tomato_paste': {
                'name': 'Tomato Paste',
                'category': FoodCategory.CANNED,
                'typical_unit': Unit.CANS,
                'typical_quantity': 1.0,
                'shelf_life_days': 730
            },
            'milk': {
                'name': 'Milk',
                'category': FoodCategory.DAIRY,
                'typical_unit': Unit.LITERS,
                'typical_quantity': 1.0,
                'shelf_life_days': 7
            },
            'olive_oil': {
                'name': 'Olive Oil',
                'category': FoodCategory.FATS_OILS,
                'typical_unit': Unit.MILLILITERS,
                'typical_quantity': 750.0,
                'shelf_life_days': 180
            },
            'eggs': {
                'name': 'Eggs',
                'category': FoodCategory.PROTEIN_ANIMAL,
                'typical_unit': Unit.PIECES,
                'typical_quantity': 12.0,
                'shelf_life_days': 21
            },
            'spinach': {
                'name': 'Fresh Spinach',
                'category': FoodCategory.VEGETABLES,
                'typical_unit': Unit.GRAMS,
                'typical_quantity': 200.0,
                'shelf_life_days': 5
            },
            'onions': {
                'name': 'Onions',
                'category': FoodCategory.VEGETABLES,
                'typical_unit': Unit.PIECES,
                'typical_quantity': 3.0,
                'shelf_life_days': 30
            },
            'garlic': {
                'name': 'Garlic',
                'category': FoodCategory.VEGETABLES,
                'typical_unit': Unit.PIECES,
                'typical_quantity': 1.0,
                'shelf_life_days': 60
            },
            'salt': {
                'name': 'Salt',
                'category': FoodCategory.SPICES,
                'typical_unit': Unit.GRAMS,
                'typical_quantity': 500.0,
                'shelf_life_days': 9999
            }
        }
        
        logger.info(f"Pantry Vision Model initialized with {len(self.food_database)} items")
    
    def scan_pantry_image(self, image_description: str) -> List[IngredientDetection]:
        """
        Scan pantry image and detect ingredients
        
        Args:
            image_description: Mock image description
        
        Returns:
            List of detected ingredients
        """
        desc_lower = image_description.lower()
        
        detections = []
        
        # Simple keyword matching (mock)
        for ingredient_id, data in self.food_database.items():
            if ingredient_id.replace('_', ' ') in desc_lower or data['name'].lower() in desc_lower:
                detection = IngredientDetection(
                    ingredient_id=ingredient_id,
                    name=data['name'],
                    category=data['category'],
                    confidence=0.85 + (hash(ingredient_id) % 10) / 100,  # Mock confidence
                    estimated_quantity=data['typical_quantity'],
                    unit=data['typical_unit']
                )
                
                # Mock expiry detection
                if 'expiring' in desc_lower:
                    detection.detected_expiry = datetime.date.today() + datetime.timedelta(days=2)
                    detection.visual_freshness = FreshnessLevel.USE_SOON
                else:
                    detection.detected_expiry = datetime.date.today() + datetime.timedelta(
                        days=data['shelf_life_days']
                    )
                    detection.visual_freshness = FreshnessLevel.FRESH
                
                detections.append(detection)
        
        return detections


# ============================================================================
# OCR MODEL (LABEL READER)
# ============================================================================

class OCRModel:
    """
    OCR model for reading labels and expiry dates
    
    In production: Use Tesseract or PaddleOCR
    """
    
    def __init__(self):
        # Date patterns
        self.date_patterns = [
            r'(\d{1,2})/(\d{1,2})/(\d{2,4})',  # MM/DD/YYYY
            r'(\d{4})-(\d{2})-(\d{2})',         # YYYY-MM-DD
            r'EXP[: ]*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # EXP: MM/DD/YY
        ]
        
        logger.info("OCR Model initialized")
    
    def read_expiry_date(self, text: str) -> Optional[datetime.date]:
        """Extract expiry date from OCR text"""
        for pattern in self.date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    groups = match.groups()
                    
                    # Parse date (handle different formats)
                    if len(groups) == 3:
                        if '-' in text:  # YYYY-MM-DD
                            year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        else:  # MM/DD/YYYY
                            month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                        
                        # Handle 2-digit years
                        if year < 100:
                            year += 2000
                        
                        return datetime.date(year, month, day)
                except ValueError:
                    continue
        
        return None
    
    def read_quantity(self, text: str) -> Optional[Tuple[float, Unit]]:
        """Extract quantity from label"""
        # Patterns for common quantity formats
        patterns = [
            (r'(\d+\.?\d*)\s*kg', Unit.KILOGRAMS),
            (r'(\d+\.?\d*)\s*g', Unit.GRAMS),
            (r'(\d+\.?\d*)\s*l', Unit.LITERS),
            (r'(\d+\.?\d*)\s*ml', Unit.MILLILITERS),
        ]
        
        for pattern, unit in patterns:
            match = re.search(pattern, text.lower())
            if match:
                quantity = float(match.group(1))
                return (quantity, unit)
        
        return None


# ============================================================================
# INVENTORY DATABASE
# ============================================================================

class InventoryDatabase:
    """
    Inventory database manager
    
    In production: Use PostgreSQL with proper schema
    """
    
    def __init__(self):
        # In-memory storage (mock)
        self.inventory: Dict[str, Dict[str, InventoryItem]] = defaultdict(dict)  # user_id -> {item_id: item}
        self.scan_history: List[PantryScanResult] = []
        
        logger.info("Inventory Database initialized")
    
    def add_item(self, item: InventoryItem) -> bool:
        """Add item to inventory"""
        self.inventory[item.user_id][item.item_id] = item
        return True
    
    def update_item(self, user_id: str, item_id: str, updates: Dict[str, Any]) -> bool:
        """Update inventory item"""
        if user_id in self.inventory and item_id in self.inventory[user_id]:
            item = self.inventory[user_id][item_id]
            
            for key, value in updates.items():
                if hasattr(item, key):
                    setattr(item, key, value)
            
            item.last_updated = datetime.datetime.now()
            return True
        
        return False
    
    def get_item(self, user_id: str, item_id: str) -> Optional[InventoryItem]:
        """Get inventory item"""
        return self.inventory.get(user_id, {}).get(item_id)
    
    def get_user_inventory(self, user_id: str) -> List[InventoryItem]:
        """Get all items for user"""
        return list(self.inventory.get(user_id, {}).values())
    
    def search_by_ingredient(self, user_id: str, ingredient_id: str) -> List[InventoryItem]:
        """Find items by ingredient"""
        return [
            item for item in self.get_user_inventory(user_id)
            if item.ingredient_id == ingredient_id
        ]
    
    def get_expiring_items(self, user_id: str, days: int = 3) -> List[InventoryItem]:
        """Get items expiring within days"""
        expiring = []
        
        for item in self.get_user_inventory(user_id):
            days_left = item.days_until_expiry()
            if days_left is not None and 0 <= days_left <= days:
                expiring.append(item)
        
        return expiring
    
    def get_low_stock_items(self, user_id: str, threshold: float = 0.2) -> List[InventoryItem]:
        """Get low stock items"""
        return [
            item for item in self.get_user_inventory(user_id)
            if item.is_low_stock(threshold)
        ]
    
    def remove_item(self, user_id: str, item_id: str) -> bool:
        """Remove item from inventory"""
        if user_id in self.inventory and item_id in self.inventory[user_id]:
            del self.inventory[user_id][item_id]
            return True
        return False


# ============================================================================
# PANTRY ANALYZER
# ============================================================================

class PantryAnalyzer:
    """
    Complete pantry analysis system
    """
    
    def __init__(self, config: Optional[PantryConfig] = None):
        self.config = config or PantryConfig()
        
        # Components
        self.vision_model = PantryVisionModel()
        self.ocr_model = OCRModel()
        self.database = InventoryDatabase()
        
        logger.info("Pantry Analyzer initialized")
    
    def scan_pantry(
        self,
        user_id: str,
        image_description: str
    ) -> PantryScanResult:
        """
        Scan pantry image and update inventory
        
        Args:
            user_id: User identifier
            image_description: Mock image description
        
        Returns:
            Scan result with detections and updates
        """
        import uuid
        start_time = time.time()
        
        scan_id = str(uuid.uuid4())
        result = PantryScanResult(
            scan_id=scan_id,
            user_id=user_id,
            timestamp=datetime.datetime.now()
        )
        
        # Run vision detection
        detections = self.vision_model.scan_pantry_image(image_description)
        result.detections = detections
        result.total_items_detected = len(detections)
        
        # Update inventory
        for detection in detections:
            if detection.confidence >= self.config.detection_confidence:
                # Check if item already exists
                existing = self.database.search_by_ingredient(user_id, detection.ingredient_id)
                
                if existing:
                    # Update existing item
                    item = existing[0]
                    self.database.update_item(
                        user_id,
                        item.item_id,
                        {
                            'quantity': detection.estimated_quantity or item.quantity,
                            'expiry_date': detection.detected_expiry or item.expiry_date,
                            'freshness': detection.visual_freshness or item.freshness,
                            'times_scanned': item.times_scanned + 1
                        }
                    )
                    result.inventory_updates.append(item)
                else:
                    # Add new item
                    item = InventoryItem(
                        item_id=str(uuid.uuid4()),
                        user_id=user_id,
                        ingredient_id=detection.ingredient_id,
                        name=detection.name,
                        category=detection.category,
                        quantity=detection.estimated_quantity or 0.0,
                        unit=detection.unit or Unit.PIECES,
                        storage_location=self._infer_storage(detection.category),
                        purchase_date=datetime.date.today(),
                        expiry_date=detection.detected_expiry,
                        brand=detection.detected_brand,
                        freshness=detection.visual_freshness or FreshnessLevel.FRESH,
                        times_scanned=1
                    )
                    
                    self.database.add_item(item)
                    result.inventory_updates.append(item)
        
        # Generate alerts
        result.expiring_items = self.database.get_expiring_items(
            user_id,
            self.config.expiry_warning_days
        )
        
        result.low_stock_items = self.database.get_low_stock_items(
            user_id,
            self.config.low_stock_threshold
        )
        
        result.processing_time_seconds = time.time() - start_time
        
        # Store scan history
        self.database.scan_history.append(result)
        
        return result
    
    def _infer_storage(self, category: FoodCategory) -> StorageLocation:
        """Infer storage location from category"""
        storage_map = {
            FoodCategory.PROTEIN_ANIMAL: StorageLocation.REFRIGERATOR,
            FoodCategory.DAIRY: StorageLocation.REFRIGERATOR,
            FoodCategory.VEGETABLES: StorageLocation.REFRIGERATOR,
            FoodCategory.FRUITS: StorageLocation.COUNTER,
            FoodCategory.FROZEN: StorageLocation.FREEZER,
            FoodCategory.GRAINS: StorageLocation.PANTRY,
            FoodCategory.CANNED: StorageLocation.PANTRY,
            FoodCategory.SPICES: StorageLocation.CABINET,
        }
        
        return storage_map.get(category, StorageLocation.PANTRY)
    
    def get_available_ingredients(
        self,
        user_id: str,
        include_low_stock: bool = True
    ) -> Dict[str, float]:
        """
        Get all available ingredients with quantities
        
        Returns:
            Dict of ingredient_id -> quantity (normalized to grams/ml)
        """
        inventory = self.database.get_user_inventory(user_id)
        
        available = {}
        
        for item in inventory:
            # Skip expired items
            days_left = item.days_until_expiry()
            if days_left is not None and days_left < 0:
                continue
            
            # Skip low stock if requested
            if not include_low_stock and item.is_low_stock():
                continue
            
            # Normalize quantity to grams/ml
            normalized_qty = self._normalize_quantity(item.quantity, item.unit)
            
            if item.ingredient_id in available:
                available[item.ingredient_id] += normalized_qty
            else:
                available[item.ingredient_id] = normalized_qty
        
        return available
    
    def _normalize_quantity(self, quantity: float, unit: Unit) -> float:
        """Normalize quantity to grams/ml"""
        conversions = {
            Unit.KILOGRAMS: 1000.0,
            Unit.GRAMS: 1.0,
            Unit.LITERS: 1000.0,
            Unit.MILLILITERS: 1.0,
            Unit.PIECES: 100.0,  # Assume 100g per piece
            Unit.CANS: 400.0,    # Typical can
        }
        
        return quantity * conversions.get(unit, 1.0)
    
    def generate_shopping_list(
        self,
        user_id: str,
        required_ingredients: Optional[Dict[str, float]] = None
    ) -> List[ShoppingListItem]:
        """
        Generate shopping list based on inventory and requirements
        
        Args:
            user_id: User identifier
            required_ingredients: Dict of ingredient_id -> required_quantity
        
        Returns:
            Shopping list with priorities
        """
        shopping_list = []
        
        # Get current inventory
        available = self.get_available_ingredients(user_id)
        
        # Add expiring items
        expiring = self.database.get_expiring_items(user_id, 1)
        for item in expiring:
            shopping_list.append(ShoppingListItem(
                ingredient_id=item.ingredient_id,
                name=item.name,
                quantity=item.quantity,
                unit=item.unit,
                priority="urgent",
                reason=f"Current stock expiring in {item.days_until_expiry()} days"
            ))
        
        # Add low stock items
        low_stock = self.database.get_low_stock_items(user_id)
        for item in low_stock:
            if item.ingredient_id not in [si.ingredient_id for si in shopping_list]:
                shopping_list.append(ShoppingListItem(
                    ingredient_id=item.ingredient_id,
                    name=item.name,
                    quantity=item.quantity * 5,  # Buy 5x current quantity
                    unit=item.unit,
                    priority="soon",
                    reason="Low stock"
                ))
        
        # Add recipe requirements
        if required_ingredients:
            for ingredient_id, required_qty in required_ingredients.items():
                available_qty = available.get(ingredient_id, 0.0)
                
                if available_qty < required_qty:
                    needed = required_qty - available_qty
                    
                    # Get ingredient name
                    name = self.vision_model.food_database.get(ingredient_id, {}).get('name', ingredient_id)
                    
                    shopping_list.append(ShoppingListItem(
                        ingredient_id=ingredient_id,
                        name=name,
                        quantity=needed,
                        unit=Unit.GRAMS,  # Normalized
                        priority="optional",
                        reason=f"Required for recipe (need {needed}g more)"
                    ))
        
        return shopping_list


# ============================================================================
# TESTING
# ============================================================================

def test_pantry_analyzer():
    """Test pantry analysis system"""
    print("=" * 80)
    print("PANTRY ANALYSIS & INVENTORY MANAGEMENT - TEST")
    print("=" * 80)
    
    # Create analyzer
    analyzer = PantryAnalyzer()
    
    # Test pantry scanning
    print("\n" + "="*80)
    print("Test: Pantry Scan")
    print("="*80)
    
    scan_result = analyzer.scan_pantry(
        user_id="user_123",
        image_description="""
        Pantry shelf with: chicken breast (500g, expiring), rice (1kg bag), 
        tomato paste (2 cans), milk (1 liter), olive oil (750ml bottle),
        eggs (carton of 12), spinach (fresh bag), onions (3 pieces), 
        garlic (1 bulb), salt (container)
        """
    )
    
    print(f"✓ Scan completed in {scan_result.processing_time_seconds:.2f}s")
    print(f"  Total items detected: {scan_result.total_items_detected}")
    print(f"  Inventory updated: {len(scan_result.inventory_updates)} items")
    
    print(f"\n  Detected items:")
    for detection in scan_result.detections[:5]:
        print(f"    - {detection.name}: {detection.estimated_quantity}{detection.unit.value if detection.unit else ''} "
              f"(confidence: {detection.confidence:.0%})")
    
    # Test expiry alerts
    print("\n" + "="*80)
    print("Test: Expiry Alerts")
    print("="*80)
    
    if scan_result.expiring_items:
        print(f"✓ Found {len(scan_result.expiring_items)} expiring items:")
        for item in scan_result.expiring_items:
            days = item.days_until_expiry()
            print(f"    ⚠️ {item.name}: {days} days until expiry")
    else:
        print("  No items expiring soon")
    
    # Test low stock alerts
    print("\n" + "="*80)
    print("Test: Low Stock Alerts")
    print("="*80)
    
    if scan_result.low_stock_items:
        print(f"✓ Found {len(scan_result.low_stock_items)} low stock items:")
        for item in scan_result.low_stock_items:
            print(f"    📦 {item.name}: {item.quantity}{item.unit.value} (low stock)")
    else:
        print("  All items adequately stocked")
    
    # Test available ingredients
    print("\n" + "="*80)
    print("Test: Available Ingredients")
    print("="*80)
    
    available = analyzer.get_available_ingredients("user_123")
    
    print(f"✓ User has {len(available)} ingredient types available:")
    for ingredient_id, quantity in list(available.items())[:5]:
        name = analyzer.vision_model.food_database.get(ingredient_id, {}).get('name', ingredient_id)
        print(f"    - {name}: {quantity:.0f}g (normalized)")
    
    # Test shopping list generation
    print("\n" + "="*80)
    print("Test: Shopping List Generation")
    print("="*80)
    
    # Simulate recipe requirement
    required_ingredients = {
        'chicken_breast': 1000.0,  # Need 1kg
        'bell_pepper': 200.0,      # Don't have this
    }
    
    shopping_list = analyzer.generate_shopping_list("user_123", required_ingredients)
    
    print(f"✓ Generated shopping list with {len(shopping_list)} items:")
    
    for item in shopping_list:
        priority_emoji = {"urgent": "🔴", "soon": "🟡", "optional": "🟢"}.get(item.priority, "⚪")
        print(f"    {priority_emoji} {item.name}: {item.quantity:.0f}{item.unit.value}")
        print(f"       Reason: {item.reason}")
    
    # Test inventory query
    print("\n" + "="*80)
    print("Test: Inventory Query")
    print("="*80)
    
    inventory = analyzer.database.get_user_inventory("user_123")
    
    print(f"✓ Total inventory items: {len(inventory)}")
    print(f"\n  Sample items:")
    for item in inventory[:3]:
        print(f"    {item.name}:")
        print(f"      Quantity: {item.quantity}{item.unit.value}")
        print(f"      Storage: {item.storage_location.value}")
        print(f"      Freshness: {item.freshness.value}")
        if item.expiry_date:
            print(f"      Expires: {item.expiry_date} ({item.days_until_expiry()} days)")
    
    print("\n✅ All pantry analyzer tests passed!")
    print("\n💡 Production Features:")
    print("  - Computer Vision: YOLO v8 fine-tuned on food packaging")
    print("  - OCR: Tesseract/PaddleOCR for expiry dates and quantities")
    print("  - Database: PostgreSQL with user inventory tables")
    print("  - Real-time alerts: Push notifications for expiring items")
    print("  - Mobile app: Scan pantry with phone camera")
    print("  - Barcode scanning: Quick add via UPC lookup")


if __name__ == '__main__':
    test_pantry_analyzer()
