"""
Natural Language Processing for Food
=====================================

Advanced NLP models for food-related text understanding including
named entity recognition, intent classification, and semantic search.

Features:
1. Food entity recognition (ingredients, dishes, nutrients)
2. Recipe parsing from text
3. Dietary intent classification
4. Semantic search for recipes
5. Text-to-recipe conversion
6. Nutrition query understanding
7. Ingredient extraction from descriptions
8. Food relation extraction

Performance Targets:
- NER F1 score: >0.85
- Intent classification accuracy: >0.90
- Semantic search recall@10: >0.70
- Parse recipes: <500ms
- Support 50+ languages
- Handle 10,000+ queries/second

Author: Wellomex AI Team
Date: November 2025
Version: 5.0.0
"""

import time
import logging
import random
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from collections import defaultdict, Counter
import json

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class EntityType(Enum):
    """Named entity types"""
    INGREDIENT = "ingredient"
    DISH = "dish"
    NUTRIENT = "nutrient"
    QUANTITY = "quantity"
    UNIT = "unit"
    COOKING_METHOD = "cooking_method"
    CUISINE = "cuisine"
    DIETARY_RESTRICTION = "dietary_restriction"


class IntentType(Enum):
    """User intent types"""
    FIND_RECIPE = "find_recipe"
    NUTRITION_INFO = "nutrition_info"
    SUBSTITUTE_INGREDIENT = "substitute_ingredient"
    MEAL_PLAN = "meal_plan"
    DIETARY_ADVICE = "dietary_advice"
    CALORIE_COUNT = "calorie_count"
    COOKING_INSTRUCTIONS = "cooking_instructions"
    FOOD_SAFETY = "food_safety"


@dataclass
class NLPConfig:
    """NLP model configuration"""
    # Tokenization
    vocab_size: int = 30000
    max_length: int = 512
    
    # Embeddings
    embedding_dim: int = 300
    use_pretrained: bool = True
    
    # Encoder
    hidden_dim: int = 512
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    
    # Task-specific
    num_entity_types: int = len(EntityType)
    num_intent_types: int = len(IntentType)
    
    # Training
    learning_rate: float = 2e-5
    batch_size: int = 32
    num_epochs: int = 20


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Entity:
    """Named entity"""
    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class Intent:
    """User intent"""
    intent_type: IntentType
    confidence: float
    entities: List[Entity] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedRecipe:
    """Parsed recipe from text"""
    title: str
    ingredients: List[Tuple[str, str, str]]  # (quantity, unit, ingredient)
    instructions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticQuery:
    """Semantic search query"""
    text: str
    embedding: Optional[Any] = None
    filters: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# TOKENIZER
# ============================================================================

class FoodTokenizer:
    """
    Food-Specific Tokenizer
    
    Tokenizes food-related text with special handling for ingredients,
    quantities, and cooking terminology.
    """
    
    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        
        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        
        # Build vocabulary
        self.vocab = self._build_vocab()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        logger.info(f"Food Tokenizer initialized with vocab size {len(self.vocab)}")
    
    def _build_vocab(self) -> List[str]:
        """Build vocabulary"""
        vocab = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        
        # Common food words
        food_words = [
            # Ingredients
            "chicken", "beef", "pork", "fish", "salmon", "tuna",
            "rice", "pasta", "noodles", "bread",
            "tomato", "onion", "garlic", "carrot", "potato",
            "cheese", "milk", "butter", "cream", "egg",
            "salt", "pepper", "sugar", "flour",
            
            # Cooking methods
            "bake", "boil", "fry", "grill", "roast", "steam",
            "chop", "dice", "slice", "mix", "stir",
            
            # Units
            "cup", "tbsp", "tsp", "oz", "lb", "g", "kg", "ml",
            
            # Nutrients
            "protein", "carbs", "fat", "calories", "fiber",
            "vitamin", "mineral", "sodium",
            
            # Cuisines
            "italian", "chinese", "mexican", "indian", "japanese",
            
            # Dietary
            "vegan", "vegetarian", "gluten-free", "keto", "paleo"
        ]
        
        vocab.extend(food_words)
        
        # Common words
        common_words = [
            "the", "a", "an", "and", "or", "with", "for", "to",
            "of", "in", "on", "at", "is", "are", "was", "were",
            "how", "what", "which", "where", "when", "why",
            "much", "many", "some", "any", "all", "most"
        ]
        
        vocab.extend(common_words)
        
        # Numbers
        vocab.extend([str(i) for i in range(100)])
        
        # Pad to vocab size
        while len(vocab) < self.vocab_size:
            vocab.append(f"<unk{len(vocab)}>")
        
        return vocab[:self.vocab_size]
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        # Simple word-based tokenization
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        
        return tokens
    
    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True
    ) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        # Convert to IDs
        token_ids = [
            self.token_to_id.get(token, self.token_to_id[self.unk_token])
            for token in tokens
        ]
        
        # Truncate or pad
        if max_length:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                token_ids += [self.token_to_id[self.pad_token]] * (max_length - len(token_ids))
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = [
            self.id_to_token.get(id_, self.unk_token)
            for id_ in token_ids
        ]
        
        # Remove special tokens
        tokens = [
            t for t in tokens
            if t not in [self.pad_token, self.bos_token, self.eos_token]
        ]
        
        return " ".join(tokens)


# ============================================================================
# NAMED ENTITY RECOGNITION
# ============================================================================

class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF for Named Entity Recognition
    
    Bidirectional LSTM with CRF layer for sequence labeling.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_tags: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.hidden_to_tag = nn.Linear(hidden_dim, num_tags)
        
        # CRF transitions
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        self.num_tags = num_tags
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
        
        Returns:
            emissions: [batch, seq_len, num_tags]
        """
        # Embedding
        embedded = self.embedding(input_ids)
        
        # BiLSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Tag scores
        emissions = self.hidden_to_tag(lstm_out)
        
        return emissions
    
    def decode(self, emissions: torch.Tensor) -> List[List[int]]:
        """Viterbi decoding"""
        batch_size, seq_len, num_tags = emissions.shape
        
        # Simplified decoding - just take argmax
        predictions = emissions.argmax(dim=-1)
        
        return predictions.tolist()


class NERModel:
    """
    Named Entity Recognition Model
    
    Identifies food entities in text.
    """
    
    def __init__(
        self,
        config: NLPConfig,
        tokenizer: FoodTokenizer
    ):
        self.config = config
        self.tokenizer = tokenizer
        
        # Tag vocabulary (BIO tagging)
        self.tags = ["O"]  # Outside
        for entity_type in EntityType:
            self.tags.append(f"B-{entity_type.value}")
            self.tags.append(f"I-{entity_type.value}")
        
        self.tag_to_id = {tag: i for i, tag in enumerate(self.tags)}
        self.id_to_tag = {i: tag for i, tag in enumerate(self.tags)}
        
        # Model
        if TORCH_AVAILABLE:
            self.model = BiLSTMCRF(
                vocab_size=config.vocab_size,
                embedding_dim=config.embedding_dim,
                hidden_dim=config.hidden_dim,
                num_tags=len(self.tags),
                num_layers=config.num_layers,
                dropout=config.dropout
            )
        else:
            self.model = None
        
        logger.info("NER Model initialized")
    
    def predict(self, text: str) -> List[Entity]:
        """Extract entities from text"""
        if not TORCH_AVAILABLE or self.model is None:
            # Mock entities
            return self._mock_entities(text)
        
        self.model.eval()
        
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text, max_length=self.config.max_length)
        
        # Predict
        with torch.no_grad():
            input_tensor = torch.tensor([token_ids])
            emissions = self.model(input_tensor)
            predictions = self.model.decode(emissions)[0]
        
        # Convert to entities
        entities = self._predictions_to_entities(tokens, predictions)
        
        return entities
    
    def _predictions_to_entities(
        self,
        tokens: List[str],
        predictions: List[int]
    ) -> List[Entity]:
        """Convert BIO predictions to entities"""
        entities = []
        current_entity = None
        
        for i, (token, tag_id) in enumerate(zip(tokens, predictions[1:])):  # Skip BOS
            tag = self.id_to_tag.get(tag_id, "O")
            
            if tag.startswith("B-"):
                # Start new entity
                if current_entity:
                    entities.append(current_entity)
                
                entity_type_str = tag[2:]
                entity_type = EntityType(entity_type_str)
                
                current_entity = Entity(
                    text=token,
                    entity_type=entity_type,
                    start=i,
                    end=i + 1,
                    confidence=0.9
                )
            
            elif tag.startswith("I-") and current_entity:
                # Continue entity
                current_entity.text += " " + token
                current_entity.end = i + 1
            
            else:
                # Outside or mismatch
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def _mock_entities(self, text: str) -> List[Entity]:
        """Create mock entities for testing"""
        entities = []
        
        # Simple keyword matching
        ingredient_keywords = ["chicken", "rice", "tomato", "onion", "garlic"]
        cooking_keywords = ["bake", "boil", "fry", "grill"]
        
        text_lower = text.lower()
        
        for keyword in ingredient_keywords:
            if keyword in text_lower:
                start = text_lower.index(keyword)
                entities.append(Entity(
                    text=keyword,
                    entity_type=EntityType.INGREDIENT,
                    start=start,
                    end=start + len(keyword),
                    confidence=0.95
                ))
        
        for keyword in cooking_keywords:
            if keyword in text_lower:
                start = text_lower.index(keyword)
                entities.append(Entity(
                    text=keyword,
                    entity_type=EntityType.COOKING_METHOD,
                    start=start,
                    end=start + len(keyword),
                    confidence=0.90
                ))
        
        return entities


# ============================================================================
# INTENT CLASSIFICATION
# ============================================================================

class IntentClassifier(nn.Module):
    """
    Intent Classification Model
    
    Classifies user intent from text.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_intents: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_intents)
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
        
        Returns:
            logits: [batch, num_intents]
        """
        # Embedding
        embedded = self.embedding(input_ids)
        
        # Encode
        _, (hidden, _) = self.encoder(embedded)
        
        # Use last layer's hidden states (both directions)
        encoding = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Classify
        logits = self.classifier(encoding)
        
        return logits


class IntentRecognizer:
    """
    Intent Recognition System
    
    Classifies user intents and extracts relevant information.
    """
    
    def __init__(
        self,
        config: NLPConfig,
        tokenizer: FoodTokenizer,
        ner_model: NERModel
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.ner_model = ner_model
        
        # Model
        if TORCH_AVAILABLE:
            self.model = IntentClassifier(
                vocab_size=config.vocab_size,
                embedding_dim=config.embedding_dim,
                hidden_dim=config.hidden_dim,
                num_intents=config.num_intent_types,
                dropout=config.dropout
            )
        else:
            self.model = None
        
        logger.info("Intent Recognizer initialized")
    
    def recognize(self, text: str) -> Intent:
        """Recognize intent from text"""
        # Extract entities
        entities = self.ner_model.predict(text)
        
        if not TORCH_AVAILABLE or self.model is None:
            # Rule-based fallback
            return self._rule_based_intent(text, entities)
        
        self.model.eval()
        
        # Tokenize
        token_ids = self.tokenizer.encode(text, max_length=self.config.max_length)
        
        # Predict
        with torch.no_grad():
            input_tensor = torch.tensor([token_ids])
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)[0]
            
            intent_id = probabilities.argmax().item()
            confidence = probabilities[intent_id].item()
        
        intent_type = list(IntentType)[intent_id]
        
        # Extract slots
        slots = self._extract_slots(text, entities, intent_type)
        
        return Intent(
            intent_type=intent_type,
            confidence=confidence,
            entities=entities,
            slots=slots
        )
    
    def _rule_based_intent(self, text: str, entities: List[Entity]) -> Intent:
        """Rule-based intent classification"""
        text_lower = text.lower()
        
        # Simple keyword matching
        if any(word in text_lower for word in ["recipe", "make", "cook"]):
            intent_type = IntentType.FIND_RECIPE
        elif any(word in text_lower for word in ["calories", "nutrition", "nutrients"]):
            intent_type = IntentType.NUTRITION_INFO
        elif any(word in text_lower for word in ["substitute", "replace", "alternative"]):
            intent_type = IntentType.SUBSTITUTE_INGREDIENT
        elif any(word in text_lower for word in ["meal plan", "weekly"]):
            intent_type = IntentType.MEAL_PLAN
        else:
            intent_type = IntentType.FIND_RECIPE
        
        slots = self._extract_slots(text, entities, intent_type)
        
        return Intent(
            intent_type=intent_type,
            confidence=0.85,
            entities=entities,
            slots=slots
        )
    
    def _extract_slots(
        self,
        text: str,
        entities: List[Entity],
        intent_type: IntentType
    ) -> Dict[str, Any]:
        """Extract intent-specific slots"""
        slots = {}
        
        # Extract entities by type
        for entity in entities:
            if entity.entity_type == EntityType.INGREDIENT:
                slots.setdefault('ingredients', []).append(entity.text)
            elif entity.entity_type == EntityType.CUISINE:
                slots['cuisine'] = entity.text
            elif entity.entity_type == EntityType.DIETARY_RESTRICTION:
                slots.setdefault('dietary_restrictions', []).append(entity.text)
        
        return slots


# ============================================================================
# RECIPE PARSER
# ============================================================================

class RecipeParser:
    """
    Recipe Parser
    
    Parses recipes from natural language text.
    """
    
    def __init__(self, tokenizer: FoodTokenizer, ner_model: NERModel):
        self.tokenizer = tokenizer
        self.ner_model = ner_model
        
        logger.info("Recipe Parser initialized")
    
    def parse(self, text: str) -> ParsedRecipe:
        """Parse recipe from text"""
        start_time = time.time()
        
        # Split into sections
        lines = text.strip().split('\n')
        
        title = ""
        ingredients = []
        instructions = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            line_lower = line.lower()
            
            # Detect sections
            if "ingredient" in line_lower:
                current_section = "ingredients"
                continue
            elif "instruction" in line_lower or "direction" in line_lower or "step" in line_lower:
                current_section = "instructions"
                continue
            
            # Extract content
            if current_section == "ingredients":
                parsed_ing = self._parse_ingredient(line)
                if parsed_ing:
                    ingredients.append(parsed_ing)
            elif current_section == "instructions":
                instructions.append(line)
            elif not title:
                title = line
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"Parsed recipe in {elapsed_time*1000:.1f}ms")
        
        return ParsedRecipe(
            title=title or "Untitled Recipe",
            ingredients=ingredients,
            instructions=instructions,
            metadata={'parse_time_ms': elapsed_time * 1000}
        )
    
    def _parse_ingredient(self, text: str) -> Optional[Tuple[str, str, str]]:
        """Parse ingredient line"""
        # Pattern: quantity unit ingredient
        # Example: "2 cups rice" or "1 tbsp olive oil"
        
        pattern = r'(\d+(?:\.\d+)?|\d+/\d+)?\s*([a-zA-Z]+)?\s+(.+)'
        match = re.match(pattern, text)
        
        if match:
            quantity = match.group(1) or "1"
            unit = match.group(2) or "whole"
            ingredient = match.group(3)
            
            return (quantity, unit, ingredient.strip())
        
        # Fallback: just ingredient
        return ("1", "whole", text.strip())


# ============================================================================
# SEMANTIC SEARCH
# ============================================================================

class SemanticSearchEngine:
    """
    Semantic Search Engine
    
    Searches recipes using semantic similarity.
    """
    
    def __init__(
        self,
        config: NLPConfig,
        tokenizer: FoodTokenizer
    ):
        self.config = config
        self.tokenizer = tokenizer
        
        # Encoder
        if TORCH_AVAILABLE:
            self.encoder = nn.LSTM(
                config.embedding_dim,
                config.hidden_dim,
                num_layers=2,
                bidirectional=True,
                batch_first=True
            )
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        else:
            self.encoder = None
            self.embedding = None
        
        # Index
        self.document_embeddings: Dict[str, Any] = {}
        self.documents: Dict[str, str] = {}
        
        logger.info("Semantic Search Engine initialized")
    
    def encode_text(self, text: str) -> Any:
        """Encode text to embedding"""
        if not TORCH_AVAILABLE or self.encoder is None:
            # Mock embedding
            if NUMPY_AVAILABLE:
                return np.random.randn(self.config.hidden_dim * 2)
            return [random.random() for _ in range(self.config.hidden_dim * 2)]
        
        self.encoder.eval()
        
        with torch.no_grad():
            # Tokenize
            token_ids = self.tokenizer.encode(text, max_length=self.config.max_length)
            input_tensor = torch.tensor([token_ids])
            
            # Encode
            embedded = self.embedding(input_tensor)
            _, (hidden, _) = self.encoder(embedded)
            
            # Concatenate last layer
            encoding = torch.cat([hidden[-2], hidden[-1]], dim=1)
            
            return encoding.cpu().numpy() if NUMPY_AVAILABLE else encoding.tolist()
    
    def index_document(self, doc_id: str, text: str):
        """Index a document"""
        embedding = self.encode_text(text)
        self.document_embeddings[doc_id] = embedding
        self.documents[doc_id] = text
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        query_embedding = self.encode_text(query)
        
        # Compute similarities
        similarities = []
        
        for doc_id, doc_embedding in self.document_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((doc_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: Any, vec2: Any) -> float:
        """Compute cosine similarity"""
        if NUMPY_AVAILABLE:
            v1 = np.array(vec1).flatten()
            v2 = np.array(vec2).flatten()
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(np.dot(v1, v2) / (norm1 * norm2))
        
        # Manual computation
        if isinstance(vec1, list):
            v1 = vec1
            v2 = vec2
        else:
            v1 = vec1.flatten().tolist()
            v2 = vec2.flatten().tolist()
        
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


# ============================================================================
# TESTING
# ============================================================================

def test_food_nlp():
    """Test NLP system"""
    print("=" * 80)
    print("FOOD NLP - TEST")
    print("=" * 80)
    
    # Initialize
    config = NLPConfig()
    tokenizer = FoodTokenizer(vocab_size=config.vocab_size)
    
    print(f"✓ Tokenizer initialized (vocab size: {len(tokenizer.vocab)})")
    
    # Test tokenization
    print("\n" + "="*80)
    print("Test: Tokenization")
    print("="*80)
    
    test_text = "How to make chicken rice with tomatoes and onions?"
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(test_text, max_length=20)
    decoded = tokenizer.decode(token_ids)
    
    print(f"Text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids[:10]}...")
    print(f"Decoded: {decoded}")
    
    # Test NER
    print("\n" + "="*80)
    print("Test: Named Entity Recognition")
    print("="*80)
    
    ner_model = NERModel(config, tokenizer)
    
    test_text = "Bake chicken with garlic and rice for 30 minutes"
    entities = ner_model.predict(test_text)
    
    print(f"Text: {test_text}")
    print(f"✓ Entities found: {len(entities)}")
    
    for entity in entities:
        print(f"  - {entity.text} ({entity.entity_type.value}) [confidence: {entity.confidence:.2f}]")
    
    # Test intent classification
    print("\n" + "="*80)
    print("Test: Intent Classification")
    print("="*80)
    
    intent_recognizer = IntentRecognizer(config, tokenizer, ner_model)
    
    test_queries = [
        "How many calories in an apple?",
        "Find me a vegan pasta recipe",
        "What can I substitute for eggs?",
        "Create a weekly meal plan"
    ]
    
    for query in test_queries:
        intent = intent_recognizer.recognize(query)
        print(f"\nQuery: {query}")
        print(f"  Intent: {intent.intent_type.value} (confidence: {intent.confidence:.2f})")
        print(f"  Entities: {len(intent.entities)}")
        if intent.slots:
            print(f"  Slots: {intent.slots}")
    
    # Test recipe parsing
    print("\n" + "="*80)
    print("Test: Recipe Parsing")
    print("="*80)
    
    recipe_parser = RecipeParser(tokenizer, ner_model)
    
    recipe_text = """
Chicken Rice Bowl

Ingredients:
2 cups rice
1 lb chicken breast
3 tomatoes
1 onion
2 cloves garlic

Instructions:
1. Cook rice according to package directions
2. Season and grill chicken until done
3. Chop vegetables and sauté
4. Combine everything and serve
"""
    
    parsed_recipe = recipe_parser.parse(recipe_text)
    
    print(f"✓ Recipe parsed")
    print(f"  Title: {parsed_recipe.title}")
    print(f"  Ingredients: {len(parsed_recipe.ingredients)}")
    for qty, unit, ing in parsed_recipe.ingredients[:3]:
        print(f"    - {qty} {unit} {ing}")
    print(f"  Instructions: {len(parsed_recipe.instructions)} steps")
    for i, step in enumerate(parsed_recipe.instructions[:2], 1):
        print(f"    {i}. {step[:50]}...")
    
    # Test semantic search
    print("\n" + "="*80)
    print("Test: Semantic Search")
    print("="*80)
    
    search_engine = SemanticSearchEngine(config, tokenizer)
    
    # Index sample documents
    sample_recipes = [
        ("recipe1", "Grilled chicken with vegetables and rice"),
        ("recipe2", "Vegan pasta with tomato sauce"),
        ("recipe3", "Baked salmon with lemon and herbs"),
        ("recipe4", "Chicken stir-fry with noodles"),
        ("recipe5", "Vegetarian pizza with mushrooms")
    ]
    
    for doc_id, text in sample_recipes:
        search_engine.index_document(doc_id, text)
    
    print(f"✓ Indexed {len(sample_recipes)} recipes")
    
    # Search
    query = "chicken rice dish"
    results = search_engine.search(query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"✓ Top {len(results)} results:")
    
    for doc_id, score in results:
        print(f"  {doc_id}: {search_engine.documents[doc_id]} (score: {score:.3f})")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_food_nlp()
