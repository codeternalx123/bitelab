"""
Training Data Pipeline & Performance Monitoring
===============================================

Collects and processes training data from:
- Conversation logs
- Food scan accuracy
- Recommendation acceptance rates
- Medication interaction catches
- Recipe generation quality
- Meal plan adherence
- Portion estimation accuracy

Tracks performance across:
- 55+ health goals
- All disease conditions
- 100+ medication interactions
- Food safety warnings
- Alternative suggestions

Generates fine-tuning datasets for custom LLM training.

Author: Wellomex AI Team
Date: November 2025
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import time

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class MetricType(Enum):
    """Performance metric types"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    USER_SATISFACTION = "user_satisfaction"
    FUNCTION_CALL_SUCCESS = "function_call_success"
    RESPONSE_TIME_MS = "response_time_ms"


class GoalCategory(Enum):
    """Health goal categories"""
    WEIGHT_MANAGEMENT = "weight_management"
    METABOLIC_HEALTH = "metabolic_health"
    CARDIOVASCULAR = "cardiovascular"
    DIGESTIVE = "digestive"
    IMMUNE = "immune"
    COGNITIVE = "cognitive"
    BONE_HEALTH = "bone_health"
    INFLAMMATION = "inflammation"
    ATHLETIC_PERFORMANCE = "athletic_performance"
    LONGEVITY = "longevity"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PerformanceMetric:
    """Single performance metric"""
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Context
    health_goal: Optional[str] = None
    disease_condition: Optional[str] = None
    function_name: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingDatapoint:
    """Single training datapoint for fine-tuning"""
    # Input
    user_message: str
    system_context: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    
    # Output
    assistant_response: str
    function_calls: List[Dict[str, Any]]
    
    # Quality metrics
    user_rating: float  # 1-5
    outcome_success: bool
    response_time_ms: float
    
    # Categorization
    health_goals: List[str]
    diseases: List[str]
    medications: List[str]
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = ""
    user_id: str = ""


@dataclass
class GoalPerformance:
    """Performance tracking for specific health goal"""
    goal_name: str
    category: GoalCategory
    
    # Metrics
    total_interactions: int = 0
    successful_outcomes: int = 0
    avg_user_rating: float = 0.0
    avg_response_time_ms: float = 0.0
    
    # Specific to goal
    recommendations_accepted: int = 0
    recommendations_rejected: int = 0
    warnings_issued: int = 0
    alternatives_provided: int = 0
    
    # Performance over time
    metrics_by_day: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))


@dataclass
class DiseasePerformance:
    """Performance tracking for disease management"""
    disease_name: str
    
    # Metrics
    total_assessments: int = 0
    risks_detected: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Safety
    critical_warnings: int = 0
    medication_interactions_caught: int = 0
    safe_alternatives_provided: int = 0
    
    # Accuracy
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


# ============================================================================
# TRAINING DATA PIPELINE
# ============================================================================

class TrainingDataPipeline:
    """
    Collects, processes, and formats training data
    """
    
    def __init__(self, data_dir: str = "data/llm_training"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.datasets_dir = os.path.join(data_dir, "datasets")
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.datasets_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # In-memory storage
        self.datapoints: List[TrainingDatapoint] = []
        
        logger.info(f"Training pipeline initialized: {data_dir}")
    
    # ========================================================================
    # DATA COLLECTION
    # ========================================================================
    
    def collect_conversation(
        self,
        session_id: str,
        user_id: str,
        messages: List[Dict[str, str]],
        function_calls: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        user_rating: float,
        outcome_success: bool,
        response_time_ms: float
    ) -> TrainingDatapoint:
        """Collect conversation for training"""
        
        # Extract last user message and assistant response
        user_msg = ""
        assistant_msg = ""
        
        for msg in reversed(messages):
            if msg["role"] == "user" and not user_msg:
                user_msg = msg["content"]
            elif msg["role"] == "assistant" and not assistant_msg:
                assistant_msg = msg["content"]
            
            if user_msg and assistant_msg:
                break
        
        # Build system context
        system_context = {
            "health_conditions": user_profile.get("health_conditions", []),
            "medications": user_profile.get("medications", []),
            "allergies": user_profile.get("allergies", []),
            "health_goals": user_profile.get("health_goals", []),
            "dietary_preferences": user_profile.get("dietary_preferences", [])
        }
        
        # Create datapoint
        datapoint = TrainingDatapoint(
            user_message=user_msg,
            system_context=system_context,
            conversation_history=messages[:-1],  # Exclude last exchange
            assistant_response=assistant_msg,
            function_calls=function_calls,
            user_rating=user_rating,
            outcome_success=outcome_success,
            response_time_ms=response_time_ms,
            health_goals=user_profile.get("health_goals", []),
            diseases=user_profile.get("health_conditions", []),
            medications=user_profile.get("medications", []),
            session_id=session_id,
            user_id=user_id
        )
        
        # Store
        self.datapoints.append(datapoint)
        
        # Save to disk
        self._save_raw_datapoint(datapoint)
        
        logger.info(f"Collected training datapoint: session={session_id}, rating={user_rating}")
        
        return datapoint
    
    def _save_raw_datapoint(self, datapoint: TrainingDatapoint):
        """Save raw datapoint to disk"""
        filename = f"{datapoint.timestamp.strftime('%Y%m%d_%H%M%S')}_{datapoint.user_rating:.1f}_{datapoint.session_id}.json"
        filepath = os.path.join(self.raw_dir, filename)
        
        # Convert to dict
        data = {
            "user_message": datapoint.user_message,
            "system_context": datapoint.system_context,
            "assistant_response": datapoint.assistant_response,
            "function_calls": datapoint.function_calls,
            "user_rating": datapoint.user_rating,
            "outcome_success": datapoint.outcome_success,
            "response_time_ms": datapoint.response_time_ms,
            "health_goals": datapoint.health_goals,
            "diseases": datapoint.diseases,
            "medications": datapoint.medications,
            "timestamp": datapoint.timestamp.isoformat(),
            "session_id": datapoint.session_id,
            "user_id": datapoint.user_id
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    # ========================================================================
    # DATA PROCESSING
    # ========================================================================
    
    def process_training_data(
        self,
        min_rating: float = 4.0,
        min_datapoints: int = 100,
        balance_by_goal: bool = True
    ) -> Dict[str, Any]:
        """Process raw data into training dataset"""
        
        logger.info("Processing training data...")
        
        # Load all raw datapoints
        self._load_raw_datapoints()
        
        # Filter by quality
        high_quality = [
            dp for dp in self.datapoints
            if dp.user_rating >= min_rating and dp.outcome_success
        ]
        
        logger.info(f"High quality datapoints: {len(high_quality)} / {len(self.datapoints)}")
        
        if len(high_quality) < min_datapoints:
            logger.warning(f"Insufficient data: {len(high_quality)} < {min_datapoints}")
            return {
                "status": "insufficient_data",
                "available": len(high_quality),
                "required": min_datapoints
            }
        
        # Balance by health goals if requested
        if balance_by_goal:
            high_quality = self._balance_by_goals(high_quality)
        
        # Convert to training format
        training_examples = []
        for dp in high_quality:
            example = self._format_for_training(dp)
            training_examples.append(example)
        
        # Save processed dataset
        dataset_path = self._save_processed_dataset(training_examples)
        
        # Generate statistics
        stats = self._generate_dataset_stats(high_quality)
        
        return {
            "status": "success",
            "dataset_path": dataset_path,
            "total_examples": len(training_examples),
            "stats": stats
        }
    
    def _load_raw_datapoints(self):
        """Load all raw datapoints from disk"""
        self.datapoints = []
        
        for filename in os.listdir(self.raw_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.raw_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    # Convert to datapoint
                    dp = TrainingDatapoint(
                        user_message=data["user_message"],
                        system_context=data["system_context"],
                        conversation_history=data.get("conversation_history", []),
                        assistant_response=data["assistant_response"],
                        function_calls=data["function_calls"],
                        user_rating=data["user_rating"],
                        outcome_success=data["outcome_success"],
                        response_time_ms=data["response_time_ms"],
                        health_goals=data.get("health_goals", []),
                        diseases=data.get("diseases", []),
                        medications=data.get("medications", []),
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        session_id=data["session_id"],
                        user_id=data["user_id"]
                    )
                    
                    self.datapoints.append(dp)
        
        logger.info(f"Loaded {len(self.datapoints)} raw datapoints")
    
    def _balance_by_goals(self, datapoints: List[TrainingDatapoint]) -> List[TrainingDatapoint]:
        """Balance dataset by health goals"""
        
        # Group by goal
        by_goal = defaultdict(list)
        for dp in datapoints:
            for goal in dp.health_goals:
                by_goal[goal].append(dp)
        
        # Find min count
        min_count = min(len(dps) for dps in by_goal.values()) if by_goal else 0
        
        if min_count == 0:
            return datapoints
        
        # Sample equally from each goal
        balanced = []
        for goal, dps in by_goal.items():
            # Take min_count samples
            sampled = dps[:min_count]
            balanced.extend(sampled)
        
        logger.info(f"Balanced dataset: {len(datapoints)} -> {len(balanced)} examples")
        
        return balanced
    
    def _format_for_training(self, datapoint: TrainingDatapoint) -> Dict[str, Any]:
        """Format datapoint for LLM fine-tuning (OpenAI format)"""
        
        # Build system message
        system_msg = "You are Wellomex AI, an advanced nutrition assistant."
        
        if datapoint.system_context:
            system_msg += "\n\nUser Profile:"
            if datapoint.system_context.get("health_conditions"):
                system_msg += f"\n- Conditions: {', '.join(datapoint.system_context['health_conditions'])}"
            if datapoint.system_context.get("medications"):
                system_msg += f"\n- Medications: {', '.join(datapoint.system_context['medications'])}"
            if datapoint.system_context.get("health_goals"):
                system_msg += f"\n- Goals: {', '.join(datapoint.system_context['health_goals'])}"
        
        # Build messages
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": datapoint.user_message},
            {"role": "assistant", "content": datapoint.assistant_response}
        ]
        
        # Add function calls if present
        if datapoint.function_calls:
            # OpenAI function calling format
            for func_call in datapoint.function_calls:
                messages.append({
                    "role": "function",
                    "name": func_call.get("name", "unknown"),
                    "content": json.dumps(func_call.get("result", {}))
                })
        
        return {"messages": messages}
    
    def _save_processed_dataset(self, examples: List[Dict[str, Any]]) -> str:
        """Save processed dataset"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"dataset_{timestamp}_{len(examples)}.jsonl"
        filepath = os.path.join(self.datasets_dir, filename)
        
        # Write JSONL format (one JSON object per line)
        with open(filepath, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Saved processed dataset: {filepath}")
        
        return filepath
    
    def _generate_dataset_stats(self, datapoints: List[TrainingDatapoint]) -> Dict[str, Any]:
        """Generate dataset statistics"""
        
        # Count by goal
        goal_counts = defaultdict(int)
        for dp in datapoints:
            for goal in dp.health_goals:
                goal_counts[goal] += 1
        
        # Count by disease
        disease_counts = defaultdict(int)
        for dp in datapoints:
            for disease in dp.diseases:
                disease_counts[disease] += 1
        
        # Average metrics
        avg_rating = sum(dp.user_rating for dp in datapoints) / len(datapoints)
        avg_response_time = sum(dp.response_time_ms for dp in datapoints) / len(datapoints)
        success_rate = sum(1 for dp in datapoints if dp.outcome_success) / len(datapoints)
        
        return {
            "total_examples": len(datapoints),
            "goal_distribution": dict(goal_counts),
            "disease_distribution": dict(disease_counts),
            "avg_user_rating": round(avg_rating, 2),
            "avg_response_time_ms": round(avg_response_time, 2),
            "success_rate": round(success_rate, 2),
            "unique_users": len(set(dp.user_id for dp in datapoints)),
            "unique_sessions": len(set(dp.session_id for dp in datapoints))
        }


# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """
    Tracks performance across all health goals and diseases
    """
    
    def __init__(self):
        # Goal tracking
        self.goal_performance: Dict[str, GoalPerformance] = {}
        
        # Disease tracking
        self.disease_performance: Dict[str, DiseasePerformance] = {}
        
        # Overall metrics
        self.metrics: List[PerformanceMetric] = []
        
        logger.info("Performance monitor initialized")
    
    # ========================================================================
    # METRIC RECORDING
    # ========================================================================
    
    def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        health_goal: Optional[str] = None,
        disease: Optional[str] = None,
        function_name: Optional[str] = None,
        **metadata
    ):
        """Record a performance metric"""
        
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            health_goal=health_goal,
            disease_condition=disease,
            function_name=function_name,
            metadata=metadata
        )
        
        self.metrics.append(metric)
        
        # Update goal performance
        if health_goal:
            self._update_goal_performance(health_goal, metric_type, value)
        
        # Update disease performance
        if disease:
            self._update_disease_performance(disease, metric_type, value)
    
    def _update_goal_performance(self, goal: str, metric_type: MetricType, value: float):
        """Update goal-specific performance"""
        if goal not in self.goal_performance:
            self.goal_performance[goal] = GoalPerformance(
                goal_name=goal,
                category=self._categorize_goal(goal)
            )
        
        perf = self.goal_performance[goal]
        perf.total_interactions += 1
        
        if metric_type == MetricType.USER_SATISFACTION:
            # Update rolling average
            n = perf.total_interactions
            perf.avg_user_rating = ((perf.avg_user_rating * (n - 1)) + value) / n
    
    def _update_disease_performance(self, disease: str, metric_type: MetricType, value: float):
        """Update disease-specific performance"""
        if disease not in self.disease_performance:
            self.disease_performance[disease] = DiseasePerformance(disease_name=disease)
        
        perf = self.disease_performance[disease]
        perf.total_assessments += 1
    
    def _categorize_goal(self, goal: str) -> GoalCategory:
        """Categorize health goal"""
        goal_lower = goal.lower()
        
        if "weight" in goal_lower:
            return GoalCategory.WEIGHT_MANAGEMENT
        elif "heart" in goal_lower or "cardiovascular" in goal_lower:
            return GoalCategory.CARDIOVASCULAR
        elif "diabetes" in goal_lower or "blood_sugar" in goal_lower:
            return GoalCategory.METABOLIC_HEALTH
        elif "immune" in goal_lower:
            return GoalCategory.IMMUNE
        elif "brain" in goal_lower or "cognitive" in goal_lower:
            return GoalCategory.COGNITIVE
        elif "bone" in goal_lower:
            return GoalCategory.BONE_HEALTH
        elif "inflammation" in goal_lower:
            return GoalCategory.INFLAMMATION
        else:
            return GoalCategory.LONGEVITY
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_metrics": len(self.metrics),
            "goals_tracked": len(self.goal_performance),
            "diseases_tracked": len(self.disease_performance),
            
            "goal_performance": {
                goal: {
                    "total_interactions": perf.total_interactions,
                    "avg_rating": round(perf.avg_user_rating, 2),
                    "success_rate": round(perf.successful_outcomes / max(perf.total_interactions, 1), 2)
                }
                for goal, perf in self.goal_performance.items()
            },
            
            "disease_performance": {
                disease: {
                    "total_assessments": perf.total_assessments,
                    "risks_detected": perf.risks_detected,
                    "precision": round(perf.precision, 2),
                    "recall": round(perf.recall, 2)
                }
                for disease, perf in self.disease_performance.items()
            },
            
            "top_goals_by_usage": self._get_top_goals(5),
            "top_diseases_by_assessments": self._get_top_diseases(5)
        }
    
    def _get_top_goals(self, n: int) -> List[Dict[str, Any]]:
        """Get top N goals by usage"""
        sorted_goals = sorted(
            self.goal_performance.items(),
            key=lambda x: x[1].total_interactions,
            reverse=True
        )
        
        return [
            {
                "goal": goal,
                "interactions": perf.total_interactions,
                "avg_rating": round(perf.avg_user_rating, 2)
            }
            for goal, perf in sorted_goals[:n]
        ]
    
    def _get_top_diseases(self, n: int) -> List[Dict[str, Any]]:
        """Get top N diseases by assessments"""
        sorted_diseases = sorted(
            self.disease_performance.items(),
            key=lambda x: x[1].total_assessments,
            reverse=True
        )
        
        return [
            {
                "disease": disease,
                "assessments": perf.total_assessments,
                "risks_detected": perf.risks_detected
            }
            for disease, perf in sorted_diseases[:n]
        ]


# Global instances
pipeline = TrainingDataPipeline()
monitor = PerformanceMonitor()
