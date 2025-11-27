"""
LLM Orchestration Module
========================

Conversational AI integration for complete nutrition system.

Components:
- llm_orchestrator: Main LLM service with function calling
- function_handler: Integration with all system services
- training_pipeline: Data collection for fine-tuning

Author: Wellomex AI Team
Date: November 2025
"""

from .llm_orchestrator import (
    LLMOrchestrator,
    LLMConfig,
    LLMProvider,
    ConversationMode,
    ConversationSession,
    Message,
    FunctionCall
)

from .training_pipeline import (
    TrainingDataPipeline,
    PerformanceMonitor,
    MetricType,
    GoalCategory
)

__all__ = [
    "LLMOrchestrator",
    "LLMConfig",
    "LLMProvider",
    "ConversationMode",
    "ConversationSession",
    "Message",
    "FunctionCall",
    "TrainingDataPipeline",
    "PerformanceMonitor",
    "MetricType",
    "GoalCategory"
]
