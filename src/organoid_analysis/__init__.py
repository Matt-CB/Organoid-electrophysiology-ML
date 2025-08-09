"""
Organoid Intelligence Analysis Package

A comprehensive toolkit for analyzing neural recordings from brain organoids,
including simulation, feature extraction, classification, and visualization.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.simulator import OrganoidRecordingSimulator
from .core.feature_extractor import EnhancedFeatureExtractor
from .core.classifier import OrganoidClassifier
from .utils.visualization import OrganoidVisualizer
from .utils.data_processing import DataProcessor
from .analysis.pipeline import OrganoidAnalysisPipeline

__all__ = [
    "OrganoidRecordingSimulator",
    "EnhancedFeatureExtractor", 
    "OrganoidClassifier",
    "OrganoidVisualizer",
    "DataProcessor",
    "OrganoidAnalysisPipeline"
]