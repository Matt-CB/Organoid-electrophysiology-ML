"""
Classifier module for organoid condition prediction.

This module provides machine learning capabilities for classifying organoid
conditions based on extracted features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, List, Optional, Tuple, Any


class OrganoidClassifier:
    """Enhanced organoid condition classifier with robust ML pipeline."""
    
    def __init__(self, use_scaler: bool = True, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize the organoid classifier.
        
        Args:
            use_scaler: Whether to use feature scaling
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.use_scaler = use_scaler
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.pipeline = None
        self.feature_names = None
        
    def build_pipeline(self, **classifier_params) -> None:
        """
        Build scikit-learn pipeline with preprocessing and classification.
        
        Args:
            **classifier_params: Additional parameters for the RandomForestClassifier
        """
        steps = []
        
        if self.use_scaler:
            steps.append(('scaler', StandardScaler()))
        
        # Default classifier parameters
        default_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': self.random_state,
            'class_weight': 'balanced'
        }
        
        # Update with user-provided parameters
        default_params.update(classifier_params)
            
        steps.append(('classifier', RandomForestClassifier(**default_params)))
        
        self.pipeline = Pipeline(steps)
        
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, 
                          feature_names: Optional[List[str]] = None,
                          **classifier_params) -> Dict[str, Any]:
        """
        Train classifier with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
            **classifier_params: Additional parameters for the classifier
            
        Returns:
            Dictionary containing cross-validation results
        """
        self.feature_names = feature_names
        self.build_pipeline(**classifier_params)
        
        # Cross-validation with stratified folds
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='accuracy')
        
        # Fit on full dataset
        self.pipeline.fit(X, y)
        
        return {
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_scores': cv_scores
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.pipeline.predict_proba(X)
        
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from trained classifier.
        
        Returns:
            DataFrame with features and their importance scores, or None if not trained
        """
        if self.pipeline is None:
            return None
            
        classifier = self.pipeline.named_steps['classifier']
        importances = classifier.feature_importances_
        
        if self.feature_names is not None:
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        else:
            return pd.DataFrame({
                'feature_index': range(len(importances)),
                'importance': importances
            }).sort_values('importance', ascending=False)
    
    def get_pipeline(self) -> Optional[Pipeline]:
        """
        Get the trained pipeline.
        
        Returns:
            Trained sklearn Pipeline or None if not trained
        """
        return self.pipeline
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.pipeline is None:
            raise ValueError("No trained model to save")
        
        import joblib
        joblib.dump({
            'pipeline': self.pipeline,
            'feature_names': self.feature_names,
            'use_scaler': self.use_scaler,
            'cv_folds': self.cv_folds,
            'random_state': self.random_state
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'OrganoidClassifier':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded OrganoidClassifier instance
        """
        import joblib
        data = joblib.load(filepath)
        
        classifier = cls(
            use_scaler=data['use_scaler'],
            cv_folds=data['cv_folds'],
            random_state=data['random_state']
        )
        
        classifier.pipeline = data['pipeline']
        classifier.feature_names = data['feature_names']
        
        return classifier