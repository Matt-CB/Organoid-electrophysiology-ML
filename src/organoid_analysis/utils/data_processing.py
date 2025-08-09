"""
Data processing utilities for organoid analysis.

This module provides data handling, preprocessing, and utility functions
for organoid recording analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")


class DataProcessor:
    """Data processing utilities for organoid recordings."""
    
    @staticmethod
    def generate_synthetic_dataset(simulator, feature_extractor, n_samples_per_condition: int = 150,
                                  random_seed: Optional[int] = 42) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Generate synthetic dataset for training and evaluation.
        
        Args:
            simulator: OrganoidRecordingSimulator instance
            feature_extractor: EnhancedFeatureExtractor instance
            n_samples_per_condition: Number of samples per condition
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (dataframe, feature_matrix, labels)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        X_data = []
        y_data = []
        
        print("Generating synthetic organoid recordings...")
        
        for condition in [0, 1]:
            condition_name = "Healthy" if condition == 0 else "Diseased"
            print(f"Generating {condition_name} recordings...")
            
            for i in range(n_samples_per_condition):
                # Vary recording parameters for robustness
                n_channels = np.random.choice([12, 16, 20])
                duration = np.random.uniform(1.5, 3.0)
                noise_level = np.random.uniform(0.2, 0.5)
                
                recording, t = simulator.simulate_recording(
                    condition=condition,
                    n_channels=n_channels, 
                    duration_s=duration,
                    noise_level=noise_level
                )
                
                features = feature_extractor.extract_features(recording)
                X_data.append(features)
                y_data.append(condition)
        
        # Convert to DataFrame
        df = pd.DataFrame(X_data)
        df['label'] = y_data
        df['condition'] = df['label'].map({0: 'Healthy', 1: 'Diseased'})
        
        # Prepare feature matrix
        feature_columns = [col for col in df.columns if col not in ['label', 'condition']]
        X = df[feature_columns].values
        y = df['label'].values
        
        # Handle any NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        print(f"Dataset generated: {len(df)} samples, {len(feature_columns)} features")
        
        return df, X, y
    
    @staticmethod
    def preprocess_features(X: np.ndarray, method: str = 'robust') -> np.ndarray:
        """
        Preprocess feature matrix.
        
        Args:
            X: Feature matrix
            method: Preprocessing method ('robust', 'standard', 'minmax', 'none')
            
        Returns:
            Preprocessed feature matrix
        """
        if method == 'none':
            return X
        
        X_processed = X.copy()
        
        if method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_processed = scaler.fit_transform(X_processed)
        elif method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            X_processed = scaler.fit_transform(X_processed)
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
        
        return X_processed
    
    @staticmethod
    def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                   random_state: int = 42, stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Proportion of test set
            random_state: Random state for reproducibility
            stratify: Whether to stratify split
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        stratify_param = y if stratify else None
        
        return train_test_split(X, y, test_size=test_size, 
                               random_state=random_state, stratify=stratify_param)
    
    @staticmethod
    def calculate_summary_statistics(df: pd.DataFrame, feature_columns: List[str], 
                                   group_by: str = 'condition') -> pd.DataFrame:
        """
        Calculate summary statistics grouped by condition.
        
        Args:
            df: DataFrame containing data
            feature_columns: List of feature column names
            group_by: Column to group by
            
        Returns:
            DataFrame with summary statistics
        """
        return df.groupby(group_by)[feature_columns].agg(['mean', 'std', 'median', 'min', 'max'])
    
    @staticmethod
    def detect_outliers(X: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> np.ndarray:
        """
        Detect outliers in feature matrix.
        
        Args:
            X: Feature matrix
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean array indicating outliers
        """
        if method == 'iqr':
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (X < lower_bound) | (X > upper_bound)
            return np.any(outliers, axis=1)
        
        elif method == 'zscore':
            z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
            return np.any(z_scores > threshold, axis=1)
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    @staticmethod
    def balance_dataset(X: np.ndarray, y: np.ndarray, method: str = 'undersample') -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance dataset to handle class imbalance.
        
        Args:
            X: Feature matrix
            y: Labels
            method: Balancing method ('undersample', 'oversample', 'smote')
            
        Returns:
            Balanced feature matrix and labels
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        
        if len(set(counts)) == 1:
            # Already balanced
            return X, y
        
        if method == 'undersample':
            min_count = min(counts)
            balanced_indices = []
            
            for class_label in unique_classes:
                class_indices = np.where(y == class_label)[0]
                selected_indices = np.random.choice(class_indices, min_count, replace=False)
                balanced_indices.extend(selected_indices)
            
            balanced_indices = np.array(balanced_indices)
            return X[balanced_indices], y[balanced_indices]
        
        elif method == 'oversample':
            max_count = max(counts)
            balanced_X = []
            balanced_y = []
            
            for class_label in unique_classes:
                class_indices = np.where(y == class_label)[0]
                class_X = X[class_indices]
                
                if len(class_indices) < max_count:
                    # Oversample
                    additional_indices = np.random.choice(len(class_indices), 
                                                        max_count - len(class_indices), 
                                                        replace=True)
                    class_X = np.vstack([class_X, class_X[additional_indices]])
                
                balanced_X.append(class_X)
                balanced_y.extend([class_label] * max_count)
            
            return np.vstack(balanced_X), np.array(balanced_y)
        
        elif method == 'smote':
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                return smote.fit_resample(X, y)
            except ImportError:
                raise ImportError("imbalanced-learn package required for SMOTE. Install with: pip install imbalanced-learn")
        
        else:
            raise ValueError(f"Unknown balancing method: {method}")
    
    @staticmethod
    def create_feature_report(df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """
        Create comprehensive feature analysis report.
        
        Args:
            df: DataFrame containing features
            feature_columns: List of feature column names
            
        Returns:
            Dictionary containing feature analysis results
        """
        report = {}
        
        # Basic statistics
        report['basic_stats'] = df[feature_columns].describe()
        
        # Missing values
        report['missing_values'] = df[feature_columns].isnull().sum()
        
        # Feature correlations
        report['correlations'] = df[feature_columns].corr()
        
        # Highly correlated features (> 0.9)
        corr_matrix = report['correlations']
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.9:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))
        report['high_correlations'] = high_corr_pairs
        
        # Feature variance
        report['feature_variance'] = df[feature_columns].var().sort_values(ascending=False)
        
        # Zero variance features
        report['zero_variance_features'] = report['feature_variance'][report['feature_variance'] == 0].index.tolist()
        
        return report
    
    @staticmethod
    def save_dataset(df: pd.DataFrame, filepath: str, format: str = 'csv') -> None:
        """
        Save dataset to file.
        
        Args:
            df: DataFrame to save
            filepath: Output file path
            format: File format ('csv', 'parquet', 'pickle')
        """
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'parquet':
            df.to_parquet(filepath, index=False)
        elif format == 'pickle':
            df.to_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def load_dataset(filepath: str, format: str = 'auto') -> pd.DataFrame:
        """
        Load dataset from file.
        
        Args:
            filepath: Input file path
            format: File format ('csv', 'parquet', 'pickle', 'auto')
            
        Returns:
            Loaded DataFrame
        """
        if format == 'auto':
            if filepath.endswith('.csv'):
                format = 'csv'
            elif filepath.endswith('.parquet'):
                format = 'parquet'
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                format = 'pickle'
            else:
                raise ValueError("Cannot determine file format automatically")
        
        if format == 'csv':
            return pd.read_csv(filepath)
        elif format == 'parquet':
            return pd.read_parquet(filepath)
        elif format == 'pickle':
            return pd.read_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")