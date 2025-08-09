"""
Main analysis pipeline for organoid intelligence analysis.

This module provides the main pipeline that coordinates all components
for comprehensive organoid analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import warnings

from ..core.simulator import OrganoidRecordingSimulator
from ..core.feature_extractor import EnhancedFeatureExtractor
from ..core.classifier import OrganoidClassifier
from ..utils.visualization import OrganoidVisualizer
from ..utils.data_processing import DataProcessor

warnings.filterwarnings("ignore")


class OrganoidAnalysisPipeline:
    """Main analysis pipeline for organoid intelligence analysis."""
    
    def __init__(self, fs: int = 1000, random_seed: int = 42):
        """
        Initialize the analysis pipeline.
        
        Args:
            fs: Sampling frequency in Hz
            random_seed: Random seed for reproducibility
        """
        self.fs = fs
        self.random_seed = random_seed
        
        # Initialize components
        self.simulator = OrganoidRecordingSimulator(fs=fs)
        self.feature_extractor = EnhancedFeatureExtractor(fs=fs)
        self.classifier = OrganoidClassifier(random_state=random_seed)
        self.visualizer = OrganoidVisualizer()
        self.data_processor = DataProcessor()
        
        # Storage for results
        self.dataset = None
        self.feature_matrix = None
        self.labels = None
        self.feature_names = None
        self.results = {}
        
    def generate_dataset(self, n_samples_per_condition: int = 150, 
                        save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate synthetic dataset for analysis.
        
        Args:
            n_samples_per_condition: Number of samples per condition
            save_path: Optional path to save the dataset
            
        Returns:
            Generated dataset DataFrame
        """
        print("=" * 50)
        print("GENERATING SYNTHETIC DATASET")
        print("=" * 50)
        
        self.dataset, self.feature_matrix, self.labels = self.data_processor.generate_synthetic_dataset(
            self.simulator, self.feature_extractor, n_samples_per_condition, self.random_seed
        )
        
        # Store feature names
        self.feature_names = [col for col in self.dataset.columns if col not in ['label', 'condition']]
        
        if save_path:
            self.data_processor.save_dataset(self.dataset, save_path)
            print(f"Dataset saved to: {save_path}")
        
        return self.dataset
    
    def train_classifier(self, **classifier_params) -> Dict[str, Any]:
        """
        Train the organoid classifier.
        
        Args:
            **classifier_params: Additional parameters for the classifier
            
        Returns:
            Training results dictionary
        """
        if self.feature_matrix is None or self.labels is None:
            raise ValueError("Dataset must be generated first")
        
        print("=" * 50)
        print("TRAINING CLASSIFIER")
        print("=" * 50)
        
        results = self.classifier.train_and_evaluate(
            self.feature_matrix, self.labels, self.feature_names, **classifier_params
        )
        
        print(f"Cross-validation accuracy: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
        print(f"Individual fold scores: {[f'{score:.3f}' for score in results['cv_scores']]}")
        
        self.results['classification'] = results
        return results
    
    def analyze_features(self) -> Dict[str, Any]:
        """
        Perform comprehensive feature analysis.
        
        Returns:
            Feature analysis results
        """
        if self.dataset is None:
            raise ValueError("Dataset must be generated first")
        
        print("=" * 50)
        print("ANALYZING FEATURES")
        print("=" * 50)
        
        # Get feature importance from trained classifier
        importance_df = self.classifier.get_feature_importance()
        
        # Create comprehensive feature report
        feature_report = self.data_processor.create_feature_report(self.dataset, self.feature_names)
        
        # Calculate summary statistics by condition
        summary_stats = self.data_processor.calculate_summary_statistics(
            self.dataset, self.feature_names[:10], 'condition'
        )
        
        analysis_results = {
            'feature_importance': importance_df,
            'feature_report': feature_report,
            'summary_statistics': summary_stats
        }
        
        self.results['feature_analysis'] = analysis_results
        
        print(f"Top 5 most important features:")
        if importance_df is not None:
            for i, row in importance_df.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return analysis_results
    
    def create_visualizations(self, save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create comprehensive visualizations.
        
        Args:
            save_dir: Optional directory to save figures
            
        Returns:
            Dictionary of generated figures
        """
        if self.dataset is None or self.results.get('classification') is None:
            raise ValueError("Dataset and classifier training must be completed first")
        
        print("=" * 50)
        print("CREATING VISUALIZATIONS")
        print("=" * 50)
        
        figures = {}
        
        # Generate sample recordings for visualization
        sample_healthy, t_h = self.simulator.simulate_recording(0, n_channels=16, duration_s=2.0)
        sample_diseased, t_d = self.simulator.simulate_recording(1, n_channels=16, duration_s=2.0)
        
        # 1. Sample recordings comparison
        fig1 = self.visualizer.plot_sample_recordings(
            sample_healthy, sample_diseased, (t_h, t_d)
        )
        figures['sample_recordings'] = fig1
        
        # 2. Power spectral density comparison
        fig2 = self.visualizer.plot_power_spectral_density(
            sample_healthy, sample_diseased, self.fs
        )
        figures['power_spectral_density'] = fig2
        
        # 3. Feature importance
        importance_df = self.results['feature_analysis']['feature_importance']
        fig3 = self.visualizer.plot_feature_importance(importance_df)
        figures['feature_importance'] = fig3
        
        # 4. Cross-validation scores
        cv_scores = self.results['classification']['cv_scores']
        cv_mean = self.results['classification']['cv_mean']
        fig4 = self.visualizer.plot_cv_scores(cv_scores, cv_mean)
        figures['cv_scores'] = fig4
        
        # 5. Feature distributions
        top_features = importance_df.head(6)['feature'].tolist()
        fig5 = self.visualizer.plot_feature_distributions(self.dataset, top_features)
        figures['feature_distributions'] = fig5
        
        # 6. Correlation matrix
        fig6 = self.visualizer.plot_correlation_matrix(self.dataset, self.feature_names[:15])
        figures['correlation_matrix'] = fig6
        
        # 7. Main dashboard
        fig7 = self.visualizer.create_analysis_dashboard(
            sample_healthy, sample_diseased, (t_h, t_d), 
            importance_df, cv_scores, cv_mean, self.fs
        )
        figures['dashboard'] = fig7
        
        # Save figures if directory provided
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            for name, fig in figures.items():
                fig.savefig(f"{save_dir}/{name}.png", dpi=300, bbox_inches='tight')
                fig.savefig(f"{save_dir}/{name}.pdf", bbox_inches='tight')
            
            print(f"Figures saved to: {save_dir}")
        
        return figures
    
    def run_complete_analysis(self, n_samples_per_condition: int = 150,
                             save_dataset_path: Optional[str] = None,
                             save_figures_dir: Optional[str] = None,
                             save_model_path: Optional[str] = None,
                             **classifier_params) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.
        
        Args:
            n_samples_per_condition: Number of samples per condition
            save_dataset_path: Path to save dataset
            save_figures_dir: Directory to save figures
            save_model_path: Path to save trained model
            **classifier_params: Additional classifier parameters
            
        Returns:
            Complete analysis results
        """
        print("ENHANCED ORGANOID INTELLIGENCE ANALYSIS PIPELINE")
        print("=" * 50)
        
        # 1. Generate dataset
        self.generate_dataset(n_samples_per_condition, save_dataset_path)
        
        # 2. Train classifier
        self.train_classifier(**classifier_params)
        
        # 3. Analyze features
        self.analyze_features()
        
        # 4. Create visualizations
        figures = self.create_visualizations(save_figures_dir)
        
        # 5. Save model if requested
        if save_model_path:
            self.classifier.save_model(save_model_path)
            print(f"Model saved to: {save_model_path}")
        
        # 6. Display results summary
        self._display_results_summary()
        
        # Compile complete results
        complete_results = {
            'dataset': self.dataset,
            'classification_results': self.results['classification'],
            'feature_analysis': self.results['feature_analysis'],
            'figures': figures,
            'feature_names': self.feature_names
        }
        
        return complete_results
    
    def _display_results_summary(self) -> None:
        """Display comprehensive results summary."""
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE - SUMMARY")
        print("=" * 50)
        
        # Dataset summary
        print(f"Dataset: {len(self.dataset)} samples, {len(self.feature_names)} features")
        healthy_count = len(self.dataset[self.dataset['condition'] == 'Healthy'])
        diseased_count = len(self.dataset[self.dataset['condition'] == 'Diseased'])
        print(f"Healthy samples: {healthy_count}, Diseased samples: {diseased_count}")
        
        # Classification performance
        cv_results = self.results['classification']
        print(f"\nClassification Performance:")
        print(f"  Cross-validation accuracy: {cv_results['cv_mean']:.3f} ± {cv_results['cv_std']:.3f}")
        print(f"  Best fold score: {max(cv_results['cv_scores']):.3f}")
        print(f"  Worst fold score: {min(cv_results['cv_scores']):.3f}")
        
        # Top features
        importance_df = self.results['feature_analysis']['feature_importance']
        print(f"\nTop 10 Most Discriminative Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")
        
        # Feature quality assessment
        feature_report = self.results['feature_analysis']['feature_report']
        zero_var_features = len(feature_report['zero_variance_features'])
        high_corr_pairs = len(feature_report['high_correlations'])
        
        print(f"\nFeature Quality Assessment:")
        print(f"  Zero variance features: {zero_var_features}")
        print(f"  Highly correlated pairs (>0.9): {high_corr_pairs}")
        
        print(f"\nAnalysis pipeline completed successfully!")
    
    def predict_new_recording(self, recording: np.ndarray) -> Dict[str, Any]:
        """
        Predict condition for a new recording.
        
        Args:
            recording: New recording data (channels x time)
            
        Returns:
            Prediction results
        """
        if self.classifier.get_pipeline() is None:
            raise ValueError("Classifier must be trained first")
        
        # Extract features
        features = self.feature_extractor.extract_features(recording)
        
        # Convert to feature vector in correct order
        feature_vector = np.array([features.get(name, 0) for name in self.feature_names]).reshape(1, -1)
        
        # Handle NaN values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Make prediction
        prediction = self.classifier.predict(feature_vector)[0]
        probabilities = self.classifier.predict_proba(feature_vector)[0]
        
        condition_names = ['Healthy', 'Diseased']
        
        return {
            'prediction': condition_names[prediction],
            'confidence': max(probabilities),
            'probabilities': {
                'healthy': probabilities[0],
                'diseased': probabilities[1]
            },
            'features': features
        }
    
    def load_pretrained_model(self, model_path: str) -> None:
        """
        Load a pretrained model.
        
        Args:
            model_path: Path to saved model
        """
        self.classifier = OrganoidClassifier.load_model(model_path)
        print(f"Model loaded from: {model_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Model information dictionary
        """
        if self.classifier.get_pipeline() is None:
            return {"status": "No model trained"}
        
        pipeline = self.classifier.get_pipeline()
        classifier_step = pipeline.named_steps['classifier']
        
        return {
            "status": "Model trained",
            "classifier_type": type(classifier_step).__name__,
            "n_features": len(self.feature_names) if self.feature_names else "Unknown",
            "feature_names": self.feature_names,
            "uses_scaler": self.classifier.use_scaler,
            "cv_folds": self.classifier.cv_folds,
            "random_state": self.classifier.random_state
        }


def main():
    """Main function to run the complete analysis pipeline."""
    # Initialize pipeline
    pipeline = OrganoidAnalysisPipeline(fs=1000, random_seed=42)
    
    # Run complete analysis
    results = pipeline.run_complete_analysis(
        n_samples_per_condition=150,
        save_dataset_path="data/organoid_dataset.csv",
        save_figures_dir="results/figures",
        save_model_path="results/organoid_classifier.pkl"
    )
    
    return pipeline, results


if __name__ == "__main__":
    pipeline, results = main()