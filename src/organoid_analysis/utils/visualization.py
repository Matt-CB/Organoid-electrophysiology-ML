"""
Visualization utilities for organoid analysis.

This module provides comprehensive visualization functions for analyzing
organoid recordings and classification results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings("ignore")


class OrganoidVisualizer:
    """Comprehensive visualization tools for organoid analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('default')
            
        self.default_figsize = figsize
        
    def plot_sample_recordings(self, healthy_recording: np.ndarray, diseased_recording: np.ndarray,
                              time_arrays: Tuple[np.ndarray, np.ndarray], 
                              channel: int = 0, figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot sample recordings from healthy and diseased conditions.
        
        Args:
            healthy_recording: Healthy condition recording data
            diseased_recording: Diseased condition recording data
            time_arrays: Tuple of time arrays for both recordings
            channel: Channel to plot
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        if figsize is None:
            figsize = self.default_figsize
            
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        t_h, t_d = time_arrays
        
        ax.plot(t_h, healthy_recording[channel], alpha=0.8, label='Healthy', color='blue')
        ax.plot(t_d, diseased_recording[channel], alpha=0.8, label='Diseased', color='red')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title(f'Sample LFP Traces (Channel {channel})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_power_spectral_density(self, healthy_recording: np.ndarray, diseased_recording: np.ndarray,
                                   fs: int = 1000, channel: int = 0, 
                                   figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot power spectral density comparison.
        
        Args:
            healthy_recording: Healthy condition recording data
            diseased_recording: Diseased condition recording data
            fs: Sampling frequency
            channel: Channel to analyze
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        if figsize is None:
            figsize = self.default_figsize
            
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        freqs_h, psd_h = signal.welch(healthy_recording[channel], fs=fs, nperseg=1024)
        freqs_d, psd_d = signal.welch(diseased_recording[channel], fs=fs, nperseg=1024)
        
        ax.semilogy(freqs_h, psd_h, alpha=0.8, label='Healthy', color='blue')
        ax.semilogy(freqs_d, psd_d, alpha=0.8, label='Diseased', color='red')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title('Power Spectral Density Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 15,
                               figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot feature importance from trained classifier.
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to display
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        if figsize is None:
            figsize = self.default_figsize
            
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        top_features = importance_df.head(top_n)
        
        bars = ax.barh(range(len(top_features)), top_features['importance'][::-1])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'][::-1])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.grid(True, alpha=0.3)
        
        # Color bars by importance
        colors = plt.cm.viridis(top_features['importance'][::-1] / top_features['importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        return fig
    
    def plot_cv_scores(self, cv_scores: np.ndarray, cv_mean: float,
                      figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot cross-validation scores.
        
        Args:
            cv_scores: Array of CV scores
            cv_mean: Mean CV score
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        if figsize is None:
            figsize = (8, 6)
            
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        bars = ax.bar(range(1, len(cv_scores) + 1), cv_scores, alpha=0.7, color='skyblue')
        ax.axhline(y=cv_mean, color='red', linestyle='--', alpha=0.7, 
                  label=f'Mean: {cv_mean:.3f}')
        
        ax.set_xlabel('CV Fold')
        ax.set_ylabel('Accuracy')
        ax.set_title('Cross-Validation Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, cv_scores)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_distributions(self, df: pd.DataFrame, features: List[str],
                                  condition_col: str = 'condition',
                                  figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot feature distributions by condition.
        
        Args:
            df: DataFrame containing features and conditions
            features: List of features to plot
            condition_col: Name of condition column
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        if figsize is None:
            figsize = (5 * n_cols, 4 * n_rows)
            
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Create violin plot
            sns.violinplot(data=df, x=condition_col, y=feature, ax=ax)
            ax.set_title(f'{feature} Distribution')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, features: List[str],
                               figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot correlation matrix of features.
        
        Args:
            df: DataFrame containing features
            features: List of features to include
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        if figsize is None:
            figsize = (10, 8)
            
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        corr_matrix = df[features].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, ax=ax, cbar_kws={"shrink": .8})
        
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        return fig
    
    def create_analysis_dashboard(self, healthy_recording: np.ndarray, diseased_recording: np.ndarray,
                                 time_arrays: Tuple[np.ndarray, np.ndarray], importance_df: pd.DataFrame,
                                 cv_scores: np.ndarray, cv_mean: float, fs: int = 1000,
                                 channel: int = 0, figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create comprehensive analysis dashboard.
        
        Args:
            healthy_recording: Healthy condition recording
            diseased_recording: Diseased condition recording
            time_arrays: Time arrays for recordings
            importance_df: Feature importance DataFrame
            cv_scores: Cross-validation scores
            cv_mean: Mean CV score
            fs: Sampling frequency
            channel: Channel to display
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        if figsize is None:
            figsize = (16, 12)
            
        fig = plt.figure(figsize=figsize)
        
        # Sample recordings
        ax1 = plt.subplot(2, 2, 1)
        t_h, t_d = time_arrays
        plt.plot(t_h, healthy_recording[channel], alpha=0.8, label='Healthy', color='blue')
        plt.plot(t_d, diseased_recording[channel], alpha=0.8, label='Diseased', color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (μV)')
        plt.title(f'Sample LFP Traces (Channel {channel})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Feature importance
        ax2 = plt.subplot(2, 2, 2)
        top_features = importance_df.head(15)
        bars = plt.barh(range(len(top_features)), top_features['importance'][::-1])
        plt.yticks(range(len(top_features)), top_features['feature'][::-1])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importances')
        plt.grid(True, alpha=0.3)
        
        # Power spectral density
        ax3 = plt.subplot(2, 2, 3)
        freqs_h, psd_h = signal.welch(healthy_recording[channel], fs=fs, nperseg=1024)
        freqs_d, psd_d = signal.welch(diseased_recording[channel], fs=fs, nperseg=1024)
        plt.semilogy(freqs_h, psd_h, alpha=0.8, label='Healthy', color='blue')
        plt.semilogy(freqs_d, psd_d, alpha=0.8, label='Diseased', color='red')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title('Power Spectral Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        
        # Cross-validation scores
        ax4 = plt.subplot(2, 2, 4)
        bars = plt.bar(range(1, len(cv_scores) + 1), cv_scores, alpha=0.7, color='skyblue')
        plt.axhline(y=cv_mean, color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean: {cv_mean:.3f}')
        plt.xlabel('CV Fold')
        plt.ylabel('Accuracy')
        plt.title('Cross-Validation Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        return fig