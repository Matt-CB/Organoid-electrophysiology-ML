#!/usr/bin/env python3
"""
Basic usage example for the organoid intelligence analysis package.

This example demonstrates how to use the main components of the package
for a simple analysis workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
from organoid_analysis import (
    OrganoidAnalysisPipeline,
    OrganoidRecordingSimulator,
    EnhancedFeatureExtractor,
    OrganoidClassifier
)


def basic_pipeline_example():
    """Example using the complete analysis pipeline."""
    print("=== Basic Pipeline Example ===")
    
    # Initialize pipeline
    pipeline = OrganoidAnalysisPipeline(fs=1000, random_seed=42)
    
    # Run complete analysis with fewer samples for quick demo
    results = pipeline.run_complete_analysis(
        n_samples_per_condition=50,  # Reduced for quick demo
        save_dataset_path="basic_demo_dataset.csv",
        save_figures_dir="basic_demo_figures"
    )
    
    print("Basic pipeline analysis completed!")
    return pipeline, results


def step_by_step_example():
    """Example using individual components step by step."""
    print("\n=== Step-by-Step Example ===")
    
    # 1. Initialize components
    simulator = OrganoidRecordingSimulator(fs=1000)
    feature_extractor = EnhancedFeatureExtractor(fs=1000)
    classifier = OrganoidClassifier()
    
    print("Components initialized")
    
    # 2. Generate some sample recordings
    print("Generating sample recordings...")
    
    # Healthy recording
    healthy_recording, t_healthy = simulator.simulate_recording(
        condition=0, n_channels=16, duration_s=2.0
    )
    
    # Diseased recording  
    diseased_recording, t_diseased = simulator.simulate_recording(
        condition=1, n_channels=16, duration_s=2.0
    )
    
    print(f"Generated recordings: {healthy_recording.shape}, {diseased_recording.shape}")
    
    # 3. Extract features
    print("Extracting features...")
    
    healthy_features = feature_extractor.extract_features(healthy_recording)
    diseased_features = feature_extractor.extract_features(diseased_recording)
    
    print(f"Extracted {len(healthy_features)} features per recording")
    print(f"Sample features: {list(healthy_features.keys())[:5]}")
    
    # 4. Visualize sample recordings
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(t_healthy, healthy_recording[0], label='Healthy', alpha=0.8)
    plt.plot(t_diseased, diseased_recording[0], label='Diseased', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV)')
    plt.title('Sample LFP Traces (Channel 0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Compare some key features
    plt.subplot(1, 2, 2)
    
    # Select a few key features to compare
    key_features = ['theta_power_mean', 'gamma_power_mean', 'burst_rate_mean', 'sync_mean']
    
    healthy_vals = [healthy_features.get(f, 0) for f in key_features]
    diseased_vals = [diseased_features.get(f, 0) for f in key_features]
    
    x = np.arange(len(key_features))
    width = 0.35
    
    plt.bar(x - width/2, healthy_vals, width, label='Healthy', alpha=0.8)
    plt.bar(x + width/2, diseased_vals, width, label='Diseased', alpha=0.8)
    
    plt.xlabel('Features')
    plt.ylabel('Feature Value')
    plt.title('Feature Comparison')
    plt.xticks(x, [f.replace('_', '\n') for f in key_features], rotation=0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('step_by_step_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Step-by-step example completed!")
    
    return {
        'simulator': simulator,
        'feature_extractor': feature_extractor,
        'classifier': classifier,
        'healthy_features': healthy_features,
        'diseased_features': diseased_features
    }


def prediction_example():
    """Example of making predictions on new recordings."""
    print("\n=== Prediction Example ===")
    
    # First, we need to train a model (using basic pipeline)
    pipeline = OrganoidAnalysisPipeline(fs=1000, random_seed=42)
    
    # Generate and train on a small dataset
    print("Training model...")
    pipeline.generate_dataset(n_samples_per_condition=30)
    pipeline.train_classifier()
    pipeline.analyze_features()
    
    # Now generate a new recording and predict its condition
    print("Generating new recording for prediction...")
    new_recording, _ = pipeline.simulator.simulate_recording(
        condition=0,  # This is actually healthy, let's see if model detects it
        n_channels=16, 
        duration_s=2.0
    )
    
    # Make prediction
    prediction_result = pipeline.predict_new_recording(new_recording)
    
    print("Prediction Results:")
    print(f"  Predicted condition: {prediction_result['prediction']}")
    print(f"  Confidence: {prediction_result['confidence']:.3f}")
    print(f"  Probabilities:")
    print(f"    Healthy: {prediction_result['probabilities']['healthy']:.3f}")
    print(f"    Diseased: {prediction_result['probabilities']['diseased']:.3f}")
    
    return prediction_result


def main():
    """Run all examples."""
    print("Organoid Intelligence Analysis - Basic Usage Examples")
    print("=" * 60)
    
    # Run examples
    try:
        # Basic pipeline
        pipeline, results = basic_pipeline_example()
        
        # Step by step
        step_results = step_by_step_example()
        
        # Prediction
        pred_results = prediction_example()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("Check the generated files:")
        print("  - basic_demo_dataset.csv")
        print("  - basic_demo_figures/")
        print("  - step_by_step_example.png")
        
        return {
            'pipeline': pipeline,
            'results': results,
            'step_results': step_results,
            'pred_results': pred_results
        }
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()