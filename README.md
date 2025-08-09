# Organoid Intelligence Analysis

A comprehensive toolkit for analyzing neural recordings from brain organoids, including simulation, feature extraction, classification, and visualization capabilities.

## Overview

This package provides a complete pipeline for analyzing electrophysiological recordings from brain organoids. It includes:

- **Realistic neural signal simulation** with organoid-specific characteristics
- **Advanced feature extraction** including spectral analysis, burst detection, and synchronization metrics
- **Machine learning classification** for condition prediction
- **Comprehensive visualization tools** for data exploration and results presentation

## Features

### Core Components

- **OrganoidRecordingSimulator**: Generate realistic synthetic organoid recordings
- **EnhancedFeatureExtractor**: Extract comprehensive features from neural signals
- **OrganoidClassifier**: Train ML models for condition classification
- **OrganoidVisualizer**: Create publication-ready visualizations
- **DataProcessor**: Handle data preprocessing and management

### Analysis Capabilities

- Multi-channel LFP signal simulation
- Burst activity detection and characterization
- Cross-frequency coupling analysis
- Network synchronization metrics
- Spectral power analysis across frequency bands
- Feature importance ranking
- Cross-validation with performance metrics

## Installation

### From Source
```bash
git clone https://github.com/yourusername/organoid-intelligence-analysis.git
cd organoid-intelligence-analysis
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/yourusername/organoid-intelligence-analysis.git
cd organoid-intelligence-analysis
pip install -e ".[dev,docs,jupyter]"
```

## Quick Start

### Basic Usage

```python
from organoid_analysis import OrganoidAnalysisPipeline

# Initialize the analysis pipeline
pipeline = OrganoidAnalysisPipeline(fs=1000, random_seed=42)

# Run complete analysis
results = pipeline.run_complete_analysis(
    n_samples_per_condition=150,
    save_dataset_path="organoid_dataset.csv",
    save_figures_dir="analysis_figures/",
    save_model_path="trained_model.pkl"
)

# Check results
print(f"Classification accuracy: {results['classification_results']['cv_mean']:.3f}")
```

### Step-by-Step Analysis

```python
from organoid_analysis import (
    OrganoidRecordingSimulator,
    EnhancedFeatureExtractor,
    OrganoidClassifier
)

# 1. Generate synthetic recordings
simulator = OrganoidRecordingSimulator(fs=1000)
healthy_recording, t = simulator.simulate_recording(condition=0, n_channels=16)
diseased_recording, t = simulator.simulate_recording(condition=1, n_channels=16)

# 2. Extract features
feature_extractor = EnhancedFeatureExtractor(fs=1000)
healthy_features = feature_extractor.extract_features(healthy_recording)
diseased_features = feature_extractor.extract_features(diseased_recording)

# 3. Train classifier (with dataset)
classifier = OrganoidClassifier()
# ... (prepare feature matrix X and labels y)
results = classifier.train_and_evaluate(X, y, feature_names)
```

### Making Predictions

```python
# Load a trained model
pipeline = OrganoidAnalysisPipeline()
pipeline.load_pretrained_model("trained_model.pkl")

# Predict on new recording
new_recording, _ = simulator.simulate_recording(condition=0, n_channels=16)
prediction = pipeline.predict_new_recording(new_recording)

print(f"Predicted condition: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']:.3f}")
```

## Documentation

### Package Structure

```
src/organoid_analysis/
├── core/                    # Core analysis components
│   ├── simulator.py         # Neural signal simulation
│   ├── feature_extractor.py # Feature extraction
│   └── classifier.py        # ML classification
├── utils/                   # Utility functions
│   ├── visualization.py     # Plotting and visualization
│   └── data_processing.py   # Data handling utilities
└── analysis/               # High-level analysis
    └── pipeline.py          # Main analysis pipeline
```

### Key Classes

#### OrganoidRecordingSimulator
Generates realistic synthetic organoid recordings with:
- Burst activity patterns
- Network oscillations (theta, gamma, etc.)
- Cross-frequency coupling
- Spatial correlations between channels
- Condition-specific alterations

#### EnhancedFeatureExtractor
Extracts comprehensive features including:
- Spectral power in multiple frequency bands
- Burst detection and characterization
- Cross-channel synchronization
- Phase-amplitude coupling
- Signal quality metrics
- Spectral entropy

#### OrganoidClassifier
Machine learning pipeline featuring:
- Random Forest classification
- Cross-validation evaluation
- Feature importance analysis
- Model persistence
- Robust preprocessing

## Examples

The `examples/` directory contains comprehensive usage examples:

- `basic_usage.py`: Getting started with the package
- `advanced_analysis.py`: Advanced analysis workflows
- Jupyter notebooks for interactive exploration

### Running Examples

```bash
# Run basic usage example
python examples/basic_usage.py

# Or use the command-line interface
organoid-analysis --help
```

## Requirements

### Core Dependencies
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0

### Optional Dependencies
- seaborn >= 0.11.0 (enhanced visualization)
- plotly >= 5.0.0 (interactive plots)
- jupyter >= 1.0.0 (notebook support)

## Development

### Setting Up Development Environment

```bash
git clone https://github.com/yourusername/organoid-intelligence-analysis.git
cd organoid-intelligence-analysis
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=organoid_analysis

# Run specific test modules
pytest tests/test_simulator.py
```

### Code Quality

```bash
# Format code
black src/ tests/ examples/

# Sort imports
isort src/ tests/ examples/

# Lint code
flake8 src/ tests/ examples/
```

### Building Documentation

```bash
pip install -e ".[docs]"
cd docs/
make html
```
## Citation

If you use this package in your research, please cite:

```bibtex
@software{organoid_intelligence_analysis,
  title={Organoid Intelligence Analysis: A Comprehensive Toolkit for Neural Signal Analysis},
  author={Matias Benitez},
  year={2025},
  url={https://github.com/Matt-CB/Organoid-electrophysiology-ML}
}
```

## Contributing

Everyone that one to do contributions is welcome! Please see the contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation for API changes
- Ensure backward compatibility when possible

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support
- **Email**: You can contact to me trough matiasbenitezcarrizo@gmail.com
- **Linkedin**: You can contact to me trough matiasbenitezcarrizo@gmail.com

## Acknowledgments

- Inspired by advances in organoid neuroscience research
- Built on Python scientific computing ecosystem
- Thanks to my friends in neuroscience community for the help with the valuable feedback and information

## Changelog

### Version 1.0.0 (2025-8-8)
- Initial release
- Core simulation and analysis functionality
- Comprehensive feature extraction
- Machine learning classification
- Visualization tools
- Documentation and examples

---

**Note**: This package is designed for research purposes. Always validate results with domain expertise and experimental data.