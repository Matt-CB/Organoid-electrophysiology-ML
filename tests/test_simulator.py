# tests/test_simulator.py
"""Tests for the OrganoidRecordingSimulator class."""

import pytest
import numpy as np
from organoid_analysis.core.simulator import OrganoidRecordingSimulator


class TestOrganoidRecordingSimulator:
    """Test suite for OrganoidRecordingSimulator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.simulator = OrganoidRecordingSimulator(fs=1000)
    
    def test_initialization(self):
        """Test simulator initialization."""
        assert self.simulator.fs == 1000
        assert 'fast' in self.simulator.spike_templates
        assert 'slow' in self.simulator.spike_templates
    
    def test_simulate_recording_shape(self):
        """Test that simulated recordings have correct shape."""
        n_channels = 16
        duration_s = 2.0
        
        recording, t = self.simulator.simulate_recording(
            condition=0, n_channels=n_channels, duration_s=duration_s
        )
        
        expected_n_time = int(duration_s * self.simulator.fs)
        assert recording.shape == (n_channels, expected_n_time)
        assert len(t) == expected_n_time
    
    def test_different_conditions(self):
        """Test that different conditions produce different recordings."""
        healthy, _ = self.simulator.simulate_recording(condition=0, n_channels=8, duration_s=1.0)
        diseased, _ = self.simulator.simulate_recording(condition=1, n_channels=8, duration_s=1.0)
        
        # Recordings should be different
        assert not np.array_equal(healthy, diseased)
    
    def test_burst_train_generation(self):
        """Test burst train generation."""
        burst_train = self.simulator._generate_burst_train(duration_s=1.0, burst_rate=1.0)
        
        assert len(burst_train) == self.simulator.fs
        assert burst_train.dtype == np.float64
        assert np.all((burst_train == 0) | (burst_train == 1))


# tests/test_feature_extractor.py
"""Tests for the EnhancedFeatureExtractor class."""

import pytest
import numpy as np
from organoid_analysis.core.feature_extractor import EnhancedFeatureExtractor


class TestEnhancedFeatureExtractor:
    """Test suite for EnhancedFeatureExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = EnhancedFeatureExtractor(fs=1000)
        
        # Create sample recording
        np.random.seed(42)
        self.n_channels = 8
        self.n_time = 2000  # 2 seconds at 1kHz
        self.recording = np.random.randn(self.n_channels, self.n_time)
    
    def test_initialization(self):
        """Test feature extractor initialization."""
        assert self.extractor.fs == 1000
        assert 'delta' in self.extractor.bands
        assert 'gamma' in self.extractor.bands
    
    def test_extract_features_output(self):
        """Test that feature extraction returns expected structure."""
        features = self.extractor.extract_features(self.recording)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check for expected feature types
        power_features = [k for k in features.keys() if 'power' in k]
        assert len(power_features) > 0
        
        sync_features = [k for k in features.keys() if 'sync' in k]
        assert len(sync_features) > 0
    
    def test_bandpower_calculation(self):
        """Test bandpower calculation."""
        band = (8, 12)  # Alpha band
        power = self.extractor._bandpower_welch(self.recording[0], band)
        
        assert isinstance(power, (int, float))
        assert power >= 0
    
    def test_burst_detection(self):
        """Test burst detection."""
        burst_rate, duration, spikes = self.extractor._detect_bursts(self.recording[0])
        
        assert isinstance(burst_rate, (int, float))
        assert isinstance(duration, (int, float))
        assert isinstance(spikes, (int, float))
        assert burst_rate >= 0
        assert duration >= 0
        assert spikes >= 0


# tests/test_classifier.py
"""Tests for the OrganoidClassifier class."""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from organoid_analysis.core.classifier import OrganoidClassifier


class TestOrganoidClassifier:
    """Test suite for OrganoidClassifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = OrganoidClassifier(random_state=42)
        
        # Create synthetic dataset
        self.X, self.y = make_classification(
            n_samples=100, n_features=20, n_classes=2, random_state=42
        )
        self.feature_names = [f'feature_{i}' for i in range(20)]
    
    def test_initialization(self):
        """Test classifier initialization."""
        assert self.classifier.use_scaler is True
        assert self.classifier.cv_folds == 5
        assert self.classifier.random_state == 42
    
    def test_train_and_evaluate(self):
        """Test training and evaluation."""
        results = self.classifier.train_and_evaluate(
            self.X, self.y, self.feature_names
        )
        
        assert 'cv_mean' in results
        assert 'cv_std' in results
        assert 'cv_scores' in results
        
        assert 0 <= results['cv_mean'] <= 1
        assert len(results['cv_scores']) == self.classifier.cv_folds
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        # First train the model
        self.classifier.train_and_evaluate(self.X, self.y, self.feature_names)
        
        importance_df = self.classifier.get_feature_importance()
        
        assert importance_df is not None
        assert len(importance_df) == len(self.feature_names)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
    
    def test_predictions(self):
        """Test making predictions."""
        # Train model
        self.classifier.train_and_evaluate(self.X, self.y, self.feature_names)
        
        # Make predictions
        predictions = self.classifier.predict(self.X[:10])
        probabilities = self.classifier.predict_proba(self.X[:10])
        
        assert len(predictions) == 10
        assert probabilities.shape == (10, 2)
        assert np.all((predictions == 0) | (predictions == 1))


# tests/test_pipeline.py
"""Tests for the OrganoidAnalysisPipeline class."""

import pytest
import numpy as np
from organoid_analysis.analysis.pipeline import OrganoidAnalysisPipeline


class TestOrganoidAnalysisPipeline:
    """Test suite for OrganoidAnalysisPipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = OrganoidAnalysisPipeline(fs=1000, random_seed=42)
    
    def test_initialization(self):
        """Test pipeline initialization."""
        assert self.pipeline.fs == 1000
        assert self.pipeline.random_seed == 42
        assert self.pipeline.simulator is not None
        assert self.pipeline.feature_extractor is not None
        assert self.pipeline.classifier is not None
    
    def test_generate_dataset(self):
        """Test dataset generation."""
        dataset = self.pipeline.generate_dataset(n_samples_per_condition=10)
        
        assert dataset is not None
        assert len(dataset) == 20  # 10 per condition
        assert 'condition' in dataset.columns
        assert 'label' in dataset.columns
        
        # Check that we have both conditions
        conditions = dataset['condition'].unique()
        assert 'Healthy' in conditions
        assert 'Diseased' in conditions
    
    @pytest.mark.slow
    def test_complete_pipeline(self):
        """Test running the complete pipeline (marked as slow)."""
        results = self.pipeline.run_complete_analysis(
            n_samples_per_condition=20  # Small dataset for testing
        )
        
        assert 'dataset' in results
        assert 'classification_results' in results
        assert 'feature_analysis' in results
        assert 'figures' in results
        
        # Check classification performance
        cv_mean = results['classification_results']['cv_mean']
        assert 0 <= cv_mean <= 1
    
    def test_model_info(self):
        """Test getting model information."""
        # Before training
        info = self.pipeline.get_model_info()
        assert info['status'] == 'No model trained'
        
        # After training
        self.pipeline.generate_dataset(n_samples_per_condition=10)
        self.pipeline.train_classifier()
        
        info = self.pipeline.get_model_info()
        assert info['status'] == 'Model trained'