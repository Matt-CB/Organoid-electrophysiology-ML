"""
Feature extraction module for organoid recordings.

This module provides comprehensive feature extraction capabilities for analyzing
neural signals from organoid recordings.
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional


class EnhancedFeatureExtractor:
    """Advanced feature extraction for organoid recordings."""
    
    def __init__(self, fs: int = 1000):
        """
        Initialize the feature extractor.
        
        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        self.bands = {
            "delta": (0.5, 4),
            "theta": (4, 8), 
            "alpha": (8, 12),
            "beta": (12, 30),
            "low_gamma": (30, 50),
            "high_gamma": (50, 100)
        }
        
    def _bandpower_welch(self, data: np.ndarray, band: Tuple[float, float], 
                        nperseg: Optional[int] = None) -> float:
        """
        Compute bandpower using Welch's method with improved parameters.
        
        Args:
            data: Input signal
            band: Frequency band (fmin, fmax) in Hz
            nperseg: Length of each segment for Welch's method
            
        Returns:
            Power in the specified frequency band
        """
        if nperseg is None:
            nperseg = min(self.fs * 2, len(data) // 4)
            
        freqs, psd = signal.welch(data, fs=self.fs, nperseg=nperseg, 
                                 overlap=nperseg//2, window='hann')
        
        fmin, fmax = band
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        
        if not np.any(idx):
            return 0.0
            
        return np.trapz(psd[idx], freqs[idx])
        
    def _spectral_entropy(self, data: np.ndarray, bands: Optional[List[Tuple[float, float]]] = None) -> float:
        """
        Compute spectral entropy as a measure of signal complexity.
        
        Args:
            data: Input signal
            bands: List of frequency bands, defaults to self.bands
            
        Returns:
            Spectral entropy value
        """
        if bands is None:
            bands = list(self.bands.values())
            
        powers = [self._bandpower_welch(data, band) for band in bands]
        powers = np.array(powers)
        powers = powers / np.sum(powers)  # Normalize
        powers = powers[powers > 0]  # Remove zeros for entropy calculation
        
        return entropy(powers, base=2)
        
    def _detect_bursts(self, data: np.ndarray, min_burst_duration: float = 0.05, 
                      min_iei: float = 0.1) -> Tuple[float, float, float]:
        """
        Detect burst events in the signal.
        
        Args:
            data: Input signal
            min_burst_duration: Minimum duration for valid burst (seconds)
            min_iei: Minimum inter-event interval to separate bursts (seconds)
            
        Returns:
            Tuple of (burst_rate, mean_burst_duration, mean_spikes_per_burst)
        """
        # Threshold-based spike detection
        threshold = np.mean(data) + 3 * np.std(data)
        spikes = data > threshold
        
        # Find spike onset times
        spike_onsets = np.where(np.diff(spikes.astype(int)) == 1)[0]
        
        if len(spike_onsets) < 2:
            return 0, 0, 0  # No bursts possible
            
        # Group spikes into bursts
        isis = np.diff(spike_onsets) / self.fs
        burst_boundaries = np.where(isis > min_iei)[0] + 1
        burst_starts = np.concatenate(([0], burst_boundaries))
        burst_ends = np.concatenate((burst_boundaries, [len(spike_onsets)]))
        
        # Filter bursts by duration and spike count
        valid_bursts = []
        for start, end in zip(burst_starts, burst_ends):
            burst_spikes = spike_onsets[start:end]
            if len(burst_spikes) >= 3:  # At least 3 spikes
                burst_duration = (burst_spikes[-1] - burst_spikes[0]) / self.fs
                if burst_duration >= min_burst_duration:
                    valid_bursts.append((start, end, burst_duration, len(burst_spikes)))
                    
        if not valid_bursts:
            return 0, 0, 0
            
        burst_rate = len(valid_bursts) / (len(data) / self.fs)  # bursts/sec
        mean_burst_duration = np.mean([b[2] for b in valid_bursts])
        mean_spikes_per_burst = np.mean([b[3] for b in valid_bursts])
        
        return burst_rate, mean_burst_duration, mean_spikes_per_burst
        
    def _phase_amplitude_coupling(self, data: np.ndarray, phase_band: Tuple[float, float] = (4, 8), 
                                 amp_band: Tuple[float, float] = (30, 80)) -> float:
        """
        Compute phase-amplitude coupling between frequency bands.
        
        Args:
            data: Input signal
            phase_band: Frequency band for phase extraction
            amp_band: Frequency band for amplitude extraction
            
        Returns:
            Modulation index (PAC strength)
        """
        # Filter signals
        phase_filt = signal.butter(4, phase_band, btype='band', fs=self.fs, output='sos')
        amp_filt = signal.butter(4, amp_band, btype='band', fs=self.fs, output='sos')
        
        phase_sig = signal.sosfiltfilt(phase_filt, data)
        amp_sig = signal.sosfiltfilt(amp_filt, data)
        
        # Extract phase and amplitude
        analytic_phase = signal.hilbert(phase_sig)
        analytic_amp = signal.hilbert(amp_sig)
        
        phase = np.angle(analytic_phase)
        amplitude = np.abs(analytic_amp)
        
        # Compute modulation index
        n_bins = 18
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        mean_amp_per_bin = np.zeros(n_bins)
        
        for i in range(n_bins):
            bin_mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
            if np.any(bin_mask):
                mean_amp_per_bin[i] = np.mean(amplitude[bin_mask])
                
        # Normalize and compute KL divergence
        p_observed = mean_amp_per_bin / np.sum(mean_amp_per_bin)
        p_uniform = np.ones(n_bins) / n_bins
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p_observed = p_observed + epsilon
        p_observed = p_observed / np.sum(p_observed)
        
        modulation_index = entropy(p_uniform, p_observed, base=2)
        return modulation_index
        
    def extract_features(self, recording: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive features from organoid recording.
        
        Args:
            recording: Multi-channel recording data (channels x time)
            
        Returns:
            Dictionary of extracted features
        """
        n_channels, n_time = recording.shape
        features = {}
        
        # 1. Spectral power features
        for band_name, band in self.bands.items():
            powers = [self._bandpower_welch(recording[ch], band) for ch in range(n_channels)]
            features[f"{band_name}_power_mean"] = np.mean(powers)
            features[f"{band_name}_power_std"] = np.std(powers)
            features[f"{band_name}_power_cv"] = np.std(powers) / (np.mean(powers) + 1e-10)
            
        # 2. Spectral complexity
        entropies = [self._spectral_entropy(recording[ch]) for ch in range(n_channels)]
        features["spectral_entropy_mean"] = np.mean(entropies)
        features["spectral_entropy_std"] = np.std(entropies)
        
        # 3. Burst analysis
        burst_rates = []
        burst_durations = []
        spikes_per_burst = []
        
        for ch in range(n_channels):
            br, bd, spb = self._detect_bursts(recording[ch])
            burst_rates.append(br)
            if bd > 0:  # Valid burst detected
                burst_durations.append(bd)
                spikes_per_burst.append(spb)
                
        features["burst_rate_mean"] = np.mean(burst_rates)
        features["burst_rate_std"] = np.std(burst_rates)
        
        if burst_durations:
            features["burst_duration_mean"] = np.mean(burst_durations)
            features["spikes_per_burst_mean"] = np.mean(spikes_per_burst)
        else:
            features["burst_duration_mean"] = 0
            features["spikes_per_burst_mean"] = 0
            
        # 4. Cross-channel synchronization
        corr_matrix = np.corrcoef(recording)
        upper_triangle = corr_matrix[np.triu_indices(n_channels, k=1)]
        
        features["sync_mean"] = np.nanmean(upper_triangle)
        features["sync_std"] = np.nanstd(upper_triangle)
        features["sync_max"] = np.nanmax(upper_triangle)
        
        # Global synchronization index
        global_signal = np.mean(recording, axis=0)
        sync_with_global = [np.corrcoef(recording[ch], global_signal)[0, 1] 
                           for ch in range(n_channels)]
        features["global_sync_mean"] = np.nanmean(sync_with_global)
        
        # 5. Phase-amplitude coupling
        pac_values = []
        for ch in range(n_channels):
            try:
                pac = self._phase_amplitude_coupling(recording[ch])
                if not np.isnan(pac):
                    pac_values.append(pac)
            except Exception:
                continue  # Skip problematic channels
                
        features["pac_mean"] = np.mean(pac_values) if pac_values else 0
        features["pac_std"] = np.std(pac_values) if len(pac_values) > 1 else 0
        
        # 6. Signal quality metrics
        snr_values = []
        for ch in range(n_channels):
            signal_power = np.var(recording[ch])
            # Estimate noise from high-frequency content
            high_freq_filt = signal.butter(4, [80, 200], btype='band', fs=self.fs, output='sos')
            try:
                noise_sig = signal.sosfiltfilt(high_freq_filt, recording[ch])
                noise_power = np.var(noise_sig)
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                snr_values.append(snr)
            except Exception:
                snr_values.append(0)
                
        features["snr_mean"] = np.mean(snr_values)
        features["snr_std"] = np.std(snr_values)
        
        return features