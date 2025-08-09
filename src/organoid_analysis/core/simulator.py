"""
Organoid recording simulator module.

This module provides realistic neural dynamics simulation for organoid recordings.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class OrganoidRecordingSimulator:
    """Enhanced organoid recording simulator with realistic neural dynamics."""
    
    def __init__(self, fs: int = 1000):
        """
        Initialize the organoid recording simulator.
        
        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        self.spike_templates = self._generate_spike_templates()
        
    def _generate_spike_templates(self) -> Dict[str, np.ndarray]:
        """Generate realistic spike templates with different shapes."""
        templates = {}
        t_spike = np.arange(-0.002, 0.003, 1/self.fs)  # 5ms window
        
        # Type 1: Fast spike (pyramidal-like)
        templates['fast'] = -np.exp(-0.5 * (t_spike / 0.0005)**2) * np.exp(t_spike / 0.001)
        
        # Type 2: Slow spike (interneuron-like)
        templates['slow'] = -np.exp(-0.5 * (t_spike / 0.001)**2) * (1 + 0.3 * np.sin(2*np.pi*200*t_spike))
        
        return templates
        
    def _generate_burst_train(self, duration_s: float, burst_rate: float = 0.5, 
                             spikes_per_burst: int = 5) -> np.ndarray:
        """
        Generate burst spike patterns common in organoids.
        
        Args:
            duration_s: Duration of recording in seconds
            burst_rate: Rate of burst occurrence (bursts/second)
            spikes_per_burst: Average number of spikes per burst
            
        Returns:
            Binary spike train array
        """
        n_time = int(duration_s * self.fs)
        spike_train = np.zeros(n_time)
        
        # Generate burst onset times
        n_bursts = np.random.poisson(burst_rate * duration_s)
        if n_bursts == 0:
            return spike_train
            
        burst_times = np.random.choice(n_time, size=min(n_bursts, n_time//100), replace=False)
        
        for burst_start in burst_times:
            n_spikes = np.random.poisson(spikes_per_burst)
            # Spikes within burst follow exponential ISI distribution
            isis = np.random.exponential(0.01 * self.fs, n_spikes)  # 10ms mean ISI
            spike_times = burst_start + np.cumsum(isis).astype(int)
            valid_spikes = spike_times[spike_times < n_time]
            spike_train[valid_spikes] = 1
            
        return spike_train
        
    def _add_network_oscillations(self, t: np.ndarray, condition: int, 
                                 base_freq_range: Tuple[float, float]) -> np.ndarray:
        """
        Add realistic network oscillations with cross-frequency coupling.
        
        Args:
            t: Time array
            condition: 0 for healthy, 1 for diseased/altered
            base_freq_range: Base frequency range (not used in current implementation)
            
        Returns:
            Oscillatory signal component
        """
        sig = np.zeros_like(t)
        
        if condition == 0:  # Healthy condition
            # Strong gamma oscillations with nested theta
            theta_freq = np.random.uniform(4, 8)
            gamma_freq = np.random.uniform(30, 80)
            
            theta_amp = np.random.uniform(0.3, 0.8)
            gamma_amp = np.random.uniform(0.2, 0.5)
            
            # Phase-amplitude coupling
            theta_phase = 2 * np.pi * theta_freq * t
            gamma_modulation = 1 + 0.3 * np.cos(theta_phase)
            
            sig += theta_amp * np.sin(theta_phase + np.random.uniform(0, 2*np.pi))
            sig += gamma_amp * gamma_modulation * np.sin(2*np.pi*gamma_freq*t + np.random.uniform(0, 2*np.pi))
            
        else:  # Diseased/altered condition
            # Disrupted oscillations, more low-frequency activity
            delta_freq = np.random.uniform(1, 4)
            beta_freq = np.random.uniform(12, 25)
            
            delta_amp = np.random.uniform(0.5, 1.0)  # Elevated delta
            beta_amp = np.random.uniform(0.1, 0.3)   # Reduced high-frequency
            
            sig += delta_amp * np.sin(2*np.pi*delta_freq*t + np.random.uniform(0, 2*np.pi))
            sig += beta_amp * np.sin(2*np.pi*beta_freq*t + np.random.uniform(0, 2*np.pi))
            
        return sig
        
    def simulate_recording(self, condition: int, n_channels: int = 16, 
                          duration_s: float = 2.0, noise_level: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate enhanced synthetic organoid recording.
        
        Args:
            condition: 0 for healthy, 1 for diseased/altered
            n_channels: Number of recording channels
            duration_s: Duration of recording in seconds
            noise_level: Level of background noise
            
        Returns:
            Tuple of (recording_data, time_array)
        """
        t = np.arange(0, duration_s, 1/self.fs)
        n_time = len(t)
        data = np.zeros((n_channels, n_time))
        
        # Spatial correlation matrix for realistic channel relationships
        channel_positions = np.random.rand(n_channels, 2) * 10  # 10mm organoid
        distances = np.sqrt(np.sum((channel_positions[:, None, :] - channel_positions[None, :, :])**2, axis=2))
        spatial_correlation = np.exp(-distances / 2.0)  # 2mm correlation length
        
        for ch in range(n_channels):
            # Base oscillations
            sig = self._add_network_oscillations(t, condition, base_freq_range=(1, 100))
            
            # Add burst activity
            if condition == 0:
                burst_rate = np.random.uniform(2, 5)  # Healthy: more bursts
                spikes_per_burst = np.random.poisson(8) + 3
            else:
                burst_rate = np.random.uniform(0.5, 2)  # Diseased: fewer bursts
                spikes_per_burst = np.random.poisson(4) + 1
                
            burst_train = self._generate_burst_train(duration_s, burst_rate, spikes_per_burst)
            
            # Add spikes with realistic templates
            template_type = np.random.choice(['fast', 'slow'], p=[0.7, 0.3])
            template = self.spike_templates[template_type]
            
            spike_indices = np.where(burst_train)[0]
            for spike_idx in spike_indices:
                start_idx = max(0, spike_idx - len(template)//2)
                end_idx = min(n_time, start_idx + len(template))
                template_section = template[:end_idx-start_idx]
                
                amplitude = np.random.uniform(2, 5) if condition == 0 else np.random.uniform(1, 3)
                sig[start_idx:end_idx] += amplitude * template_section
                
            # Add correlated noise based on spatial distance
            noise = np.random.randn(n_time) * noise_level
            for other_ch in range(n_channels):
                if other_ch != ch:
                    correlation = spatial_correlation[ch, other_ch]
                    if correlation > 0.3:  # Only correlate nearby channels
                        shared_noise = np.random.randn(n_time) * noise_level * correlation
                        sig += shared_noise * 0.1
                        
            # Add 1/f background noise
            freqs = np.fft.fftfreq(n_time, 1/self.fs)
            freqs[0] = 1  # Avoid division by zero
            pink_noise_fft = np.random.randn(n_time) / np.sqrt(np.abs(freqs))
            pink_noise = np.real(np.fft.ifft(pink_noise_fft)) * noise_level * 0.5
            sig += pink_noise
            
            data[ch] = sig
            
        return data, t