import os
import uuid
import tempfile
import logging
import numpy as np
from scipy.io import wavfile
from scipy import signal
from flask import current_app
from werkzeug.utils import secure_filename
from pydub import AudioSegment

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def save_audio_file(file):
    """Save the uploaded audio file to a temporary location and convert to WAV if needed"""
    if file and allowed_file(file.filename):
        # Get the file extension
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        
        # Create a unique filename to avoid collisions
        unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        original_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(original_filepath)
        
        # If the file is opus, convert it to wav
        if file_ext.lower() == 'opus':
            try:
                # Create a new wav filename
                wav_filename = f"{uuid.uuid4()}.wav"
                wav_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], wav_filename)
                
                # Use pydub to convert opus to wav without specifying format
                # This makes it more flexible for handling different opus file variants
                sound = AudioSegment.from_file(original_filepath)
                sound.export(wav_filepath, format="wav")
                
                # Remove the original opus file
                os.remove(original_filepath)
                
                return wav_filepath
            except Exception as e:
                logging.error(f"Error converting opus to wav: {str(e)}")
                # Log more details for debugging
                logging.error(f"File path: {original_filepath}, File extension: {file_ext}")
                logging.error(f"Output from ffmpeg/avlib: {e}")
                
                try:
                    # Alternative: Try using FFmpeg directly
                    import subprocess
                    wav_filename = f"{uuid.uuid4()}.wav"
                    wav_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], wav_filename)
                    
                    cmd = ['ffmpeg', '-i', original_filepath, wav_filepath]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if os.path.exists(wav_filepath):
                        os.remove(original_filepath)
                        return wav_filepath
                    else:
                        logging.error(f"FFmpeg conversion failed: {result.stderr}")
                        return None
                except Exception as e2:
                    logging.error(f"Alternative conversion failed: {str(e2)}")
                    return None
        
        return original_filepath
    return None

def extract_embedding(audio_path):
    """Extract robust voice embeddings focused on vocal characteristics"""
    try:
        # Read the WAV file
        sample_rate, audio_data = wavfile.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Ensure consistent data type
        audio_data = audio_data.astype(np.float32)
        
        # Trim silence from beginning and end to focus on voice
        # Simple energy-based trimming
        energy = np.abs(audio_data)
        threshold = np.mean(energy) * 0.1  # 10% of mean energy
        mask = energy > threshold
        
        # Find first and last non-silence
        if np.any(mask):
            start = np.where(mask)[0][0]
            end = np.where(mask)[0][-1]
            # Add some margin
            start = max(0, start - int(0.1 * sample_rate))  # 100ms margin
            end = min(len(audio_data), end + int(0.1 * sample_rate))
            audio_data = audio_data[start:end]
        
        # Normalize to range [-1, 1]
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
            
        # Add deliberate noise to test for uniqueness, for testing only
        # This ensures the feature extraction is sensitive to differences
        # audio_data += np.random.normal(0, 0.001, audio_data.shape)
        
        # Apply stronger pre-emphasis to highlight higher frequencies (formants more than consonants)
        # Higher pre-emphasis coefficient to better capture formant structure
        pre_emphasis = 0.98
        audio_emphasized = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        
        # Create frames for analysis - use larger frames for better frequency resolution
        # This helps capture formant structure that's unique to a speaker regardless of speech content
        frame_size = int(0.035 * sample_rate)  # 35ms (larger for better frequency resolution)
        frame_stride = int(0.015 * sample_rate)  # 15ms (for better temporal coverage)
        
        # Ensure enough data for analysis
        if len(audio_emphasized) < frame_size:
            audio_emphasized = np.pad(audio_emphasized, (0, frame_size - len(audio_emphasized)))
        
        # Calculate number of frames
        num_frames = 1 + int(np.floor((len(audio_emphasized) - frame_size) / frame_stride))
        
        # Apply windowing to frames
        window = np.hamming(frame_size)
        frames = np.zeros((num_frames, frame_size))
        for i in range(num_frames):
            start = i * frame_stride
            end = start + frame_size
            if end <= len(audio_emphasized):
                frames[i] = audio_emphasized[start:end] * window
        
        # Compute FFT for each frame - increased for better frequency resolution
        # Higher resolution helps capture more detailed formant structure (speaker-specific)
        NFFT = 1024  # Doubled for higher frequency resolution
        magnitude_spectra = np.absolute(np.fft.rfft(frames, NFFT))
        
        # Compute power spectra
        power_spectra = ((1.0 / NFFT) * (magnitude_spectra ** 2))
        
        # Extract frequency range most important for speaker identity with focus on formants
        # Formants (especially F1-F3) are critical for speaker identification and are less affected by content
        low_freq = 250  # Lower to catch more of F1 (first formant)
        high_freq = min(4000, sample_rate // 2)  # Higher to better capture F3/F4 formants
        
        # Convert Hz to FFT bin indices
        low_bin = int(low_freq * NFFT / sample_rate)
        high_bin = int(high_freq * NFFT / sample_rate)
        
        # Extract relevant frequency range
        voice_range = power_spectra[:, low_bin:high_bin]
        
        # Apply log to spectra (similar to human perception)
        log_power_spectra = np.log10(voice_range + 1e-10)
        
        # Feature extraction approach 1: Statistical features across time and frequency
        # These capture the distribution and variability of spectral energy
        
        # Frequency-domain features (voice characteristics)
        frame_means = np.mean(log_power_spectra, axis=0)  # Average energy at each frequency
        frame_stds = np.std(log_power_spectra, axis=0)    # Energy variability at each frequency
        
        # Time-domain features (temporal patterns)
        freq_means = np.mean(log_power_spectra, axis=1)   # Average energy in each frame
        freq_stds = np.std(log_power_spectra, axis=1)     # Energy spread in each frame
        
        # Energy distribution (vocal intensity patterns)
        energy_per_frame = np.sum(voice_range, axis=1)
        
        # Spectral flatness - differentiates between tonal (voice) and noise signals
        # Ratio of geometric mean to arithmetic mean
        epsilon = 1e-10
        spectral_flatness = []
        for frame in voice_range:
            if np.sum(frame) > epsilon:
                geo_mean = np.exp(np.mean(np.log(frame + epsilon)))
                arith_mean = np.mean(frame)
                if arith_mean > epsilon:
                    flatness = geo_mean / arith_mean
                    spectral_flatness.append(flatness)
                else:
                    spectral_flatness.append(0)
            else:
                spectral_flatness.append(0)
        
        # Spectral centroid - brightness of sound
        spec_centroid = []
        freq_range = np.linspace(low_freq, high_freq, high_bin - low_bin)
        for frame in voice_range:
            if np.sum(frame) > epsilon:
                centroid = np.sum(freq_range * frame) / np.sum(frame)
                spec_centroid.append(centroid)
            else:
                spec_centroid.append(0)
        
        # Calculate first and second derivatives of features to capture speaking rate/style
        def get_deltas(features, width=3):
            padded = np.pad(features, (width, width), mode='edge')
            deltas = []
            for i in range(width, len(padded) - width):
                delta = 0
                for j in range(1, width + 1):
                    delta += j * (padded[i + j] - padded[i - j])
                deltas.append(delta / (2 * sum([j**2 for j in range(1, width + 1)])))
            return np.array(deltas)
        
        # Delta features (rate of change in spectra)
        delta_means = get_deltas(freq_means)
        delta_energy = get_deltas(energy_per_frame)
        
        # Sample key features at regular intervals to reduce dimensionality
        def sample_feature(feature, num_samples=20):
            # Convert to numpy array for consistent handling
            feature_array = np.array(feature)
            
            if len(feature_array) <= num_samples:
                return np.pad(feature_array, (0, num_samples - len(feature_array)), 'constant')
            
            # Get samples at regular intervals
            indices = np.linspace(0, len(feature_array) - 1, num_samples, dtype=int)
            return feature_array[indices]
        
        # Downsample to fixed dimensions for consistent embedding size
        frame_means_sampled = sample_feature(frame_means, 20)  # Frequency characteristics
        frame_stds_sampled = sample_feature(frame_stds, 15)    # Frequency variability
        freq_means_sampled = sample_feature(freq_means, 20)    # Time characteristics
        freq_stds_sampled = sample_feature(freq_stds, 15)      # Time variability
        energy_sampled = sample_feature(energy_per_frame, 15)  # Energy pattern
        flatness_sampled = sample_feature(spectral_flatness, 10)  # Voice tonality
        centroid_sampled = sample_feature(spec_centroid, 10)   # Voice brightness
        delta_means_sampled = sample_feature(delta_means, 15)  # Speaking rate
        delta_energy_sampled = sample_feature(delta_energy, 10)  # Energy changes
        
        # Add global statistics that are highly indicative of speaker identity
        global_stats = [
            np.mean(spectral_flatness),       # Overall voice tonality
            np.std(spectral_flatness),        # Variation in tonality
            np.mean(spec_centroid),           # Overall voice brightness
            np.std(spec_centroid),            # Variation in brightness
            np.max(energy_per_frame) / (np.mean(energy_per_frame) + epsilon),  # Peak-to-average ratio
            np.percentile(energy_per_frame, 90) / (np.percentile(energy_per_frame, 10) + epsilon),  # Dynamic range
            np.mean(delta_means),             # Average speaking rate
            np.std(delta_means),              # Variation in speaking rate
            np.mean(delta_energy),            # Energy modulation
            np.std(delta_energy)              # Variation in energy
        ]
        
        # Construct the final embedding
        embedding = np.concatenate([
            frame_means_sampled,    # 20 features - Spectral distribution
            frame_stds_sampled,     # 15 features - Spectral variability
            freq_means_sampled,     # 20 features - Temporal distribution
            freq_stds_sampled,      # 15 features - Temporal variability
            energy_sampled,         # 15 features - Energy pattern
            flatness_sampled,       # 10 features - Voice tonality
            centroid_sampled,       # 10 features - Voice brightness
            delta_means_sampled,    # 15 features - Speaking rate
            delta_energy_sampled,   # 10 features - Energy dynamics
            global_stats            # 10 features - Global voice characteristics
        ])
        
        # Apply L2 normalization to the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        # Log success and details
        logging.debug(f"Extracted embedding with shape {embedding.shape} from audio of length {len(audio_data)}")
        
        return embedding
        
    except Exception as e:
        logging.error(f"Error extracting voice embedding: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_similarity(embedding1, embedding2):
    """Calculate voice similarity with specific focus on voice biometric features"""
    try:
        if embedding1 is None or embedding2 is None:
            logging.error("Cannot calculate similarity with None embedding")
            return 0.0
        
        # Check for identical embeddings (self-comparison) before any processing
        if np.array_equal(np.array(embedding1), np.array(embedding2)):
            logging.debug("Identical embeddings detected - returning 1.0 similarity")
            return 1.0  # Perfect match for identical embeddings
            
        # Ensure numpy arrays
        if not isinstance(embedding1, np.ndarray):
            embedding1 = np.array(embedding1)
        if not isinstance(embedding2, np.ndarray):
            embedding2 = np.array(embedding2)
        
        # Logging for debug
        logging.debug(f"Comparing embeddings of shapes {embedding1.shape} and {embedding2.shape}")
        
        # Truncate to same length if needed
        min_length = min(embedding1.shape[0], embedding2.shape[0])
        embedding1 = embedding1[:min_length]
        embedding2 = embedding2[:min_length]
        
        # Simple print of first few values for debugging
        logging.debug(f"Embedding1 first 5 values: {embedding1[:5]}")
        logging.debug(f"Embedding2 first 5 values: {embedding2[:5]}")
            
        # Calculate differences at feature level (useful for debugging)
        abs_diff = np.abs(embedding1 - embedding2)
        max_diff_idx = np.argmax(abs_diff)
        logging.debug(f"Max difference at index {max_diff_idx}: {abs_diff[max_diff_idx]}")
        
        # Feature groups in our embedding (based on how we constructed it)
        # Numbers correspond to indices in concatenated embedding
        feature_groups = {
            'spectral_dist': (0, 20),      # Frequency distribution
            'spectral_var': (20, 35),      # Frequency variability
            'temporal_dist': (35, 55),     # Time distribution
            'temporal_var': (55, 70),      # Time variability
            'energy': (70, 85),            # Energy pattern
            'flatness': (85, 95),          # Voice tonality
            'centroid': (95, 105),         # Voice brightness
            'delta_means': (105, 120),     # Speaking rate
            'delta_energy': (120, 130),    # Energy dynamics
            'global_stats': (130, 140)     # Global voice characteristics
        }
        
        # Weights for different feature groups (biometric importance)
        # Higher weight = more important for voice identity
        # Even stronger focus on content-independent voice characteristics
        # These weights dramatically prioritize vocal tract features and global voice properties
        group_weights = {
            'spectral_dist': 4.0,    # Essential - represents vocal tract shape (most reliable biometric)
            'spectral_var': 0.5,     # Reduced - speaking style can vary with content
            'temporal_dist': 0.3,    # Greatly reduced - heavily affected by speech content
            'temporal_var': 0.2,     # Greatly reduced - heavily affected by speech content
            'energy': 0.3,           # Reduced - affected by recording conditions and content
            'flatness': 2.5,         # Increased - voice tonality is a core biometric
            'centroid': 2.0,         # Increased - voice brightness/timbre is speaker-specific
            'delta_means': 0.2,      # Greatly reduced - extremely dependent on speech content
            'delta_energy': 0.2,     # Greatly reduced - extremely dependent on speech patterns
            'global_stats': 3.5      # Dramatically increased - these are the most reliable biometric features
        }
        
        # Calculate weighted group similarities
        group_similarities = {}
        total_weight = 0
        weighted_sum = 0
        
        for group, (start, end) in feature_groups.items():
            # Get the feature vectors for this group
            e1 = embedding1[start:end]
            e2 = embedding2[start:end]
            
            # Only process if we have valid vectors
            if len(e1) > 0 and len(e2) > 0:
                # Normalize the vectors
                n1 = np.linalg.norm(e1)
                n2 = np.linalg.norm(e2)
                
                if n1 > 1e-10 and n2 > 1e-10:
                    # Calculate multiple distance metrics for more robust comparison
                    
                    # 1. Cosine Similarity - good for capturing overall pattern similarity
                    e1_norm = e1 / n1
                    e2_norm = e2 / n2
                    cosine_sim = np.dot(e1_norm, e2_norm)
                    cosine_sim = max(-1.0, min(1.0, cosine_sim))  # Ensure in range [-1, 1]
                    
                    # 2. Euclidean Distance - good for capturing absolute differences
                    # Scale to [0, 1] similarity space (closer to 1 is more similar)
                    euclidean_dist = np.linalg.norm(e1_norm - e2_norm)
                    euclidean_sim = 1.0 / (1.0 + euclidean_dist)  # Convert to similarity
                    
                    # 3. Peak difference - captures the worst case scenario differences
                    peak_diff = np.max(np.abs(e1_norm - e2_norm))
                    peak_sim = 1.0 - min(1.0, peak_diff)  # Convert to similarity
                    
                    # Ensure all similarity values are valid (not NaN)
                    cosine_sim = 0.0 if np.isnan(cosine_sim) else cosine_sim
                    euclidean_sim = 0.0 if np.isnan(euclidean_sim) else euclidean_sim
                    peak_sim = 0.0 if np.isnan(peak_sim) else peak_sim
                    
                    # Special case for delta_energy which often causes NaN issues
                    if group == 'delta_energy' and (np.any(np.isnan(e1_norm)) or np.any(np.isnan(e2_norm))):
                        logging.debug(f"Setting zero importance for delta_energy due to NaN values")
                        group_weights[group] = 0.0  # Set the weight to zero for this problematic feature
                    
                    # Combine all similarity metrics with weights favoring the most discriminative
                    if group in ['spectral_dist', 'flatness', 'global_stats', 'centroid']:
                        # For critical biometric features, use a weighted average that emphasizes
                        # the worst scoring metric to make it harder to get high similarity
                        group_sim = (
                            cosine_sim * 0.4 +          # Pattern similarity
                            euclidean_sim * 0.3 +       # Overall difference
                            peak_sim * 0.3              # Worst case difference
                        )
                        
                        # For critical voice features, apply much stronger non-linear penalty
                        # This creates a clearer separation between true matches and false matches
                        if group_sim < 0.95:
                            # Exponential penalty function that drops quickly below 0.95
                            exponent = 3.0 if group_sim < 0.85 else 2.0
                            group_sim = group_sim ** exponent
                    else:
                        # For less critical features, use primarily cosine similarity
                        group_sim = (
                            cosine_sim * 0.7 +          # Pattern similarity is most important
                            euclidean_sim * 0.2 +       # Some weight on overall difference
                            peak_sim * 0.1              # Minor weight on worst case
                        )
                        
                        # Still apply a penalty, but less aggressive
                        if group_sim < 0.9:
                            # Apply penalty safely with NaN check
                            if not np.isnan(group_sim) and group_sim > 0:
                                group_sim = group_sim ** 1.5
                            else:
                                group_sim = 0.0  # Set to zero if NaN or negative
                    
                    # Get importance weight for this feature group
                    weight = group_weights[group]
                    
                    # Store the similarity for debugging
                    group_similarities[group] = group_sim
                    
                    # Add to weighted sum
                    weighted_sum += weight * group_sim
                    total_weight += weight
                else:
                    group_similarities[group] = 0
            else:
                group_similarities[group] = 0
        
        # Calculate the final weighted similarity
        if total_weight > 0:
            final_similarity = weighted_sum / total_weight
            # Ensure the similarity is a valid number, not NaN
            if np.isnan(final_similarity):
                logging.error("NaN detected in final similarity calculation!")
                final_similarity = 0.0  # Default to zero similarity if we have a NaN
        else:
            final_similarity = 0.0
            
        # Final state-of-the-art scaling function with extreme separation between speakers
        # 1. Almost completely suppress low similarities for different speakers (→ 0%)
        # 2. Create an extraordinary steep "cliff" transition for perfect separation
        # 3. Highly boost similarity scores for same speaker voice samples (→ 95%+)
        
        # Apply hyperbolic tangent for extreme contrast - creating a true binary response curve
        # This creates an extremely sharp non-linear "cliff" right at the decision boundary
        def hyperbolic_cliff(x, steepness=8.0, midpoint=0.72):
            """
            Creates an extremely sharp transition in the similarity scores
            - Below the midpoint: scores collapse toward zero (different speaker)
            - Above the midpoint: scores skyrocket toward one (same speaker)
            - The transition is near-instantaneous, like a true cliff
            """
            # Center at the midpoint and apply steepness factor
            normalized = (x - midpoint) * steepness
            # Apply hyperbolic tangent for S-curve with near-vertical transition
            enhanced = np.tanh(normalized)
            # Rescale to [0, 1] range
            return (enhanced + 1) / 2.0
            
        # Apply basic scaling first, then the extreme cliff transformation
        if final_similarity < 0.4:
            # Very low similarity - clearly different speakers
            # Apply extreme suppression to push different voices to near-zero
            base_scaled = final_similarity * 0.05  # Even more aggressive reduction (95% reduction)
        elif final_similarity < 0.64:
            # Low-medium similarity - almost certainly different speakers
            # Create an extremely slow rise to maintain clear separation 
            base_scaled = 0.02 + (final_similarity - 0.4) * 0.25  # Very slow linear growth
        elif final_similarity < 0.72:
            # Medium similarity - approaching the decision boundary
            # Create a slightly faster rise, but still suppress values
            base_scaled = 0.08 + (final_similarity - 0.64) * 0.8  # Moderate growth
        elif final_similarity < 0.8:
            # Critical transition zone - the "cliff" where decisions are made
            # Create an extraordinary steep rise through the binary decision boundary
            # This extreme cliff creates maximum possible separation between different/same speakers
            boost_factor = 8.5  # Dramatically increased for near-vertical transition
            base_scaled = 0.15 + (final_similarity - 0.72) * boost_factor
        elif final_similarity < 0.88:
            # High similarity - definitely same speaker
            # Continue steep curve to maximize scores for real matches
            base_scaled = 0.83 + (final_similarity - 0.8) * 2.0
        else:
            # Very high scores (>0.88) - absolutely certain same speaker
            base_scaled = 0.99  # Peg at near-perfect score for extremely high confidence
            
        # Apply second-stage extreme hyperbolic cliff transformation
        # This creates an almost binary response, perfect for biometric verification
        scaled_similarity = hyperbolic_cliff(base_scaled, steepness=10.0, midpoint=0.5)
            
        # Add much more aggressive biometric penalties to forcefully separate different voices
        # These penalties are essential for preventing false positives
        
        # Always apply the biometric feature check, not just in a specific range
        # Get critical features that are most reliable for speaker identification
        spec_dist_sim = group_similarities.get('spectral_dist', 0)
        flat_sim = group_similarities.get('flatness', 0) 
        global_sim = group_similarities.get('global_stats', 0)
        centroid_sim = group_similarities.get('centroid', 0)
        
        # Revised similarity criteria to:
        # 1. Be more tolerant of content-dependent differences 
        # 2. Still strictly enforce similarity in core biometric features
        
        # Calculate a detailed, content-aware penalty that adapts to same-speaker variation
        if final_similarity > 0.65:  # Only apply to potentially matching voices, with relaxed threshold
            # Look for critical biometric feature differences
            # Relaxed thresholds for content-dependent features, while maintaining strict checks
            # on core biometric markers (spectral distribution and global stats)
            bad_markers = False
            
            # Core content-independent voice markers must be VERY similar
            if spec_dist_sim < 0.8:  # Relaxed slightly from 0.85
                bad_markers = True
                logging.debug(f"Major difference in spectral distribution: {spec_dist_sim:.4f}")
                
            # Secondary voice markers can have more variation with different content
            # These metrics are more affected by what's being said
            min_acceptable = {
                'flatness': 0.55,      # Greatly relaxed from 0.8 - can vary substantially with content 
                'centroid': 0.65,      # Greatly relaxed from 0.85 - pitch can vary a lot
                'global_stats': 0.82   # Relaxed but still important for overall voice character
            }
            
            # For verification with higher threshold, we now add a specific different-speaker detection
            # This is applied only when verifying against the higher threshold (not affecting enrollment)
            # Implementation of strict voice biometric difference detection
            different_speaker_markers = [
                spec_dist_sim < 0.65,                    # Major spectral distribution difference
                flat_sim < 0.35 and centroid_sim < 0.45, # Both flatness and centroid are very different
                global_sim < 0.7 and spec_dist_sim < 0.75 # Different global stats with different spectral shape
            ]
            
            # If any different-speaker markers are detected, flag as bad markers
            if any(different_speaker_markers):
                bad_markers = True
                logging.debug("Detected specific different-speaker voice biometric markers")
            
            # Count how many secondary features fail their thresholds
            fail_count = 0
            if flat_sim < min_acceptable['flatness']:
                fail_count += 1
                logging.debug(f"Low flatness similarity: {flat_sim:.4f}")
            if centroid_sim < min_acceptable['centroid']:
                fail_count += 1
                logging.debug(f"Low centroid similarity: {centroid_sim:.4f}")
            if global_sim < min_acceptable['global_stats']:
                fail_count += 1
                logging.debug(f"Low global stats similarity: {global_sim:.4f}")
                
            # Apply penalty if either primary marker is bad OR multiple secondary markers fail
            if bad_markers or fail_count >= 2:
                # Calculate weighted penalty based on how different the critical features are
                # But with greater tolerance for some features
                critical_features_avg = (
                    spec_dist_sim * 3.5 +      # Most important - vocal tract shape
                    flat_sim * 1.5 +           # Important but can vary with content
                    global_sim * 2.0 +         # Important, moderately content-independent  
                    centroid_sim * 1.0         # Less critical, can change with content
                ) / 8.0
                
                # Redesigned penalty system that adapts to the specific speaker differences
                if bad_markers:
                    # Primary biometric marker failure is serious - vocal tract shape should match
                    # Even stronger penalty to ensure clear separation between speakers
                    raw_penalty = 0.95 * (1.0 - critical_features_avg) 
                    
                    # Apply stronger non-linear transformation to create different zones:
                    # - Mild differences (possible content variation): very light penalty
                    # - Major differences (different speakers): extreme penalty to drive scores toward zero
                    
                    # Check for likely different speaker based on multiple key biometric markers
                    # These combinations are extremely unlikely for the same speaker
                    different_speaker_detected = (
                        (spec_dist_sim < 0.75 and flat_sim < 0.4) or  # Different vocal tract shape and resonance
                        (spec_dist_sim < 0.7 and global_sim < 0.75) or # Different spectral distribution and overall characteristics
                        (flat_sim < 0.3 and centroid_sim < 0.4) # Drastically different voice qualities
                    )
                    
                    if different_speaker_detected:
                        # Apply catastrophic penalty for likely different speakers
                        # This will drive the similarity score close to zero
                        logging.debug("DIFFERENT SPEAKER DETECTED: Applying maximum penalty")
                        penalty = min(0.95, scaled_similarity * 0.9)  # Reduce to at most 10% of current value
                    elif raw_penalty < 0.15:
                        # Mild differences: reduce penalty further (likely same speaker, different content)
                        penalty = raw_penalty * 0.6  # Even more forgiving for minor differences
                    else:
                        # Larger differences: increase penalty (likely different speakers)
                        penalty = 0.15 + (raw_penalty - 0.15) * 1.5  # More aggressive penalty slope
                else:
                    # Secondary feature failures are less critical, especially for different content
                    # Very mild penalty unless the combined feature differences are severe
                    raw_penalty = 0.3 * (1.0 - critical_features_avg)
                    
                    # Apply non-linear curve that reduces penalty for minor differences
                    # but maintains it for major ones
                    penalty = raw_penalty ** 1.5  # Power function makes small penalties smaller
                
                # Apply penalty
                scaled_similarity = max(0.0, scaled_similarity - penalty)
                logging.debug(f"Applied biometric feature penalty: -{penalty:.4f}")
            
        # Ensure in range [0, 1] and not NaN
        if np.isnan(scaled_similarity):
            logging.error("NaN detected in scaled_similarity! Defaulting to 0.0")
            scaled_similarity = 0.0
        else:
            scaled_similarity = max(0.0, min(1.0, scaled_similarity))
            
        # Log detailed group similarities for debugging
        logging.debug("Group similarities:")
        for group, sim in group_similarities.items():
            logging.debug(f"  {group}: {sim:.4f} (weight: {group_weights[group]})")
        
        logging.debug(f"Final raw similarity: {final_similarity:.4f}")
        logging.debug(f"Final scaled similarity: {scaled_similarity:.4f}")
        
        # Add some randomness for testing different voices
        # This ensures our embeddings are actually capturing voice differences
        # final_similarity = 0.3 + np.random.random() * 0.4  # Random between 0.3-0.7 for testing
        
        return float(scaled_similarity)
        
    except Exception as e:
        logging.error(f"Error calculating voice similarity: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0
