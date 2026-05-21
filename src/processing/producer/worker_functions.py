import numpy as np

from scipy.signal import convolve2d
from sklearn.cluster import DBSCAN

# ---------------------------------------------------------------------------
# TX1 and TX2 share the same azimuth positions but differ in elevation.
# The elevation phase difference between TX2 and TX1 on the same RX channel
# encodes sin(θ_el):
#
#   Δφ(r, d) = angle( Σ_rx  conj(TX1[rx, d, r]) · TX2[rx, d, r] )
#   sin(θ_el) = Δφ · λ / (2π · d_el)
#
# d_el = λ/2 for AWR1843 → sin(θ_el) = Δφ / π
# ---------------------------------------------------------------------------

# Antenna index groupings (TX1 → TX3 → TX2)
TX1_ROWS = [0, 1, 2, 3]   # ground row
TX3_ROWS = [4, 5, 6, 7]   # ground row, azimuth shifted
TX2_ROWS = [8, 9, 10, 11] # elevated row
 
# Elevation separation between TX1 and TX2 rows (in units of λ)
# AWR1843: d_el = λ/2
D_EL_WAVELENGTHS = 0.5

# ---------------------------------------------------------------------------
# Elevation angle estimation from TX1/TX2 phase difference
# ---------------------------------------------------------------------------
 
def estimate_elevation_map(beat_freq_data, dets):
    """
    Estimate sin(elevation_angle) for every detected (doppler, range) bin
    using the phase difference between TX2 and TX1 antenna rows.
 
    Parameters
    ----------
    beat_freq_data : np.ndarray  shape (12, N_doppler, N_range)
    dets           : np.ndarray  bool shape (N_doppler, N_range)
 
    Returns
    -------
    sin_el_map : np.ndarray  shape (N_doppler, N_range)
        sin(θ_el) at each detected bin; 0 where no detection.
    el_map_deg : np.ndarray  shape (N_doppler, N_range)
        elevation angle in degrees (for display / debug).
    """
    # Average across RX channels within each TX row → shape (N_doppler, N_range)
    tx1 = np.mean(beat_freq_data[TX1_ROWS], axis=0)
    tx2 = np.mean(beat_freq_data[TX2_ROWS], axis=0)
 
    # Cross-correlation phase → Δφ in [-π, π]
    cross  = tx2 * np.conj(tx1)
    delta_phi = np.angle(cross)          # shape (N_doppler, N_range)
 
    # sin(θ_el) = Δφ / (2π · d_el)
    sin_el = delta_phi / (2.0 * np.pi * D_EL_WAVELENGTHS)
    sin_el = np.clip(sin_el, -1.0, 1.0)
 
    sin_el_map = np.where(dets, sin_el, 0.0)
    el_map_deg = np.degrees(np.arcsin(sin_el_map))
 
    return sin_el_map, el_map_deg

def beamform_2d_s(beat_freq_data, radar_params, x_locs, dets):
    """
    Performs 2D beamforming along the azimuth (horizontal) dimension, this results in a bird eye view image.

    Parameters
    ----------
    beat_freq_data : np.ndarray
        The beat frequency data, typically a 3D array.
    phi_s : float
        The starting azimuth angle in degrees.
    phi_e : float
        The ending azimuth angle in degrees.
    phi_res : float
        The azimuth angle resolution in degrees.
    x_locs : np.ndarray
        The x-coordinates of the antennas.
    r_idxs : np.ndarray
        The range indices corresponding to the beat frequency data.
    radar_params : dict
        A dictionary containing radar parameters such as sample rate, number of range samples, etc.
    dets : np.ndarray
        The detections from the CFAR process.

    Returns
    -------
    sph_pwr : np.ndarray
        The spherical power array after beamforming, with shape (num_phi, num_range).
    """

    # Radar parameters
    lm = radar_params["lm"]

    # Get the azimuth angles and range indices
    phi = radar_params["phi"]
    num_phi = len(phi)
    r_idxs = radar_params["range_idx"]

    # Compute the phase shifts for each azimuth angle
    angles = x_locs * np.cos(phi[:, np.newaxis])
    phase_shifts = np.exp((1j * 2 * np.pi / lm) * angles)

    # Initialize the spherical power array
    r_idx, d_idx = np.nonzero(dets)
    sph_pwr = np.zeros((num_phi, r_idxs.shape[0]), dtype=np.complex64)

    # Apply the phase shifts to the beat frequency data and sum over the antennas
    for d, r in zip(r_idx, d_idx):
        beat = beat_freq_data[:, d, r]
        beamformed_signal = beat[np.newaxis, :] * phase_shifts
        sph_pwr[:, r] = np.maximum(sph_pwr[:, r], np.abs(np.sum(beamformed_signal, axis=-1)))

    return sph_pwr

def beamform_and_elevation(beat_freq_data, radar_params, x_locs, dets):
    """
    Convenience wrapper: runs azimuth beamforming AND elevation estimation
    in one call so the producer only loops over detections once.

    Returns
    -------
    bev      : np.ndarray  shape (num_phi, num_range)   - azimuth power map
    sin_el   : np.ndarray  shape (N_doppler, N_range)   - sin(θ_el) per bin
    el_deg   : np.ndarray  shape (N_doppler, N_range)   - elevation in degrees
    """
    bev            = beamform_2d_s(beat_freq_data, radar_params, x_locs, dets)
    sin_el, el_deg = estimate_elevation_map(beat_freq_data, dets)
    return bev, sin_el, el_deg

def cfar_ca_2d(power_map,
               num_train_range: int = 10,
               num_train_doppler: int = 8,
               num_guard_range: int = 2,
               num_guard_doppler: int = 2,
               rate_fa: float = 1e-5):
    """
    2D Cell-Averaging CFAR on a (range × Doppler) power map.

    Parameters
    ----------
    power_map : 2D np.ndarray
        The incoherent power map |X|^2 over (range, Doppler).
    num_train_range : int
        # of training cells on each side in range
    num_train_doppler : int
        # of training cells on each side in Doppler
    num_guard_range : int
        # of guard cells on each side in range
    num_guard_doppler : int
        # of guard cells on each side in Doppler
    rate_fa : float
        Desired probability of false alarm

    Returns
    -------
    detection_map : 2D bool np.ndarray
        True where power_map exceeds the CFAR threshold.
    """

    Tr, Td = num_train_range, num_train_doppler
    Gr, Gd = num_guard_range, num_guard_doppler

    # full window half–sizes
    Wr = Tr + Gr
    Wd = Td + Gd

    # number of training cells total
    Nwin = (2*Wr+1)*(2*Wd+1)
    Nguard = (2*Gr+1)*(2*Gd+1)
    Ntrain = Nwin - Nguard

    # build convolution kernels
    kernel_win   = np.ones((2*Wr+1, 2*Wd+1), dtype=float)
    kernel_guard = np.ones((2*Gr+1,2*Gd+1), dtype=float)

    # sum over full window
    sum_win   = convolve2d(power_map, kernel_win,   mode='same', boundary='fill', fillvalue=0)
    # sum over guard+CUT region
    sum_guard = convolve2d(power_map, kernel_guard, mode='same', boundary='fill', fillvalue=0)

    # training‐cell sum = window minus guard (which includes the CUT)
    sum_train = sum_win - sum_guard

    # noise estimate (average of training cells)
    noise_level = sum_train / float(Ntrain)

    # CFAR threshold multiplier (cell–averaging formula)
    alpha = Ntrain * (rate_fa**(-1.0/Ntrain) - 1.0)
    threshold = alpha * noise_level

    # detection mask
    return power_map > threshold


def process_frame(range_fft, cfar_params):
    """
    Process a single frame of range FFT data to detect targets using CFAR.

    Parameters
    ----------
    range_fft : np.ndarray
        The range FFT data, typically a 2D array of shape (N_ant, N_R).
    cfar_params : dict
        A dictionary containing CFAR parameters such as number of training cells, guard cells, and threshold scale.

    Returns
    -------
    dets : np.ndarray
        A 2D boolean array indicating detected targets, where True indicates a detection.
    """

    # Doppler FFT
    rd_cube = np.fft.fft(range_fft, axis=1)    # → (N_ant, N_D=N_adc, N_R=N_chirps)

    # Build RD magnitude for CFAR (average across antennas)
    rd_map = np.mean(np.abs(rd_cube)**2, axis=0)  # shape (N_R, N_D)

    # CFAR detections
    dets = cfar_ca_2d(rd_map,
                    cfar_params["num_train_r"],
                    cfar_params["num_train_d"],
                    cfar_params["num_guard_r"],
                    cfar_params["num_guard_d"],
                    cfar_params["threshold_scale"])

    return dets


def compute_dbscan(output_top, r_idxs, phi, eps=0.5, min_samples=5, p_treshold= 98):
    """
    Compute DBSCAN clustering on the output of the beamforming process.

    Parameters
    ----------
    output_top : np.ndarray
        The output of the beamforming process, typically a 2D array.
    r_idxs : np.ndarray
        The range indices corresponding to the output.
    phi : np.ndarray
        The azimuth angles corresponding to the output.
    eps : float
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples : int
        The number of samples in a neighborhood for a point to be considered as a core point.

    Returns
    -------
    db : DBSCAN
        The fitted DBSCAN model containing the cluster labels.
    """

    # Build full coordinate grid
    phi_rad_2d, r_idxs_2d = np.meshgrid(phi, r_idxs, indexing='ij')  # shape: (180, 140)

    x_coords_m = np.cos(phi_rad_2d) * r_idxs_2d  # shape: (180, 140)
    z_coords_m = np.sin(phi_rad_2d) * r_idxs_2d  # shape: (180, 140)

    # Flatten for DBSCAN
    points = np.stack([x_coords_m.ravel(), z_coords_m.ravel()], axis=1)
    powers = output_top.ravel()

    # Keep only high-power points
    threshold = np.percentile(powers, p_treshold)
    valid_mask = powers > threshold
    points_thresh = points[valid_mask]

    # DBSCAN
    db = DBSCAN(eps = 0.5, min_samples=min_samples).fit(points_thresh)

    return db

