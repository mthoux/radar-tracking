import numpy as np

from scipy.signal import convolve2d
from sklearn.cluster import DBSCAN


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

# ---------------------------------------------------------------------------
# Multi-elevation Beamforming
# ---------------------------------------------------------------------------
 
def beamform_2d_elevation(beat_freq_data, radar_params, x_locs, dets,
                          elevation_angle_rad: float):
    """
    Beamform with a fixed elevation steering angle, isolating power
    originating from a specific height slice.
 
    The phase shift per antenna becomes:
        φ = (2π/λ) · x_loc · cos(φ_az) · cos(θ_el)
    where θ_el is the elevation angle (0 = horizontal plane).
 
    The cos(θ_el) factor compresses the effective aperture as seen from
    the elevated direction, which is the correct 3D steering projection
    for a horizontal ULA (the AWR1843 virtual array lies in the horizontal
    plane so it has no physical elevation aperture, but this weighting
    selectively attenuates contributions that do NOT originate from the
    target elevation).
 
    Parameters
    ----------
    beat_freq_data : np.ndarray  (num_tx*num_rx, num_doppler, num_range)
    radar_params   : dict
    x_locs         : np.ndarray  antenna x-positions (m)
    dets           : np.ndarray  2-D boolean CFAR detection map
    elevation_angle_rad : float  target elevation angle in radians
 
    Returns
    -------
    sph_pwr : np.ndarray  shape (num_phi, num_range)
    """
    lm     = radar_params["lm"]
    phi    = radar_params["phi"]
    r_idxs = radar_params["range_idx"]
    num_phi = len(phi)
 
    # Project antenna spacing onto the steered direction
    cos_el = np.cos(elevation_angle_rad)
    angles       = x_locs * np.cos(phi[:, np.newaxis]) * cos_el
    phase_shifts = np.exp((1j * 2 * np.pi / lm) * angles)
 
    r_idx, d_idx = np.nonzero(dets)
    sph_pwr = np.zeros((num_phi, r_idxs.shape[0]), dtype=np.complex64)
 
    for d, r in zip(r_idx, d_idx):
        beat = beat_freq_data[:, d, r]
        beamformed_signal = beat[np.newaxis, :] * phase_shifts
        sph_pwr[:, r] = np.maximum(
            sph_pwr[:, r],
            np.abs(np.sum(beamformed_signal, axis=-1))
        )
 
    return sph_pwr
 
 
def beamform_multilevel(beat_freq_data, radar_params, x_locs, dets):
    """
    Run one beamforming pass per height level and return a list of BEV maps.
 
    The elevation angle for each height h is estimated as the median
    elevation across the visible range window:
        θ_el(h) = arctan(h / r_mid)
    where r_mid is the midpoint of the range window in metres.
 
    Parameters
    ----------
    heights_m : tuple of float
        Physical heights above the radar plane (metres).
        Defaults to (0.3 m, 0.9 m, 1.5 m) — floor / waist / head.
 
    Returns
    -------
    bev_levels : list of np.ndarray, one per height
        Each array has shape (num_phi, num_range).
    """
    range_res  = radar_params["range_res"]
    r_idxs     = radar_params["range_idx"]
    heights_m  = radar_params["elevation_heights"]
 
    # Representative range for elevation angle computation (mid of window, in m)
    r_mid_m = float(np.median(r_idxs)) * range_res  # metres
 
    bev_levels = []
    real_h = tuple(h - heights_m[1] for h in heights_m) # Elevation heights in the radar referential
    for dh in real_h:
        # Clamp to avoid arctan(inf) for very short ranges
        el_angle = np.arctan2(dh / max(r_mid_m, 0.1))
        bev = beamform_2d_elevation(
            beat_freq_data, radar_params, x_locs, dets, el_angle
        )
        bev_levels.append(bev)
 
    return bev_levels


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

