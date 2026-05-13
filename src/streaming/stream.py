import sys
import argparse
import numpy as np
from . import init
from gtrack.config import GTrackConfig2D

def main():
    """
    Initialize radar parameters and start real-time processing.
    """

    # Gestion des arguments
    parser = argparse.ArgumentParser(description="Radar Stream Processing")
    parser.add_argument('-nobgrm', action='store_false', dest='bg_removal', 
                        help="Désactive la suppression du fond (Background Removal)")
    args = parser.parse_args()

    # Physical constants and radar specifications
    c = 3e8
    f = 77e9
    #slope = 70.150e6
    slope = 70.150e12
    sample_rate = 5166000
    num_range = 992
    #num_range = 256
    
    # Resolution: ~0.044 m per index
    bandwith = slope * (num_range / sample_rate)
    #range_res_m = c / (2*bandwith)
    range_res_m = 0.044

    # Beamforming and spatial parameters
    r_idxs = np.arange(0, 100, 1)
    phi = np.deg2rad(np.arange(0, 180, 1))
    width = 100 

    def cm_to_idx(cm):
        return int(cm / (range_res_m * 100))

    def idx_to_cm(idx):
        return idx * range_res_m * 100

    # Radar geometry configuration (20cm spacing)
    D = cm_to_idx(10)
    angle = 0
    angle_rad = np.deg2rad(angle)

    cfg_radar = {
        "nb_radar" : 1,
        "range_res": range_res_m,
        "range_idx": r_idxs,
        "phi": phi,
        "width": width,
        "offset_x_1": +D,
        "offset_x_2": -D,
        "offset_y_1": 0.0,
        "offset_y_2": 0.0,
        "angle_1": angle_rad,
        "angle_2": angle_rad,
        "n_radar": 2,
        "num_tx": 3,
        "num_rx": 4,
        "num_doppler": 16,
        "num_range": num_range,
        "sample_rate": sample_rate,
        "c": c,
        "lm": c / f,
        "slope": slope,
        "do_bg_removal": args.bg_removal
    }

    # CFAR (Constant False Alarm Rate) detection parameters
    cfg_cfar = {
        "num_train_r": 10,
        "num_train_d": 8,
        "num_guard_r": 2,
        "num_guard_d": 2,
        "threshold_scale": 1e-3
    }

    # Gtrack algorithm configuration
    cfg_gtrack = GTrackConfig2D(
        max_points=200,
        max_tracks=5,
        dt=0.6,
        process_noise=0.5,
        meas_noise_range=2.0,
        meas_noise_az=1,
        gating_threshold=6,
        alloc_range_gate=0.3,
        alloc_az_gate=np.deg2rad(5),
        alloc_vel_gate=20,
        min_cluster_points=10,
        alloc_snr_threshold=0.5,
        min_snr_threshold=0.005,
        init_state_cov=1.0,
        det_to_active_count=1,
        det_to_free_count=6,
        act_to_free_count=8,
        presence_zones=[],
        pres_on_count=5,
        pres_off_count=3
    )

    print("⌛️ Starting streaming...")
    init.main(cfg_radar, cfg_gtrack, cfg_cfar)

if __name__ == "__main__":
    main()