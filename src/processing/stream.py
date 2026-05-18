import argparse
import numpy as np
from src.processing.consumer.gtrack.config import GTrackConfig2D

import sys
import time
import warnings
from multiprocessing import Process, Queue, Event

from .producer.worker import process
from .consumer.visualizer import Visualizer
from .consumer.fuser import Fuser

# Suppress COM/User warnings before they trigger
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2  # Multithreading concurrency mode for COM

def consumer(q_radar1, q_radar2, cfg_radar, cfg_gtrack, stop_event):
    q_results = Queue(maxsize=1)

    fuser = Fuser(q_radar1, q_radar2, q_results, cfg_radar, cfg_gtrack)
    visualizer = Visualizer(q_results, cfg_radar, stop_event)
    
    visualizer.taskMgr.add(fuser.process, "RadarProcessingTask")
    visualizer.run()

def launch_pipeline(cfg_radar, cfg_gtrack, cfg_cfar, cfg_network) -> None:
   
    q_main_1 = Queue(maxsize=1)
    q_main_2 = Queue(maxsize=1)
    stop_event = Event()

    data_producers = [
        Process(
            name="Producer_Radar_1",
            target=process,
            args=(q_main_1, cfg_radar, cfg_cfar, cfg_network["radar_1"]["ports"][0], cfg_network["radar_1"]["ports"][1], cfg_network["radar_1"]["ip_dev"], cfg_network["radar_1"]["ip_host"]),
            daemon=True
        ),
        Process(
            name="Producer_Radar_2",
            target=process, 
            args=(q_main_2, cfg_radar, cfg_cfar, cfg_network["radar_2"]["ports"][0], cfg_network["radar_2"]["ports"][1], cfg_network["radar_2"]["ip_dev"], cfg_network["radar_2"]["ip_host"]), 
            daemon=True
        )
    ]
    data_consumer = Process(
        name="Consumer",
        target=consumer, 
        args=(q_main_1, q_main_2, cfg_radar, cfg_gtrack, stop_event), 
        daemon=True
    )

    processes = data_producers + [data_consumer]

    print("⌛ Initializing system...")
    for p in processes:
        p.start()

    print("✅ System active. Press Ctrl+C or close the window to exit.")

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑  User interruption detected.")
    finally:
        print("🛑 Shutting down processes...")
        stop_event.set() # Ensure everyone knows we are stopping
        
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        
        print("✅ Shutdown complete.")

def main():

    # Arguments
    parser = argparse.ArgumentParser(description="Radar Stream Processing")
    parser.add_argument('-nobgrm', action='store_false', dest='bg_removal', 
                        help="Désactive la suppression du fond (Background Removal)")
    args = parser.parse_args()

    cfg_radar = {
        "nb_radar" : 1,
        "range_res": 0.044,
        "range_idx": np.arange(0, 100, 1),
        "phi": np.deg2rad(np.arange(0, 180, 1)),
        "width": 100,
        "offset_x_1": +int(10 / (0.044 * 100)), #cm to idx : int(cm / (range_res_m * 100))
        "offset_x_2": -int(10 / (0.044 * 100)),
        "offset_y_1": 0.0,
        "offset_y_2": 0.0,
        "angle_1": np.deg2rad(0),
        "angle_2": np.deg2rad(0),
        "n_radar": 2,
        "num_tx": 3,
        "num_rx": 4,
        "num_doppler": 16,
        "num_range": 992,
        "sample_rate": 5166000,
        "c": 3e8,
        "lm": 3e8 / 77e9, # c / f
        "slope": 70.150e12,
        "do_bg_removal": args.bg_removal
    }

    cfg_network = {
        "radar_1": {
            "ip_dev": "192.168.33.30",
            "ip_host": "192.168.33.180",
            "ports": [4096, 4098]
        },
        "radar_2": {
            "ip_dev": "192.168.33.32",
            "ip_host": "192.168.33.182",
            "ports": [4099, 5000]
        }
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
        max_points=200, # max detections per frame
        max_tracks=5,   # max simultaneous tracks
        dt=0.6,         # time between frames (s)
        process_noise=0.5,  # Q spectral density
        meas_noise_range=2.0,   # σ² range noise (m²)
        meas_noise_az=1,        # σ² azimuth noise (rad²)
        gating_threshold=6, # ≈95% gate for 2-DOF chi²
        alloc_range_gate=0.3,   # cluster gate (m)
        alloc_az_gate=np.deg2rad(5),  # cluster gate (rad)
        alloc_vel_gate=20,          # cluster gate (m/s)
        min_cluster_points=10,  # you can increase if you want multi-point seeds
        alloc_snr_threshold=0.5,    # sum-SNR threshold
        min_snr_threshold=0.005,    # min SNR for new track
        init_state_cov=1.0,         # starting P for new tracks
        det_to_active_count=1,      # hits needed to go ACTIVE
        det_to_free_count=8,        # misses to drop DETECTION
        act_to_free_count=10,        # misses to drop ACTIVE
        presence_zones=[],          # e.g. [PresenceZone2D(-10,10,-5,5)]
        pres_on_count=5,            # frames to confirm presence on
        pres_off_count=3            # frames to confirm presence off
    )

    # cfg_gtrack = GTrackConfig2D(
    #     max_points=200,  # max detections per frame
    #     max_tracks=5,  # max simultaneous tracks
    #     dt=0.2,  # time between frames (s)
    #     process_noise=0.5,  # Q spectral density
    #     meas_noise_range=2.0,  # σ² range noise (m²)
    #     meas_noise_az=1,  # σ² azimuth noise (rad²)
    #     gating_threshold=6,  # ≈95% gate for 2-DOF chi²
    #     alloc_range_gate=0.5,  # cluster gate (m)
    #     alloc_az_gate=np.deg2rad(10),  # cluster gate (rad)
    #     alloc_vel_gate=20,  # cluster gate (m/s)
    #     min_cluster_points=10,  # you can increase if you want multi-point seeds
    #     alloc_snr_threshold=1,  # sum-SNR threshold
    #     min_snr_threshold=0.01,  # min SNR for new track
    #     init_state_cov=1.0,  # starting P for new tracks
    #     det_to_active_count=12,  # hits needed to go ACTIVE
    #     det_to_free_count=3,  # misses to drop DETECTION
    #     act_to_free_count=8,  # misses to drop ACTIVE
    #     presence_zones=[],  # e.g. [PresenceZone2D(-10,10,-5,5)]
    #     pres_on_count=5, # frames to confirm presence on
    #     pres_off_count=3 # frames to confirm presence off
    # )

    print("⌛️ Starting streaming...")
    launch_pipeline(cfg_radar, cfg_gtrack, cfg_cfar, cfg_network)

if __name__ == "__main__":
    main()