import argparse
import numpy as np
from src.processing.consumer.gtrack.config import GTrackConfig2D

import sys
import time
import warnings
from multiprocessing import Process, Queue, Event

from ..producer.worker import process
from ..consumer.visualizer import Visualizer
from ..consumer.fuser import Fuser

# Suppress COM/User warnings before they trigger
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2  # Multithreading concurrency mode for COM

def consumer(q_radar1, q_radar2, cfg_radar, cfg_gtrack, stop_event, cfg_arduino):
    q_results = Queue(maxsize=1)

    fuser = Fuser(q_radar1, q_radar2, q_results, cfg_radar, cfg_gtrack, cfg_arduino)
    visualizer = Visualizer(q_results, cfg_radar, stop_event)
    
    visualizer.taskMgr.add(fuser.process, "RadarProcessingTask")
    visualizer.run()

def launch_pipeline(cfg_radar, cfg_gtrack, cfg_cfar, cfg_network, cfg_arduino) -> None:
   
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
        args=(q_main_1, q_main_2, cfg_radar, cfg_gtrack, stop_event, cfg_arduino), 
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
        "range_res": 0.044,
        "range_idx": np.arange(0, 100, 1),
        "phi": np.deg2rad(np.arange(-50, 51, 1)),
        "width": 50,
        "D_x": 0.60, # Distance that separate both radars on axis X (in m)
        "angle_1": np.deg2rad(0),
        "angle_2": np.deg2rad(0),
        "num_tx": 3,                # Number of TX (for worker)
        "num_rx": 4,                # Number of RX (for worker)
        "num_doppler": 16,          # (for worker)
        "num_range": 992,           # Total number of samples made by the radar (for worker)
        "sample_rate": 5166000,
        "c": 3e8,
        "lm": 3e8 / 77e9, # c / f
        "slope": 70.150e12,
        "radar_mount_height": 0.8,   # metres — set to 0.0 if radar is on the floor
        "do_bg_removal": args.bg_removal,
        "smoothing": True,
        "alpha_smoothing": 0.5,  # Facteur de lissage (0.1 = très lent/stable, 0.9 = très nerveux)
        "fall_detection_active": False,
        "debug": {
            "do_plot_individually": False
        }
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

    cfg_arduino = {
        "port": "/dev/tty.usbmodem1201",
        "warning": False
    }

    # Gtrack algorithm configuration
    cfg_gtrack = GTrackConfig2D(
        max_points=200,
        max_tracks=2,
        dt=0.6,
        process_noise=0.05,              # was 0.05 — higher to avoid track stealing
        meas_noise_range=0.5,           # was 2.0 — tighter range gate
        meas_noise_az=0.05,              # was 1 — ±6° instead of ±57°
        gating_threshold=3,
        alloc_range_gate=0.5,           # was 0.5 — tighter
        alloc_az_gate=np.deg2rad(7),    # was 10° — tighter
        alloc_vel_gate=20,
        min_cluster_points=6,
        alloc_snr_threshold=0.5,
        min_snr_threshold=0.005,
        init_state_cov=1.0,
        det_to_active_count=3,          # was 1 — require 3 frames before ACTIVE
        det_to_free_count=6,
        act_to_free_count=8,
        presence_zones=[],
        pres_on_count=5,
        pres_off_count=3
    )

    print("⌛️ Starting streaming...")
    launch_pipeline(cfg_radar, cfg_gtrack, cfg_cfar, cfg_network, cfg_arduino)

if __name__ == "__main__":
    main()