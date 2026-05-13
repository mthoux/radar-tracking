import sys
import time
import warnings
from multiprocessing import Process, Queue, Event
from typing import Dict, Any

from .producer import producer_real_time_1843
from .visualizer import Visualizer
from .processor import Processor

# Suppress COM/User warnings before they trigger
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2  # Multithreading concurrency mode for COM

def consumer(
    q_radar1: Queue, 
    q_radar2: Queue, 
    cfg_radar: Dict[str, Any], 
    cfg_gtrack: Any, 
    stop_event: Event
) -> None:
    """
    Orchestrates the processing and visualization pipeline.
    
    This function runs in a dedicated process. It connects the data 
    processor to the visualizer using an internal results queue.
    """
    # Internal queue to bridge processed tracks to the UI
    q_results = Queue(maxsize=1)
    
    # Initialize the processing engine
    proc = Processor(q_radar1, q_radar2, q_results, cfg_radar, cfg_gtrack)
    
    # Initialize the visualizer
    vis = Visualizer(q_results, cfg_radar, stop_event)
    
    # Register the processing loop into Panda3D's task manager
    vis.taskMgr.add(proc.process, "RadarProcessingTask")
    
    # Start the visualization loop (blocking)
    vis.run()

def main(cfg_radar: Dict[str, Any], cfg_gtrack: Any, cfg_cfar: Dict[str, Any]) -> None:
    """
    Main entry point to launch the real-time radar streaming system.

    Spawns producer processes for data acquisition and a consumer 
    process for processing/visualization. Handles graceful shutdown.

    Args:
        cfg_radar: Physical radar parameters (offsets, resolution, etc.)
        cfg_gtrack: GTrack algorithm configuration object.
        cfg_cfar: Constant False Alarm Rate (CFAR) detection parameters.
    """
    # Communication channels - Using maxsize=1 to ensure real-time processing 
    # by dropping old frames if the processor lags.
    q_main_1 = Queue(maxsize=1)
    q_main_2 = Queue(maxsize=1)
    stop_event = Event()

    # Define background workers
    # 1. Producers: Fetch raw data from network/hardware
    producers = [
        Process(
            name="Producer_Radar_1",
            target=producer_real_time_1843,
            args=(q_main_1, cfg_radar, cfg_cfar, 4096, 4098, "192.168.33.30", "192.168.33.180"),
            daemon=True
        ),
        Process(
            name="Producer_Radar_2",
            target=producer_real_time_1843, 
            args=(q_main_2, cfg_radar, cfg_cfar, 4099, 5000, "192.168.33.32", "192.168.33.182"), 
            daemon=True
        )
    ]

    # 2. Consumer: Process data and render UI
    ui_consumer = Process(
        name="UI_Consumer",
        target=consumer, 
        args=(q_main_1, q_main_2, cfg_radar, cfg_gtrack, stop_event), 
        daemon=True
    )

    all_processes = producers + [ui_consumer]

    print("⌛ Initializing system...")
    for p in all_processes:
        p.start()

    print("✅ System active. Press Ctrl+C or close the window to exit.")

    try:
        # Monitor the stop_event triggered by the Visualizer or KeyboardInterrupt
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑  User interruption detected.")
    finally:
        print("🛑 Shutting down processes...")
        stop_event.set() # Ensure everyone knows we are stopping
        
        for p in all_processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        
        print("✅ Shutdown complete.")