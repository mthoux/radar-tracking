import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import time
import numpy as np

from multiprocessing import Process, Queue

import matplotlib
matplotlib.use('Qt5Agg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-dark')

from panda3d.core import loadPrcFileData
loadPrcFileData('', 'window-type none')   # no native GL window

from .prod_dca import producer_real_time_1843


def consumer(q1, q2, cfg_radar, cfg_gtrack):
    """
    Consumer function that processes data from two queues, performs beamforming,

    Parameters
    ----------
    q1 : Queue
        The first queue containing radar data.
    q2 : Queue
        The second queue containing radar data.
    cfg_radar : dict
        Configuration dictionary for the radar, including parameters like phi, range indices, and offsets.
    cfg_gtrack : dict
        Configuration dictionary for the GTrack module, including parameters like minimum SNR threshold.
    """
    app = MyApp(q1, q2, cfg_radar, cfg_gtrack)
    app.run()



def main(cfg_radar, cfg_gtrack, cfg_cfar):
    """
    Main function to start the real-time radar streaming and processing.

    Parameters
    ----------
    cfg_radar : dict
        Configuration dictionary for the radar, including parameters like phi, range indices, and offsets.
    cfg_gtrack : dict
        Configuration dictionary for the GTrack module, including parameters like minimum SNR threshold.
    cfg_cfar : dict
        Configuration dictionary for the CFAR processing, including parameters like window size and guard size.
    """

    # Set up the queues
    q_main_1 = Queue(maxsize=1)  # ❗️ Only keep latest
    q_main_2 = Queue(maxsize=1)

    # Create producer and consumer processes
    producers = [
        Process(target=producer_real_time_1843,args=(q_main_1, cfg_radar, cfg_cfar, 4096, 4098, "192.168.33.30", "192.168.33.180"), daemon=True),
        Process(target=producer_real_time_1843, args=(q_main_2, cfg_radar, cfg_cfar, 4099, 5000, "192.168.33.32", "192.168.33.182"), daemon=True)
    ]
    consumers = [Process(target=consumer, args=(q_main_1, q_main_2, cfg_radar, cfg_gtrack), daemon=True)]

    # Start the producer and consumer processes
    for p in producers: p.start()
    for c in consumers: c.start()

    print("✅ Streaming started.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for p in producers: p.terminate()
        for c in consumers: c.terminate()
        for p in producers: p.join()
        for c in consumers: c.join()
        print("✅ Shutdown complete.")

