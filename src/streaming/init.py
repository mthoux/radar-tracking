import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import time

from multiprocessing import Process, Queue, Event

from .producer import producer_real_time_1843

from .visualizer import Visualizer
from .processor import Processor

def consumer(q1, q2, cfg_radar, cfg_gtrack, stop_event):
    # 1. On crée une queue interne pour passer les résultats du calcul vers l'affichage
    q_results = Queue(maxsize=1)
    
    # 2. On instancie le "Cerveau" (Processor)
    # Note : on lui passe q1, q2 et la queue de sortie
    proc = Processor(q1, q2, q_results, cfg_radar, cfg_gtrack)
    
    # 3. On instancie le "Visage" (Visualizer)
    # Note : MyApp ne connaît plus que q_results
    vis = Visualizer(q_results, cfg_radar, stop_event)
    
    # 4. On ajoute la tâche de calcul au moteur de Panda3D
    # Panda3D va appeler proc.process() en boucle
    vis.taskMgr.add(proc.process, "processingTask")
    
    # 5. On lance la fenêtre
    vis.run()

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
    stop_event = Event()

    # Create producer and consumer processes
    producers = [
        Process(target=producer_real_time_1843,args=(q_main_1, cfg_radar, cfg_cfar, 4096, 4098, "192.168.33.30", "192.168.33.180"), daemon=True),
        Process(target=producer_real_time_1843, args=(q_main_2, cfg_radar, cfg_cfar, 4099, 5000, "192.168.33.32", "192.168.33.182"), daemon=True)
        ]
    consumers = [Process(target=consumer, args=(q_main_1, q_main_2, cfg_radar, cfg_gtrack, stop_event), daemon=True)]

    # Start the producer and consumer processes
    for p in producers: p.start()
    for c in consumers: c.start()

    print("✅ Streaming started.")

    try:
        # AU LIEU DE while True:
        while not stop_event.is_set():
            time.sleep(0.1) # Attend que la fenêtre soit fermée
    except KeyboardInterrupt:
        print("Interruption par clavier.")
    finally:
        # On nettoie tout
        for p in producers + consumers: 
            p.terminate()
            p.join()
        print("✅ Shutdown complete.")



