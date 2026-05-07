import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import time
import numpy as np

from scipy.interpolate import RegularGridInterpolator
from multiprocessing import Process, Queue
from direct.showbase.ShowBase import ShowBase
from direct.task import Task

import matplotlib
matplotlib.use('Qt5Agg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-dark')

from panda3d.core import loadPrcFileData
loadPrcFileData('', 'window-type none')   # no native GL window

from PyQt5 import QtWidgets

from .prod_dca import producer_real_time_1843
from visualization.visualization import configure_ax_bf, configure_ax_db, configure_ax_gtrack, update_ax_gtrack
from utils.utils import cart2pol
from gtrack.config import Detection
from gtrack.module import GTrackModule2D

def consumer(*args):
    """
    Consommateur générique acceptant un nombre variable de files d'attente.
    Les derniers arguments doivent être les dictionnaires de configuration.
    """
    # Extraction des configurations (les deux derniers arguments)
    queues = args[:-2]
    cfg_radar = args[-2]
    cfg_gtrack = args[-1]

    # Initialisation de l'application avec la liste des queues
    app = MyApp(queues, cfg_radar, cfg_gtrack)
    app.run()


class MyApp(ShowBase):
    def __init__(self, queues, cfg_radar, cfg_gtrack):
        ShowBase.__init__(self)
        # Gestion dynamique des files d'attente
        self.queues = queues
        self.nb_radar = len(queues)
        self.latest_msg = {}
        self.msg_count = set()

        self.phi = cfg_radar["phi"]
        self.r_idxs = cfg_radar["range_idx"]
        self.treshold = cfg_gtrack.min_snr_threshold

        # Initialisation des graphiques
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='polar')
        self.im = configure_ax_bf(self.ax, self.phi, self.r_idxs)

        self.fig_3 = plt.figure(figsize=(8, 6), constrained_layout=True)
        self.ax_3 = self.fig_3.add_subplot(111)
        configure_ax_gtrack(self.ax_3, cfg_radar["width"], len(self.r_idxs))

        # Récupération dynamique des offsets (ex: offset_x_1, offset_x_2, ...)
        self.offsets_x = [cfg_radar[f"offset_x_{i+1}"] for i in range(self.nb_radar)]
        self.offsets_y = [cfg_radar[f"offset_y_{i+1}"] for i in range(self.nb_radar)]

        # Grille cartésienne globale
        self.x = np.arange(-cfg_radar["width"], cfg_radar["width"], 1)
        self.y = self.r_idxs
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')

        self.tracker = GTrackModule2D(cfg_gtrack)
        self.last_artists = []
        self.last_fps_time = time.time()
        self.frame_counter = 0
        self.fps_text = self.ax_3.text(0.00, 1.05, "", transform=self.ax_3.transAxes, color='blue')

        self.taskMgr.add(self.updateTask, "updateTask")

    def updateTask(self, task):
        # Lecture de toutes les files d'attente
        for pid, q in enumerate(self.queues):
            try:
                while not q.empty():
                    msg = q.get_nowait()
                    if msg[0] == 'bev':
                        self.latest_msg[pid] = msg[1]
                        self.msg_count.add(pid)
            except:
                continue

        # Vérification si tous les radars ont envoyé une donnée
        if len(self.msg_count) == self.nb_radar:
            z_fused = None

            for i in range(self.nb_radar):
                # Calcul des coordonnées polaires relatives à ce radar spécifique
                phi_rel = np.arctan2((self.Y - self.offsets_y[i]).ravel(), (self.X - self.offsets_x[i]).ravel())
                r_rel = np.hypot(self.X.ravel() - self.offsets_x[i], self.Y.ravel() - self.offsets_y[i])
                pts_rel = np.column_stack((phi_rel, r_rel))

                interp = RegularGridInterpolator(
                    (self.phi, self.r_idxs),
                    self.latest_msg[i],
                    method='linear', bounds_error=False, fill_value=0
                )
                
                z_radar = interp(pts_rel).reshape(self.X.shape)
                
                # Fusion multiplicative (Z1 * Z2 * ... * Zn)
                if z_fused is None:
                    z_fused = z_radar
                else:
                    z_fused *= z_radar

            # Interpolation vers la vue polaire finale
            interp_cart = RegularGridInterpolator((self.y, self.x), z_fused, method='linear', bounds_error=False, fill_value=0)
            
            PHI, R = np.meshgrid(self.phi, self.r_idxs, indexing='ij')
            pts_back = np.column_stack(((R * np.sin(PHI)).ravel(), (R * np.cos(PHI)).ravel()))
            z_polar = np.flip(interp_cart(pts_back).reshape(PHI.shape), axis=0)

            # Normalisation et affichage
            to_plot = np.abs(z_polar)
            max_val = np.max(to_plot)
            if max_val > 0: to_plot /= max_val
            to_plot = to_plot ** 8
            
            self.im.set_array(to_plot.ravel())

            # Detections et GTrack
            detections = [
                Detection(r=self.r_idxs[i], az=self.phi[j], v=0, snr=to_plot[j, i])
                for i in range(len(self.r_idxs)) for j in range(len(self.phi))
                if to_plot[j, i] >= self.treshold
            ]
            
            update_ax_gtrack(self.ax_3, self.tracker.step(detections)['tracks'], self.last_artists)

            # Mise à jour UI
            self.update_fps()
            self.fig.canvas.draw_idle()
            self.fig_3.canvas.draw_idle()
            QtWidgets.QApplication.processEvents()
            
            self.msg_count.clear()
            plt.pause(0.001)

        return Task.cont

    def update_fps(self):
        self.frame_counter += 1
        now = time.time()
        dt = now - self.last_fps_time
        if dt >= 1.0:
            fps = self.frame_counter / dt
            self.fps_text.set_text(f"FPS: {fps:.2f}")
            self.last_fps_time = now
            self.frame_counter = 0


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

    nb_radar = cfg_radar.get("nb_radar", 1)
    queues = []
    producers = []

    # Temporaire
    radar_configs = [
        {"port_in": 4096, "port_out": 4098, "ip_local": "192.168.33.30", "ip_remote": "192.168.33.180"},
        {"port_in": 4099, "port_out": 5000, "ip_local": "192.168.33.32", "ip_remote": "192.168.33.182"},
    ]

    # Initialisation dynamique des Queues et Producteurs
    for i in range(nb_radar):
        q = Queue(maxsize=1)
        queues.append(q)
        
        conf = radar_configs[i]
        p = Process(
            target=producer_real_time_1843,
            args=(q, cfg_radar, cfg_cfar, conf["port_in"], conf["port_out"], conf["ip_local"], conf["ip_remote"]),
            daemon=True
        )
        producers.append(p)

    consumers = [Process(target=consumer, args=(*queues, cfg_radar, cfg_gtrack), daemon=True)]

    for p in producers + consumers:
        p.start()

    print(f"✅ Streaming started for {nb_radar} radar(s).")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for p in producers + consumers:
            p.terminate()
            p.join()
        print("✅ Shutdown complete.")