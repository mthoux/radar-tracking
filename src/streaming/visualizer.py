import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import time

from direct.showbase.ShowBase import ShowBase
from direct.task import Task

import matplotlib
matplotlib.use('Qt5Agg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-dark')

from panda3d.core import loadPrcFileData
loadPrcFileData('', 'window-type none')   # no native GL window

from PyQt5 import QtWidgets

from visualization.visualization import configure_ax_bf, configure_ax_db, configure_ax_gtrack, update_ax_gtrack
from matplotlib.gridspec import GridSpec


class Visualizer(ShowBase):
    def __init__(self, queue_out, cfg_radar, stop_event):
        ShowBase.__init__(self)
        self.q_out = queue_out  # On ne reçoit plus que la queue de sortie du Processor
        self.stop_event = stop_event

        # Config minimale pour l'affichage
        self.phi = cfg_radar["phi"]
        self.r_idxs = cfg_radar["range_idx"]

        # Préparation Matplotlib (inchangé)
        gs = GridSpec(1, 2, width_ratios=[1, 1.5])
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(gs[0], projection='polar')
        self.im = configure_ax_bf(self.ax, self.phi, self.r_idxs)

        self.ax_3 = self.fig.add_subplot(gs[1])
        configure_ax_gtrack(self.ax_3, cfg_radar["width"], len(self.r_idxs))

        self.last_artists = []
        self.fps_text = self.ax_3.text(0.00, 1.05, "", transform=self.ax_3.transAxes, fontsize=10, color='blue')

        # Gestion du temps pour les FPS
        self.last_fps_time = time.time()
        self.frame_counter = 0

        # Connexion fermeture fenêtre
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        # Lancement de la tâche de mise à jour graphique
        self.taskMgr.add(self.updateTask, "updateTask")

    def updateTask(self, task):
        # On vide la file pour ne garder que la donnée la plus récente (anti-lag)
        data = None
        while not self.q_out.empty():
            data = self.q_out.get_nowait()

        if data is not None:
            # 1. Mise à jour de la Heatmap
            self.im.set_array(data["heatmap"].ravel())

            # 2. Mise à jour du Tracking
            update_ax_gtrack(self.ax_3, data["tracks"], self.last_artists)

            # 3. Mise à jour du titre (Learning ou non)
            if data["learning_left"] > 0:
                self.ax.set_title(f"Learning background... ({data['learning_left']} frames left)")
            else:
                self.ax.set_title("Radar Active")

            # 4. Calcul FPS (affichage uniquement)
            self.frame_counter += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                fps = self.frame_counter / (current_time - self.last_fps_time)
                self.fps_text.set_text(f"FPS: {fps:.2f}")
                self.last_fps_time = current_time
                self.frame_counter = 0

            # 5. Rendu
            self.fig.canvas.draw_idle()
            QtWidgets.QApplication.processEvents()

            plt.pause(0.005)

        return Task.cont

    def on_close(self, event):
        self.stop_event.set()
        sys.exit(0)
