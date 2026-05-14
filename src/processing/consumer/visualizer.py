import sys
import time
import warnings
import numpy as np
from PyQt5 import QtWidgets
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import loadPrcFileData

# Internal visualization imports
from src.processing.consumer.visualizer_functions import (
    configure_ax_bf, 
    configure_ax_db, 
    configure_ax_gtrack, 
    update_ax_gtrack
)

# Initial configurations
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import matplotlib
matplotlib.use('Qt5Agg')  # Use TkAgg backend for interactive plotting
plt.style.use('seaborn-v0_8-dark')

# Disable native Panda3D window
loadPrcFileData('', 'window-type none')

class Visualizer(ShowBase):
    """
    Main visualization class handling radar heatmap and tracking display.
    Inherits from Panda3D ShowBase to manage tasks.
    """
    def __init__(self, queue_out, cfg_radar, stop_event):
        ShowBase.__init__(self)
        self.q_out = queue_out
        self.stop_event = stop_event     
        self.do_bg_removal = cfg_radar["do_bg_removal"]

        # Display parameters from config
        self.phi = cfg_radar["phi"]
        self.r_idxs = cfg_radar["range_idx"]

        # Matplotlib figure initialization
        gs = GridSpec(3, 2, width_ratios=[0.4, 0.6], height_ratios=[0.5, 0.25, 0.25])
        self.fig = plt.figure(figsize=(14, 12))
                
        # 1. Bird Eye View (Polar)
        self.ax = self.fig.add_subplot(gs[0, 0], projection='polar')
        self.im = configure_ax_bf(self.ax, self.phi, self.r_idxs)   

        # 2. GTrack (Cartesian)
        self.ax_3 = self.fig.add_subplot(gs[:, 1])
        configure_ax_gtrack(self.ax_3, cfg_radar["width"], len(self.r_idxs))

        # 3. 1D Plot (Power/Range Profile)
        self.ax_1d = self.fig.add_subplot(gs[1, 0])
        
        # Récupération de la résolution et création de l'axe en mètres
        res = cfg_radar.get("range_res", 1.0) # On récupère la valeur de la config
        self.r_metres = self.r_idxs * res      # Conversion des indices en mètres
        
        # On trace avec self.r_metres au lieu de self.r_idxs
        self.line_1d, = self.ax_1d.plot(self.r_metres, np.zeros_like(self.r_idxs), color='green')
        
        self.ax_1d.set_ylim(0, 1.1)
        self.ax_1d.set_xlim(self.r_metres[0], self.r_metres[-1]) # Calage parfait de l'axe
        self.ax_1d.set_title(f"Profil de Puissance (Res: {res*100:.1f}cm)")
        self.ax_1d.set_xlabel("Distance réelle (m)")
        self.ax_1d.grid(True, alpha=0.3)

        # 4. Profil Azimutal (Puissance vs Angle)
        self.ax_azimuth = self.fig.add_subplot(gs[2, 0])

        # On convertit en degrés ET on décale de -90 pour avoir l'échelle [-90, 90]
        self.phi_deg = np.degrees(self.phi) - 90 

        self.line_azimuth, = self.ax_azimuth.plot(self.phi_deg, np.zeros_like(self.phi), color='blue')
        self.ax_azimuth.set_ylim(0, 1.1)

        # Mise à jour auto des limites
        self.ax_azimuth.set_xlim(self.phi_deg[0], self.phi_deg[-1]) 
        self.ax_azimuth.set_title("Profil de Puissance par Angle (Centré)")
        self.ax_azimuth.set_xlabel("Angle (degrés)")

        # Artists and UI elements
        self.last_artists = []
        self.fps_text = self.ax_3.text(0.00, 1.05, "", transform=self.ax_3.transAxes, fontsize=10, color='blue')

        # FPS calculation state
        self.last_fps_time = time.time()
        self.frame_counter = 0

        # Event connections
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        # Launch the update task
        self.taskMgr.add(self.updateTask, "updateTask")

        # Texte pour le log des chutes (à gauche ou droite selon tes besoins)
        self.fall_log_text = self.ax_3.text(
            1.02, 1.0, "Chutes :\n—",
            transform=self.ax_3.transAxes,
            fontsize=8, color='red', verticalalignment='top'
        )

    def updateTask(self, task):
        """
        Panda3D task that updates the plots with the latest data from the queue.
        """
        # Drain the queue to keep only the latest data (anti-lag)
        data = None
        while not self.q_out.empty():
            data = self.q_out.get_nowait()

        if data is not None:

            # 1. Update Heatmap & Tracks
            self.im.set_array(data["heatmap"].ravel())
            update_ax_gtrack(self.ax_3, data["tracks"], self.last_artists)

            # Update Graphique 1D ---
            if "range_profile" in data:
                self.line_1d.set_ydata(data["range_profile"])

            # Mise à jour du profil en angle (Y)
            if "azimuth_profile" in data:
                self.line_azimuth.set_ydata(data["azimuth_profile"])

            # 2. Update Title based on learning state
            if self.do_bg_removal:
                if data["learning_left"] > 0:
                    self.ax.set_title(f"Learning background... ({data['learning_left']} frames left)")
            else:
                self.ax.set_title("Radar Active")

            # 3. Update Fall Log & Title
            if data.get("fall_events"):
                # On prend le dernier événement pour le titre
                last_event = data["fall_events"][-1]
                ts = time.strftime("%H:%M:%S", time.localtime(last_event["timestamp"]))
                self.ax_3.set_title(f"⚠ CHUTE : Track {last_event['track_id']} à {ts}", color="red")

            # Mise à jour de l'historique complet
            history = [
                f"{time.strftime('%H:%M:%S', time.localtime(e['timestamp']))} - ID {e['track_id']}"
                for e in data.get("all_falls", [])
            ]
            self.fall_log_text.set_text("Chutes :\n" + "\n".join(history[-5:])) # Affiche les 5 dernières

            # 4. Calculate FPS
            self.frame_counter += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                fps = self.frame_counter / (current_time - self.last_fps_time)
                self.fps_text.set_text(f"FPS: {fps:.2f}")
                self.last_fps_time = current_time
                self.frame_counter = 0

            # 5. Render update
            self.fig.canvas.draw_idle()
            QtWidgets.QApplication.processEvents()
            plt.pause(0.005)

        return Task.cont

    def on_close(self, event):
        """
        Callback for window closure.
        """
        self.stop_event.set()
        sys.exit(0)