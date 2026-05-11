import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import time
import numpy as np

from scipy.interpolate import RegularGridInterpolator
from direct.task import Task

from gtrack.config import Detection
from gtrack.module import GTrackModule2D

class Processor:
    def __init__(self, queue_1, queue_2, queue_out, cfg_radar, cfg_gtrack):

        self.q1 = queue_1
        self.q2 = queue_2
        self.q_out = queue_out  # File pour envoyer les résultats au visualiseur
        
        self.latest_msg = {}
        self.msg_count = set()

        # Config Radar & GTrack
        self.phi = cfg_radar["phi"]
        self.r_idxs = cfg_radar["range_idx"]
        self.treshold = cfg_gtrack.min_snr_threshold
        
        # Offsets et angles
        self.x1, self.y1 = cfg_radar["offset_x_1"], cfg_radar["offset_y_1"]
        self.x2, self.y2 = cfg_radar["offset_x_2"], cfg_radar["offset_y_2"]
        self.angle_1, self.angle_2 = cfg_radar["angle_1"], cfg_radar["angle_2"]

        # Grilles cartésiennes
        self.x = np.arange(-cfg_radar["width"], cfg_radar["width"], 1)
        self.y = self.r_idxs
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')

        self.tracker = GTrackModule2D(cfg_gtrack)
        
        # Background removal 
        self.CLUTTER_LEARN = 50
        self.clutter_frames = []
        self.clutter_map = None


    def process(self, task):
        """
        Update task that processes the radar data from the queues, performs beamforming,

        Parameters
        ----------
        task : Task (unused)
            The task object provided by Panda3D's task manager.
        """
    
        # 1. Récupération des données
        try:
            for pid, q in enumerate((self.q1, self.q2)):
                while not q.empty():
                    msg = q.get_nowait()
                    if msg[0] == 'bev':
                        self.latest_msg[pid] = msg[1]
                        self.msg_count.add(pid)
        except: pass

        # 2. Traitement si on a les deux radars
        if self.msg_count == {0, 1}:
            # --- Logique de fusion / interpolation ---

            # Unpack the latest message
            bf_1 = self.latest_msg[0]
            bf_2 = self.latest_msg[1]

            phi1 = np.arctan2((self.Y - self.y1).ravel(), (self.X - self.x1).ravel()) - self.angle_1
            r1   = np.hypot(self.X.ravel() - self.x1, self.Y.ravel() - self.y1)
            cart2pol1 = np.column_stack((phi1, r1))

            phi2 = np.arctan2((self.Y - self.y2).ravel(), (self.X - self.x2).ravel()) - self.angle_2
            r2   = np.hypot(self.X.ravel() - self.x2, self.Y.ravel() - self.y2)
            cart2pol2 = np.column_stack((phi2, r2))

            # Cartesian interpolators
            interp1 = RegularGridInterpolator(
                (self.phi, self.r_idxs),  # φ axis, r axis
                bf_1,
                method='linear', bounds_error=False, fill_value=0
            )
            interp2 = RegularGridInterpolator(
                (self.phi, self.r_idxs),
                bf_2,
                method='linear', bounds_error=False, fill_value=0
            )

            # sample at every global (x,y) for each radar
            Z1 = interp1(cart2pol1).reshape(self.X.shape)
            Z2 = interp2(cart2pol2).reshape(self.X.shape)

            # Fuse
            #Z_cart = (Z1 * Z2)
            Z_cart = np.maximum(Z1, Z2)
            #overlap = (Z1 > 0) & (Z2 > 0)
            #Z_cart = np.where(overlap, (Z1 + Z2) / 2, np.maximum(Z1, Z2))

            # Build a Cartesian->grid interpolator once for the fused map
            interp_cart2pol = RegularGridInterpolator(
                (self.y, self.x),  # note order (row=y, col=x)
                Z_cart,
                method='linear',
                bounds_error=False,
                fill_value=0
            )

            # Sample back on your original polar mesh
            PHI, R = np.meshgrid(self.phi, self.r_idxs, indexing='ij')
            pts_back = np.column_stack((
                (R * np.sin(PHI)).ravel(),  # y
                (R * np.cos(PHI)).ravel()  # x
            ))
            Z_polar = interp_cart2pol(pts_back).reshape(PHI.shape)

            # Flip the polar map to match the expected orientation
            Z_polar = np.flip(Z_polar, axis=0)
            
            to_plot = np.abs(Z_polar)
            max_val = np.max(to_plot)
            if max_val > 0: to_plot /= max_val

            # --- Background / Clutter ---
            is_learning = len(self.clutter_frames) < self.CLUTTER_LEARN
            if is_learning:
                self.clutter_frames.append(to_plot.copy())
                self.clutter_map = np.mean(self.clutter_frames, axis=0)
            else:
                to_plot = np.clip(to_plot - self.clutter_map, 0, None)

            to_plot = to_plot ** 8

            # --- GTrack ---
            detections = [
                Detection(r=self.r_idxs[i], az=self.phi[j], v=0, snr=to_plot[j, i])
                for i in range(len(self.r_idxs))
                for j in range(len(self.phi))
                if to_plot[j, i] >= self.treshold
            ]
            gtrack_output = self.tracker.step(detections)

            # 3. ENVOI DES RÉSULTATS
            # On envoie un dictionnaire propre à la queue de sortie
            self.q_out.put({
                "heatmap": to_plot,
                "tracks": gtrack_output['tracks'],
                "learning_left": self.CLUTTER_LEARN - len(self.clutter_frames) if is_learning else 0
            })
            
            self.msg_count.clear()

        return task.cont