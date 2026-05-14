import sys
import warnings
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from direct.task import Task
from typing import Dict, Any, List
import serial

# Local imports
from src.processing.consumer.gtrack.config import Detection
from src.processing.consumer.gtrack.module import GTrackModule2D
from .fall_detection import FallDetector

# Global configuration to avoid COM initialization issues on some systems
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

class Fuser:
    """
    Handles the fusion of multiple radar streams, background subtraction, 
    and target tracking using GTrack.
    """

    def __init__(self, queue_1: Any, queue_2: Any, queue_out: Any, cfg_radar: Dict[str, Any], cfg_gtrack: Any):
        """
        Initializes the processor with radar geometry and tracking configurations.
        """
        self.q1 = queue_1
        self.q2 = queue_2
        self.q_out = queue_out
        
        # Latest data storage to sync asynchronous radar streams
        self.latest_msg = {0: None, 1: None}
        self.msg_ready = [False, False]

        # Radar & Tracking Parameters
        self.phi = cfg_radar["phi"]
        self.r_idxs = cfg_radar["range_idx"]
        self.snr_threshold = cfg_gtrack.min_snr_threshold
        
        # Geometric Offsets
        self.x1, self.y1 = cfg_radar["offset_x_1"], cfg_radar["offset_y_1"]
        self.x2, self.y2 = cfg_radar["offset_x_2"], cfg_radar["offset_y_2"]
        self.angle_1, self.angle_2 = cfg_radar["angle_1"], cfg_radar["angle_2"]

        # Define Cartesian Grid for Fusion
        self.x_grid = np.arange(-cfg_radar["width"], cfg_radar["width"], 1)
        self.y_grid = self.r_idxs
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid, indexing='xy')

        # GTrack Module Initialization
        self.tracker = GTrackModule2D(cfg_gtrack)

        # Récupération de l'option (True par défaut si absente de la config)
        self.do_bg_removal = cfg_radar.get("do_bg_removal", True)
        
        # Background / Clutter Removal State
        self.CLUTTER_LEARN_LIMIT = 50
        self.clutter_frames: List[np.ndarray] = []
        self.clutter_map: np.ndarray = None

        # PRE-COMPUTATION: Mapping Polar coordinates to Cartesian points once.
        # This prevents costly trigonometric calculations inside the processing loop.
        
        # Radar 1 Mapping
        phi1 = np.arctan2((self.Y - self.y1).ravel(), (self.X - self.x1).ravel()) - self.angle_1
        r1 = np.hypot(self.X.ravel() - self.x1, self.Y.ravel() - self.y1)
        self.pts1 = np.column_stack((phi1, r1))

        # Radar 2 Mapping
        phi2 = np.arctan2((self.Y - self.y2).ravel(), (self.X - self.x2).ravel()) - self.angle_2
        r2 = np.hypot(self.X.ravel() - self.x2, self.Y.ravel() - self.y2)
        self.pts2 = np.column_stack((phi2, r2))

        # Back-sampling Mapping (Cartesian -> Polar display)
        PHI_MESH, R_MESH = np.meshgrid(self.phi, self.r_idxs, indexing='ij')
        self.pts_back = np.column_stack((
            (R_MESH * np.sin(PHI_MESH)).ravel(), # y-coord on cartesian grid
            (R_MESH * np.cos(PHI_MESH)).ravel()  # x-coord on cartesian grid
        ))
        self.POLAR_SHAPE = PHI_MESH.shape

        # Initialisation du détecteur de chute
        self.fall_detector = FallDetector(fall_threshold_frames=20)
        self.last_fps = 20.0 # Valeur par défaut pour le seuil initial

        # ARDUINO OPTIONNEL
        self.arduino = None
        try:
            # Remplace 'COM3' par ton port (ex: '/dev/ttyACM0' sur Linux)
            self.arduino = serial.Serial('/dev/tty.usbmodem1401', 9600, timeout=0.1)
            print("✅ Arduino détecté et connecté.")
        except Exception as e:
            print(f"⚠️ Arduino non détecté : {e}. Mode sans LED activé.")

        # Dans le __init__
        self.smooth_heatmap = None
        self.alpha = 0.5  # Facteur de lissage (0.1 = très lent/stable, 0.9 = très nerveux)

    def _get_latest_from_queues(self):
        """
        Drains all messages from input queues to ensure we only process the 
        most recent frame (avoids lag accumulation).
        """
        for i, q in enumerate([self.q1, self.q2]):

            try:
                while not q.empty():
                    msg = q.get_nowait()
                    if msg[0] == 'bev':
                        self.latest_msg[i] = msg[1]
                        self.msg_ready[i] = True

            except:
                pass
        return any(self.msg_ready)

    # Data processing is performed here
    def process(self, task: Task) -> int:
        """
        Main processing loop called by the task manager.
        Performs fusion, clutter removal, and tracking.
        """
        # 1. Update data from queues
        has_new_data = self._get_latest_from_queues()

        # 2. Process only if we have a frame from both radars
        if has_new_data and all(self.msg_ready):
            bf_1, bf_2 = self.latest_msg[0], self.latest_msg[1]
            
            # --- FUSION ENGINE ---
            # Instantiate interpolators (Note: Moving to map_coordinates would be even faster)
            interp1 = RegularGridInterpolator((self.phi, self.r_idxs), bf_1, bounds_error=False, fill_value=0)
            interp2 = RegularGridInterpolator((self.phi, self.r_idxs), bf_2, bounds_error=False, fill_value=0)

            # Map both radars to the same Cartesian space and fuse using Maximum Intensity Projection
            Z_cart = np.maximum(interp1(self.pts1), interp2(self.pts2)).reshape(self.X.shape)

            # --- PERSISTENCE (Lissage temporel) ---
            if self.smooth_heatmap is None:
                self.smooth_heatmap = Z_cart
            else:
                # On mélange l'ancienne et la nouvelle frame
                self.smooth_heatmap = (self.alpha * Z_cart) + ((1 - self.alpha) * self.smooth_heatmap)

            # On continue le process avec la version lissée
            interp_fused = RegularGridInterpolator((self.y_grid, self.x_grid), self.smooth_heatmap, bounds_error=False, fill_value=0)
       
            Z_polar = np.flip(interp_fused(self.pts_back).reshape(self.POLAR_SHAPE), axis=0)

            # Normalize 
            to_plot = np.abs(Z_polar)
            norm_factor = np.max(to_plot)
            if norm_factor > 0:
                to_plot /= norm_factor

            # --- BACKGROUND SUBTRACTION ---

            if self.do_bg_removal:
                if len(self.clutter_frames) < self.CLUTTER_LEARN_LIMIT:
                    self.clutter_frames.append(to_plot.copy())
                    self.clutter_map = np.mean(self.clutter_frames, axis=0)
                elif self.clutter_map is not None:
                    to_plot = np.clip(to_plot - self.clutter_map, 0, None)

            # Sharpen the heatmap for point detection
            to_plot = to_plot ** 8

            # --- GTRACKING ---
            # Only generate detections for points above the SNR threshold
            indices = np.argwhere(to_plot >= self.snr_threshold)
            
            # Optimization: limit detections to avoid saturating the tracker
            detections = [
                Detection(r=self.r_idxs[i], az=self.phi[j], v=0, snr=to_plot[j, i])
                for j, i in indices[:200] # Caps at 200 points
            ]
            
            gtrack_output = self.tracker.step(detections)
            tracks = gtrack_output.get('tracks', [])

            # --- LOGIQUE DE DETECTION DE CHUTE ---
            # 1. Mise à jour des positions pour le détecteur
            for t in tracks:
                self.fall_detector.last_positions[t['uid']] = (t['pos'][0], t['pos'][1])

            # 2. Détection
            active_ids = {t['uid'] for t in tracks}
            fall_events = self.fall_detector.update(active_ids)


            # --- LOGIQUE LED ARDUINO ---
            if self.arduino:
                try:
                    # Gestion Tracking (LED Verte/Standard)
                    self.arduino.write(b'1' if tracks else b'0')
                    
                    # Gestion Chute (LED Rouge)
                    # On allume si des événements de chute sont détectés dans cette frame
                    if fall_events:
                        self.arduino.write(b'F')
                    else:
                        # Optionnel : tu peux décider de laisser la LED allumée 
                        # jusqu'à ce qu'un bouton soit pressé, ou l'éteindre si aucune chute n'est active
                        self.arduino.write(b'N')
                except:
                    self.arduino = None
                    print("❌ Connexion Arduino perdue.")

            # --- CALCUL DES PROFILS ---
            # Puissance vs Distance (déjà présent)
            range_profile = np.max(to_plot, axis=0)

            # Puissance vs Angle (Nouveau)
            # On prend le max sur l'axe des distances (axis 1) pour chaque angle
            azimuth_profile = np.max(to_plot, axis=1)

            # --- DATA OUTPUT ---
            # Send this to the visualizer
            try:
                if not self.q_out.full():
                    self.q_out.put_nowait({
                        "heatmap": to_plot,
                        "range_profile": range_profile, # Nouvelle donnée
                        "azimuth_profile": azimuth_profile,
                        "tracks": tracks,
                        "fall_events": fall_events, # On envoie les nouveaux événements
                        "all_falls": self.fall_detector.fall_events, # Historique complet
                        "learning_left": max(0, self.CLUTTER_LEARN_LIMIT - len(self.clutter_frames))
                    })
            except:
                pass # Queue full, skip frame to maintain real-time
            
            # Reset readiness for next sync point
            self.msg_ready = [False, False]

        return task.cont