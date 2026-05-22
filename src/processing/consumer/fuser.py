import numpy as np
from scipy.interpolate import RegularGridInterpolator
from direct.task import Task
from typing import Dict, Any, List
import serial

# Local imports
from src.processing.consumer.gtrack.config import Detection
from src.processing.consumer.gtrack.module import GTrackModule2D
from .fall_detection import FallDetector

import sys
import warnings

# Suppress COM/User warnings before they trigger
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2  # Multithreading concurrency mode for COM

class Fuser:
    """
    Handles the fusion of multiple radar streams, background subtraction, 
    and target tracking using GTrack.
    """

    def __init__(self, queue_1: Any, queue_2: Any, queue_out: Any, cfg_radar: Dict[str, Any], cfg_gtrack: Any, cfg_arduino):
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
        self.range_res = cfg_radar["range_res"]
 
        # ---------- FUSION DEFINITIONS ----------
        
        # Geometric Offsets
        self.x1, self.x2 = +cfg_radar["D_x"]/2 * self.range_res,  -cfg_radar["D_x"]/2 * self.range_res # convert in bins
        self.angle_1, self.angle_2 = cfg_radar["angle_1"], cfg_radar["angle_2"]

        # Define Cartesian Grid for Fusion
        self.x_grid = np.arange(-cfg_radar["width"], cfg_radar["width"], 1)
        self.y_grid = self.r_idxs
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid, indexing='xy')

        # PRE-COMPUTATION: Mapping Polar coordinates to Cartesian points once.
        # This prevents costly trigonometric calculations inside the processing loop.
        
        # Radar 1 Mapping (Correction de l'ordre arctan2)
        phi1 = np.arctan2((self.X - self.x1).ravel(), self.Y.ravel()) - self.angle_1
        r1 = np.hypot(self.X.ravel() - self.x1, self.Y.ravel())
        self.pts1 = np.column_stack((phi1, r1))

        # Radar 2 Mapping (Correction de l'ordre arctan2)
        phi2 = np.arctan2((self.X - self.x2).ravel(), self.Y.ravel()) - self.angle_2
        r2 = np.hypot(self.X.ravel() - self.x2, self.Y.ravel())
        self.pts2 = np.column_stack((phi2, r2))

        # Back-sampling Mapping (Cartesian -> Polar display)
        PHI_MESH, R_MESH = np.meshgrid(self.phi, self.r_idxs, indexing='ij')
        self.pts_back = np.column_stack((
            (R_MESH * np.cos(PHI_MESH)).ravel(), # y-coord (profondeur) = R * cos(phi)
            (R_MESH * np.sin(PHI_MESH)).ravel()  # x-coord (largeur) = R * sin(phi)
        ))
        self.POLAR_SHAPE = PHI_MESH.shape

        # ----------------------------------------

        # GTrack Module Initialization
        self.tracker = GTrackModule2D(cfg_gtrack)
        
        # Background / Clutter Removal State
        self.do_bg_removal = cfg_radar["do_bg_removal"]
        self.CLUTTER_LEARN_LIMIT = 50
        self.clutter_frames: List[np.ndarray] = []
        self.clutter_map: np.ndarray = None

        # Initialisation du détecteur de chute
        self.fall_detector = FallDetector(fall_threshold_frames=20)
        self.last_fps = 20.0 # Valeur par défaut pour le seuil initial

        self.disappear_counter = {} 
        self.DISAPPEAR_LIMIT = 5 # Nombre de frames de tolérance

        # ARDUINO OPTIONNEL
        self.arduino = None
        try:
            self.arduino = serial.Serial(cfg_arduino["port"], 9600, timeout=0.1)
            print("✅ Arduino detected and connected.")
        except Exception as e:
            if cfg_arduino["warning"]:
                print(f"⚠️ Arduino not detected : {e}. Streaming without physical response.")

        # Smoothing
        self.do_smoothing = cfg_radar["smoothing"]
        self.alpha = cfg_radar["alpha_smoothing"]
        self.last_Z = None

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
            #Z_cart = (interp1(self.pts1) + interp2(self.pts2)).reshape(self.X.shape) / 2.0

            # --- PERSISTENCE (Lissage temporel) ---
            if self.do_smoothing:
                if self.last_Z is None: self.last_Z = Z_cart
                Z_cart = (self.alpha * Z_cart) + ((1 - self.alpha) * self.last_Z)
                self.last_Z = Z_cart

            # On continue le process avec la version lissée
            interp_fused = RegularGridInterpolator((self.y_grid, self.x_grid), Z_cart, bounds_error=False, fill_value=0)
       
            Z_polar = np.flip(interp_fused(self.pts_back).reshape(self.POLAR_SHAPE), axis=0)

            # Normalize 
            to_plot = np.abs(Z_polar)
            norm_factor = np.max(to_plot)
            if norm_factor > 0:
                to_plot /= norm_factor

           # --- BACKGROUND SUBTRACTION ---
            is_learning = len(self.clutter_frames) < self.CLUTTER_LEARN_LIMIT
            
            if self.do_bg_removal:
                if is_learning:
                    self.clutter_frames.append(to_plot.copy())
                    self.clutter_map = np.mean(self.clutter_frames, axis=0)
                    if len(self.clutter_frames) == self.CLUTTER_LEARN_LIMIT - 1:
                        print("Background subtraction completed")
                elif self.clutter_map is not None:
                    to_plot = np.clip(to_plot - self.clutter_map, 0, None)

            # Sharpen the heatmap for point detection
            to_plot = to_plot ** 8

            # --- GTRACKING ---
            if is_learning:
                tracks = []
            else:
                # 1. On trouve tous les points au-dessus du seuil
                indices = np.argwhere(to_plot >= self.snr_threshold)

                # if len(indices) > 0:
                #     # 2. On récupère les valeurs de SNR pour ces indices
                #     snr_values = to_plot[indices[:, 0], indices[:, 1]]
                    
                #     # 3. On trie par ordre décroissant (du plus grand SNR au plus petit)
                #     #sorted_idx = np.argsort(snr_values)[::-1]
                #     #indices = indices[sorted_idx]

                # 4. Optimization: On garde les 200 m   eilleurs pour éviter de saturer le tracker
                # detections = [
                #     Detection(r=self.r_idxs[i]*self.range_res, az=np.pi/2 - self.phi[j], v=0, snr=to_plot[j, i])
                #     for j, i in indices[:200]  # Ici, ce sont bien les 200 meilleurs !
                # ]
                
                phi_flipped = self.phi[::-1]  # match the flip applied to Z_polar

                detections = [
                    Detection(r=self.r_idxs[i]*self.range_res, az=phi_flipped[j], v=0, snr=to_plot[j, i])
                    for j, i in indices[:200]
                ]

                gtrack_output = self.tracker.step(detections)
                tracks = gtrack_output.get('tracks', [])

            # --- LOGIQUE DE DETECTION DE CHUTE (Dans Fuser.py) ---
            if is_learning:
                fall_events = []
            else:
                active_ids = {t['uid'] for t in tracks}
                
                # Mise à jour des positions
                for t in tracks:
                    uid = t['uid']
                    self.fall_detector.last_positions[uid] = (t['pos'][0], t['pos'][1])
                
                # C. Détection
                fall_events = self.fall_detector.update(active_ids)

            # --- LOGIQUE LED ARDUINO ---
            if self.arduino:
                try:
                    self.arduino.write(b'1' if tracks else b'0') # Write Green LED for tracking
                    if fall_events: self.arduino.write(b'F') # Write red LED for falls
                    else: self.arduino.write(b'N') # Shutdown red LED when no more falls
                except:
                    self.arduino = None
                    print("❌ Lost connection with Arduino.")

            # --- CALCUL DES PROFILS ---
            range_profile = np.max(to_plot, axis=0)
            azimuth_profile = np.max(to_plot, axis=1)

            # --- DATA OUTPUT ---
            # Send this to the visualizer
            try:
                if not self.q_out.full():
                    self.q_out.put_nowait({
                        "heatmap": to_plot,
                        "tracks": tracks,
                        "fall_events": fall_events, # On envoie les nouveaux événements
                        "all_falls": self.fall_detector.fall_events, # Historique complet
                        "learning_left": self.CLUTTER_LEARN_LIMIT - len(self.clutter_frames),
                        "profile": {
                            "range": range_profile,
                            "azimuth": azimuth_profile
                        }
                    })
            except:
                pass # Queue full, skip frame to maintain real-time
            
            # Reset readiness for next sync point
            self.msg_ready = [False, False]

        return task.cont