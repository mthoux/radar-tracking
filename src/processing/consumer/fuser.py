import warnings
import numpy as np
import serial
from scipy.ndimage import map_coordinates
from direct.task import Task

from src.processing.consumer.gtrack.config import Detection
from src.processing.consumer.gtrack.module import GTrackModule2D
from .fall_detection import FallDetector

warnings.simplefilter("ignore", UserWarning)

class Fuser:
    def __init__(self, queue_1, queue_2, queue_out, cfg_radar, cfg_gtrack):
        self.q1, self.q2, self.q_out = queue_1, queue_2, queue_out
        
        # --- CONFIGURATION RADAR ---
        self.phi = cfg_radar["phi"]
        self.r_idxs = cfg_radar["range_idx"]
        self.snr_threshold = cfg_gtrack.min_snr_threshold
        self.alpha = 0.7
        
        # Grille Cartésienne
        self.x_grid = np.arange(-cfg_radar["width"], cfg_radar["width"], 1)
        self.y_grid = self.r_idxs
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid, indexing='xy')

        # --- ÉTATS ---
        self.latest_msg = {0: None, 1: None}
        self.msg_ready = [False, False]
        self.smooth_heatmap = None
        self.clutter_frames, self.clutter_map = [], None
        self.CLUTTER_LEARN_LIMIT = 50
        self.do_bg_removal = cfg_radar.get("do_bg_removal", True)

        # --- INITIALISATION MODULES ---
        self.tracker = GTrackModule2D(cfg_gtrack)
        self.fall_detector = FallDetector(fall_threshold_frames=20)
        self._init_arduino()

        # --- PRE-COMPUTATION GÉOMÉTRIQUE ---
        self._setup_mappings(cfg_radar)

    def _init_arduino(self):
        self.arduino = None
        try:
            self.arduino = serial.Serial('/dev/tty.usbmodem1401', 9600, timeout=0.1)
            print("✅ Arduino connecté.")
        except Exception as e:
            print(f"⚠️ Mode sans LED : {e}")

    def _setup_mappings(self, cfg):
        """Prépare les index pour map_coordinates une fois pour toutes."""
        # Mapping Radar 1
        p1 = np.arctan2((self.Y - cfg["offset_y_1"]).ravel(), (self.X - cfg["offset_x_1"]).ravel()) - cfg["angle_1"]
        r1 = np.hypot(self.X.ravel() - cfg["offset_x_1"], self.Y.ravel() - cfg["offset_y_1"])
        
        # Mapping Radar 2
        p2 = np.arctan2((self.Y - cfg["offset_y_2"]).ravel(), (self.X - cfg["offset_x_2"]).ravel()) - cfg["angle_2"]
        r2 = np.hypot(self.X.ravel() - cfg["offset_x_2"], self.Y.ravel() - cfg["offset_y_2"])

        # Conversion en index flottants pour map_coordinates
        def to_idx(p, r):
            idx_p = (p - self.phi[0]) / (self.phi[1] - self.phi[0])
            idx_r = (r - self.r_idxs[0]) / (self.r_idxs[1] - self.r_idxs[0])
            return np.vstack((idx_p, idx_r))

        self.map_idx1 = to_idx(p1, r1)
        self.map_idx2 = to_idx(p2, r2)

        # Back-sampling (Cartésien -> Polaire pour affichage)
        PHI_M, R_M = np.meshgrid(self.phi, self.r_idxs, indexing='ij')
        self.POLAR_SHAPE = PHI_M.shape
        y_back = (R_M * np.sin(PHI_M)).ravel()
        x_back = (R_M * np.cos(PHI_M)).ravel()
        
        idx_y = (y_back - self.y_grid[0]) / (self.y_grid[1] - self.y_grid[0])
        idx_x = (x_back - self.x_grid[0]) / (self.x_grid[1] - self.x_grid[0])
        self.map_back = np.vstack((idx_y, idx_x))

    def _get_latest_data(self):
        for i, q in enumerate([self.q1, self.q2]):
            while not q.empty():
                msg = q.get_nowait()
                if msg[0] == 'bev':
                    self.latest_msg[i], self.msg_ready[i] = msg[1], True
        return all(self.msg_ready)

    def process(self, task: Task):
        if not self._get_latest_data():
            return task.cont

        # 1. Fusion Cartésienne
        v1 = map_coordinates(self.latest_msg[0], self.map_idx1, order=1).reshape(self.X.shape)
        v2 = map_coordinates(self.latest_msg[1], self.map_idx2, order=1).reshape(self.X.shape)
        Z_cart = np.maximum(v1, v2)

        # 2. Lissage Temporel
        if self.smooth_heatmap is None: self.smooth_heatmap = Z_cart
        else: self.smooth_heatmap = (self.alpha * Z_cart) + ((1 - self.alpha) * self.smooth_heatmap)

        # 3. Projection pour affichage
        Z_pol = map_coordinates(self.smooth_heatmap, self.map_back, order=1).reshape(self.POLAR_SHAPE)
        to_plot = np.abs(np.flip(Z_pol, axis=0))
        
        # Normalisation & Background
        max_val = np.max(to_plot)
        if max_val > 0: to_plot /= max_val
        
        if self.do_bg_removal:
            if len(self.clutter_frames) < self.CLUTTER_LEARN_LIMIT:
                self.clutter_frames.append(to_plot.copy())
                self.clutter_map = np.mean(self.clutter_frames, axis=0)
            elif self.clutter_map is not None:
                to_plot = np.clip(to_plot - self.clutter_map, 0, None)

        to_plot = to_plot ** 8

        # 4. Tracking & Fall Detection
        indices = np.argwhere(to_plot >= self.snr_threshold)
        if len(indices) > 200:
            snr_vals = to_plot[indices[:, 0], indices[:, 1]]
            top_idx = np.argsort(snr_vals)[::-1][:200]
            indices = indices[top_idx]

        detections = [Detection(r=self.r_idxs[i], az=self.phi[j], v=0, snr=to_plot[j,i]) for j,i in indices]
        
        tracks = self.tracker.step(detections).get('tracks', [])
        for t in tracks: self.fall_detector.last_positions[t['uid']] = (t['pos'][0], t['pos'][1])
        fall_events = self.fall_detector.update({t['uid'] for t in tracks})

        # 5. Sortie & Arduino
        self._update_arduino(tracks, fall_events)
        self._send_to_visualizer(to_plot, tracks, fall_events)

        self.msg_ready = [False, False]
        return task.cont

    def _update_arduino(self, tracks, falls):
        if not self.arduino: return
        try:
            self.arduino.write(b'1' if tracks else b'0')
            self.arduino.write(b'F' if falls else b'N')
        except: self.arduino = None

    def _send_to_visualizer(self, heatmap, tracks, falls):
        if not self.q_out.full():
            # On recalcule combien de frames il reste à apprendre
            learning_left = max(0, self.CLUTTER_LEARN_LIMIT - len(self.clutter_frames))
            
            self.q_out.put_nowait({
                "heatmap": heatmap,
                "range_profile": np.max(heatmap, axis=0),
                "azimuth_profile": np.max(heatmap, axis=1),
                "tracks": tracks,
                "fall_events": falls,
                "learning_left": learning_left  # <--- Ajoute cette ligne !
            })