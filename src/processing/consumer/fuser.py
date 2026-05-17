import warnings
import numpy as np
import serial
from scipy.ndimage import map_coordinates
from direct.task import Task

from src.processing.consumer.gtrack.config import Detection
# /!\ Attention : GTrackModule2D devra probablement être remplacé par une version 3D (GTrackModule3D)
from src.processing.consumer.gtrack.module import GTrackModule2D 
from .fall_detection import FallDetector

warnings.simplefilter("ignore", UserWarning)

class Fuser:
    def __init__(self, queue_1, queue_2, queue_out, cfg_radar, cfg_gtrack):
        self.q1, self.q2, self.q_out = queue_1, queue_2, queue_out
        
        # --- CONFIGURATION RADAR ---
        self.phi = cfg_radar["phi"]          # Vecteur Azimut (ex: -60 à +60°)
        self.theta = cfg_radar["theta"]      # Vecteur Élévation (ex: -40 à +40°) -> NOUVEAU
        self.r_idxs = cfg_radar["range_idx"]  # Vecteur Distance
        self.snr_threshold = cfg_gtrack.min_snr_threshold
        self.alpha = 0.7
        
        # Grille Cartésienne 3D (X, Y, Z)
        self.x_grid = np.arange(-cfg_radar["width"], cfg_radar["width"], 1)
        self.y_grid = self.r_idxs
        self.z_grid = np.arange(-cfg_radar["height"], cfg_radar["height"], 1) # NOUVEAU
        
        # Génération de la grille 3D
        self.X, self.Y, self.Z = np.meshgrid(self.x_grid, self.y_grid, self.z_grid, indexing='xy')

        # --- States ---
        self.latest_msg = {0: None, 1: None}
        self.msg_ready = [False, False]
        self.smooth_heatmap_3d = None

        # --- BG REMOVAL ---
        self.clutter_frames, self.clutter_map = [], None
        self.CLUTTER_LEARN_LIMIT = cfg_radar["bgrm_learning_frames"]
        self.do_bg_removal = cfg_radar.get("do_bg_removal", True)

        # --- INITIALISATION MODULES ---
        # Note : Pensez à passer le tracker en 3D si votre modèle le supporte
        self.tracker = GTrackModule2D(cfg_gtrack) 
        self.fall_detector = FallDetector(fall_threshold_frames=20)
        self._init_arduino()

        # --- PRE-COMPUTATION GÉOMÉTRIQUE 3D ---
        self._setup_mappings_3d(cfg_radar)

    def _init_arduino(self):
        self.arduino = None
        try:
            self.arduino = serial.Serial('/dev/tty.usbmodem1401', 9600, timeout=0.1)
            print("✅ Arduino connecté.")
        except Exception as e:
            print(f"⚠️ Mode sans LED : {e}")

    def _setup_mappings_3d(self, cfg):
        """Prépare les index 3D pour map_coordinates (Sphérique -> Cartésien)"""
        
        def compute_indices(offset_x, offset_y, angle_radar):
            # Décalage de la grille par rapport au radar
            dx = self.X.ravel() - offset_x
            dy = self.Y.ravel() - offset_y
            dz = self.Z.ravel()  # On suppose les radars à la même hauteur Z=0, sinon ajoutez offset_z
            
            # Calcul de la distance 3D
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Azimut avec rotation du radar
            p = np.arctan2(dy, dx) - angle_radar
            
            # Élévation
            t = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
            
            # Conversion en index flottants pour map_coordinates
            # L'ordre dépend du shape de votre matrice brute radar. 
            # Supposons ici que la matrice brute a la forme (Ancrage_Theta, Ancrage_Phi, Range)
            idx_t = (t - self.theta[0]) / (self.theta[1] - self.theta[0])
            idx_p = (p - self.phi[0]) / (self.phi[1] - self.phi[0])
            idx_r = (r - self.r_idxs[0]) / (self.r_idxs[1] - self.r_idxs[0])
            
            return np.vstack((idx_t, idx_p, idx_r))

        self.map_idx1 = compute_indices(cfg["offset_x_1"], cfg["offset_y_1"], cfg["angle_1"])
        self.map_idx2 = compute_indices(cfg["offset_x_2"], cfg["offset_y_2"], cfg["angle_2"])

        # Back-sampling (Cartésien 3D -> Sphérique Radar pour l'affichage classique si besoin)
        # Pour simplifier et rester compatible avec votre affichage 2D, on va projeter la grille 
        # Sphérique 2D (Phi, R) sur le plan Z=0 de notre cube Cartésien fusionné.
        PHI_M, R_M = np.meshgrid(self.phi, self.r_idxs, indexing='ij')
        self.POLAR_SHAPE = PHI_M.shape
        y_back = (R_M * np.sin(PHI_M)).ravel()
        x_back = (R_M * np.cos(PHI_M)).ravel()
        z_back = np.zeros_like(x_back) # Projection sur le plan horizontal médian
        
        idx_y = (y_back - self.y_grid[0]) / (self.y_grid[1] - self.y_grid[0])
        idx_x = (x_back - self.x_grid[0]) / (self.x_grid[1] - self.x_grid[0])
        idx_z = (z_back - self.z_grid[0]) / (self.z_grid[1] - self.z_grid[0])
        
        # map_coordinates prendra (Y, X, Z) correspondant à la shape de self.X (Meshgrid 'xy' donne une shape (y, x, z))
        self.map_back = np.vstack((idx_y, idx_x, idx_z))

    def _get_latest_data(self):
        for i, q in enumerate([self.q1, self.q2]):
            while not q.empty():
                msg = q.get_nowait()
                self.latest_msg[i], self.msg_ready[i] = msg, True
        return all(self.msg_ready)

    def process(self, task: Task):
        if not self._get_latest_data():
            return task.cont
        
        # Les matrices d'entrées latest_msg[0] et [1] sont maintenant supposées 3D 
        # Shape attendue : (len(theta), len(phi), len(range))
        data_3D_1 = self.latest_msg[0]
        data_3D_2 = self.latest_msg[1]

        # 1. Fusion Cartésienne 3D
        # map_coordinates reconstruit le cube Cartésien (Shape de self.X -> (len(y), len(x), len(z)))
        v1 = map_coordinates(data_3D_1, self.map_idx1, order=1).reshape(self.X.shape)
        v2 = map_coordinates(data_3D_2, self.map_idx2, order=1).reshape(self.X.shape)
        Z_cart_3d = np.maximum(v1, v2)

        # 2. Lissage Temporel 3D
        if self.smooth_heatmap_3d is None: 
            self.smooth_heatmap_3d = Z_cart_3d
        else: 
            self.smooth_heatmap_3d = (self.alpha * Z_cart_3d) + ((1 - self.alpha) * self.smooth_heatmap_3d)

        # 3. Projection pour affichage (On extrait une coupe 2D du cube 3D lissé pour votre affichage Polar)
        Z_pol = map_coordinates(self.smooth_heatmap_3d, self.map_back, order=1).reshape(self.POLAR_SHAPE)
        to_plot = np.abs(np.flip(Z_pol, axis=0))
        
        # Normalisation & Background (Inchangé, s'applique sur l'image 2D finale)
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
        # (Ici le tracker reste en 2D basé sur la projection polaire plane pour ne pas casser votre FallDetector)
        indices = np.argwhere(to_plot >= self.snr_threshold)
        if len(indices) > 200:
            snr_vals = to_plot[indices[:, 0], indices[:, 1]]
            top_idx = np.argsort(snr_vals)[::-1][:200]
            indices = indices[top_idx]

        detections = [Detection(r=self.r_idxs[i], az=self.phi[j], v=0, snr=to_plot[j,i]) for j,i in indices]
        
        tracks = self.tracker.step(detections).get('tracks', [])
        for t in tracks: self.fall_detector.last_positions[t['uid']] = (t['pos'][0], t['pos'][1])
        fall_events = self.fall_detector.update({t['uid'] for t in tracks})

        # 5. Extraction de la carte d'élévation pour la sortie (Utile pour voir la hauteur !)
        # On fait un max sur l'axe X et Y du cube cartésien, ou directement depuis le Radar 1.
        # Ici, extraction de l'élévation sur le cube fusionné (axe 0 = Y, axe 1 = X, axe 2 = Z)
        elevation_profile_3d = np.max(self.smooth_heatmap_3d, axis=(0, 1)) 

        # Sortie & Arduino
        self._update_arduino(tracks, fall_events)

        if not self.q_out.full():
            learning_left = max(0, self.CLUTTER_LEARN_LIMIT - len(self.clutter_frames))
            
            self.q_out.put_nowait({
                "heatmap": to_plot,
                "range_profile": np.max(to_plot, axis=0),
                "azimuth_profile": np.max(to_plot, axis=1),
                "tracks": tracks,
                "fall_events": fall_events,
                "learning_left": learning_left,
                "elevation_profile": elevation_profile_3d  # Profil 3D mis à jour
            })

        self.msg_ready = [False, False]
        return task.cont

    def _update_arduino(self, tracks, falls):
        if not self.arduino: return
        try:
            self.arduino.write(b'1' if tracks else b'0')
            self.arduino.write(b'F' if falls else b'N')
        except: self.arduino = None