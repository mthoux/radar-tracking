import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
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

        # ---------- POSITION SMOOTHING ----------
        self.smooth_positions = {}  # uid → smoothed (x, y)
        self.pos_alpha = 0.2        # 0.1=very smooth/laggy, 0.5=reactive

        # ---------- FUSION DEFINITIONS ----------
        
        # Geometric Offsets (converted to bins)
        self.x1 = +cfg_radar["D_x"] / 2 / self.range_res
        self.x2 = -cfg_radar["D_x"] / 2 / self.range_res
        self.angle_1, self.angle_2 = cfg_radar["angle_1"], cfg_radar["angle_2"]

        # Define Cartesian Grid for Fusion
        self.x_grid = np.arange(-cfg_radar["width"], cfg_radar["width"], 1)
        self.y_grid = self.r_idxs
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid, indexing='xy')

        # PRE-COMPUTATION: Mapping Polar coordinates to Cartesian points once.
        # This prevents costly trigonometric calculations inside the processing loop.
        
        # Radar 1 Mapping
        phi1 = np.arctan2((self.X - self.x1).ravel(), self.Y.ravel()) - self.angle_1
        r1   = np.hypot(self.X.ravel() - self.x1, self.Y.ravel())
        self.pts1 = np.column_stack((phi1, r1))

        # Radar 2 Mapping
        phi2 = np.arctan2((self.X - self.x2).ravel(), self.Y.ravel()) - self.angle_2
        r2   = np.hypot(self.X.ravel() - self.x2, self.Y.ravel())
        self.pts2 = np.column_stack((phi2, r2))

        # Back-sampling Mapping (Cartesian -> Polar display)
        PHI_MESH, R_MESH = np.meshgrid(self.phi, self.r_idxs, indexing='ij')
        self.pts_back = np.column_stack((
            (R_MESH * np.cos(PHI_MESH)).ravel(),  # y-coord (depth)   = R * cos(phi)
            (R_MESH * np.sin(PHI_MESH)).ravel()   # x-coord (lateral) = R * sin(phi)
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

        # Fall detector
        self.fall_detector = FallDetector(fall_threshold_frames=20)
        self.last_fps = 20.0

        self.disappear_counter = {}
        self.DISAPPEAR_LIMIT = 5

        # ARDUINO (optional)
        self.arduino = None
        try:
            self.arduino = serial.Serial(cfg_arduino["port"], 9600, timeout=0.1)
            print("✅ Arduino detected and connected.")
        except Exception as e:
            if cfg_arduino["warning"]:
                print(f"⚠️ Arduino not detected : {e}. Streaming without physical response.")

        # Temporal smoothing (EMA on Z_cart)
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

            # --- NORMALIZE RAW RADAR DATA ---
            # Normalize each radar independently before fusion so the
            # SNR-weighted average operates on a consistent 0→1 scale
            bf_1 = bf_1 / (np.max(bf_1) + 1e-9)
            bf_2 = bf_2 / (np.max(bf_2) + 1e-9)
            
            # --- FUSION ENGINE ---
            interp1 = RegularGridInterpolator((self.phi, self.r_idxs), bf_1, bounds_error=False, fill_value=0)
            interp2 = RegularGridInterpolator((self.phi, self.r_idxs), bf_2, bounds_error=False, fill_value=0)

            v1 = interp1(self.pts1)
            v2 = interp2(self.pts2)

            # SNR-weighted average fusion:
            # - Strong signal in both radars → both contribute equally → real target ✅
            # - Strong signal in R1, weak sidelobe in R2 → R1 dominates → sidelobe suppressed ✅
            # - Two real people → each person strong in both radars → stay separated ✅
            w1 = np.clip(v1, 0, None)
            w2 = np.clip(v2, 0, None)
            total_w = w1 + w2 + 1e-9  # avoid division by zero
            Z_cart = ((w1 * v1 + w2 * v2) / total_w).reshape(self.X.shape)

            # --- SPATIAL BLUR ---
            # Merges any residual offset between the two radar blobs
            # sigma=1.5 bins ≈ 6.6cm — merges nearby duplicates, preserves person separation
            Z_cart = gaussian_filter(Z_cart, sigma=1.5)

            # --- TEMPORAL SMOOTHING (EMA) ---
            if self.do_smoothing:
                if self.last_Z is None:
                    self.last_Z = Z_cart
                Z_cart = (self.alpha * Z_cart) + ((1 - self.alpha) * self.last_Z)
                self.last_Z = Z_cart

            # --- BACK-SAMPLE TO POLAR FOR DISPLAY ---
            interp_fused = RegularGridInterpolator((self.y_grid, self.x_grid), Z_cart, bounds_error=False, fill_value=0)
            Z_polar = interp_fused(self.pts_back).reshape(self.POLAR_SHAPE)

            to_plot = np.abs(Z_polar)

            # --- FIRST NORMALIZATION (before clutter learning) ---
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
                        print("✅ Background subtraction completed.")
                elif self.clutter_map is not None:
                    to_plot = np.clip(to_plot - self.clutter_map, 0, None)
                    # Only renormalize if meaningful signal remains.
                    # Threshold prevents noise amplification when room is empty.
                    norm_factor2 = np.max(to_plot)
                    if norm_factor2 > 0.6:
                        to_plot /= norm_factor2
                    else:
                        to_plot[:] = 0.0

            # --- SHARPENING ---
            # Increases contrast between peaks and background.
            # **8 creates sharp distinct blobs — important for multi-person separation.
            to_plot = to_plot ** 8

            # --- GTRACKING ---
            if is_learning:
                tracks = []
            else:
                # Find all points above SNR threshold
                indices = np.argwhere(to_plot >= self.snr_threshold)

                # Sort by SNR descending — give GTrack strongest detections first.
                # This ensures track seeding happens on real targets before noise.
                if len(indices) > 0:
                    snr_values = to_plot[indices[:, 0], indices[:, 1]]
                    sorted_idx = np.argsort(snr_values)[::-1]
                    indices = indices[sorted_idx]

                detections = [
                    Detection(r=self.r_idxs[i] * self.range_res, az=self.phi[j], v=0, snr=to_plot[j, i])
                    for j, i in indices[:200]
                ]

                gtrack_output = self.tracker.step(detections)
                tracks = gtrack_output.get('tracks', [])

                # --- POSITION SMOOTHING ---
                # Decoupled from Kalman filter — smooths display jitter without
                # widening the Kalman gate (which would cause track stealing).
                for t in tracks:
                    uid = t['uid']
                    x, y = t['pos']
                    if uid not in self.smooth_positions:
                        self.smooth_positions[uid] = (x, y)
                    else:
                        sx, sy = self.smooth_positions[uid]
                        self.smooth_positions[uid] = (
                            self.pos_alpha * x + (1 - self.pos_alpha) * sx,
                            self.pos_alpha * y + (1 - self.pos_alpha) * sy
                        )
                    t['pos'] = np.array(self.smooth_positions[uid])

                # Clean up positions for tracks that no longer exist
                active_uids = {t['uid'] for t in tracks}
                for uid in list(self.smooth_positions.keys()):
                    if uid not in active_uids:
                        del self.smooth_positions[uid]

            # --- FALL DETECTION ---
            if is_learning:
                fall_events = []
            else:
                active_ids = {t['uid'] for t in tracks}
                
                for t in tracks:
                    uid = t['uid']
                    # Uses smoothed position (already updated above)
                    self.fall_detector.last_positions[uid] = (t['pos'][0], t['pos'][1])
                
                fall_events = self.fall_detector.update(active_ids)

            # --- ARDUINO LED ---
            if self.arduino:
                try:
                    self.arduino.write(b'1' if tracks else b'0')
                    if fall_events:
                        self.arduino.write(b'F')
                    else:
                        self.arduino.write(b'N')
                except:
                    self.arduino = None
                    print("❌ Lost connection with Arduino.")

            # --- PROFILES ---
            range_profile   = np.max(to_plot, axis=0)
            azimuth_profile = np.max(to_plot, axis=1)

            # --- OUTPUT ---
            try:
                if not self.q_out.full():
                    self.q_out.put_nowait({
                        "heatmap": to_plot,
                        "tracks": tracks,
                        "fall_events": fall_events,
                        "all_falls": self.fall_detector.fall_events,
                        "learning_left": self.CLUTTER_LEARN_LIMIT - len(self.clutter_frames),
                        "profile": {
                            "range": range_profile,
                            "azimuth": azimuth_profile
                        }
                    })
            except:
                pass  # Queue full — skip frame to maintain real-time

            # Reset readiness for next sync point
            self.msg_ready = [False, False]

        return task.cont