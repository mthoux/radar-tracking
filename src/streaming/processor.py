import sys
import warnings
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from direct.task import Task
from typing import Dict, Any, List

# Local imports
from gtrack.config import Detection
from gtrack.module import GTrackModule2D

# Global configuration to avoid COM initialization issues on some systems
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

class Processor:
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

    def _get_latest_from_queues(self):
        """
        Drains all messages from input queues to ensure we only process the 
        most recent frame (avoids lag accumulation).
        """
        for i, q in enumerate([self.q1, self.q2]):
            new_data = False
            try:
                while not q.empty():
                    msg = q.get_nowait()
                    if msg[0] == 'bev':
                        self.latest_msg[i] = msg[1]
                        self.msg_ready[i] = True
                        new_data = True
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
            
            # --- FUSION ENGINE ---
            # Instantiate interpolators (Note: Moving to map_coordinates would be even faster)
            interp1 = RegularGridInterpolator((self.phi, self.r_idxs), bf_1, bounds_error=False, fill_value=0)
            interp2 = RegularGridInterpolator((self.phi, self.r_idxs), bf_2, bounds_error=False, fill_value=0)

            # Map both radars to the same Cartesian space and fuse using Maximum Intensity Projection
            Z_cart = np.maximum(interp1(self.pts1), interp2(self.pts2)).reshape(self.X.shape)

            # Re-sample to Polar space for visualization
            interp_fused = RegularGridInterpolator((self.y_grid, self.x_grid), Z_cart, bounds_error=False, fill_value=0)
            Z_polar = np.flip(interp_fused(self.pts_back).reshape(self.POLAR_SHAPE), axis=0)

            # Normalize 
            to_plot = np.abs(Z_polar)
            norm_factor = np.max(to_plot)
            if norm_factor > 0:
                to_plot /= norm_factor

            # --- BACKGROUND SUBTRACTION ---
            if len(self.clutter_frames) < self.CLUTTER_LEARN_LIMIT:
                self.clutter_frames.append(to_plot.copy())
                self.clutter_map = np.mean(self.clutter_frames, axis=0)
            else:
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

            # --- DATA OUTPUT ---
            # Send results to the visualizer queue without blocking
            try:
                if not self.q_out.full():
                    self.q_out.put_nowait({
                        "heatmap": to_plot,
                        "tracks": gtrack_output.get('tracks', []),
                        "learning_left": max(0, self.CLUTTER_LEARN_LIMIT - len(self.clutter_frames))
                    })
            except:
                pass # Queue full, skip frame to maintain real-time
            
            # Reset readiness for next sync point
            self.msg_ready = [False, False]

        return task.cont