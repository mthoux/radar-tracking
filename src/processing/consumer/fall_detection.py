import numpy as np
import time
from collections import deque
from typing import Dict, Set, List, Tuple, Optional

# Number of frames to look back just before disappearance (and the minimum too)
RECENT_WINDOW_FOR_SPEED = 10
# Minimum variance of the centroid history to consider speed reliable.
MIN_CENTROID_VARIANCE = 0.05  # m²

class FallDetector:
    """
    Détecte les chutes en surveillant la disparition prolongée de tracks.
    
    Une track est considérée "chutée" si elle n'apparaît plus dans
    gtrack_output pendant au moins `fall_threshold_frames` frames consécutives.
    Le délai de grâce (grace_frames) absorbe les occultations courtes ou
    les pertes de détection temporaires dues au bruit radar.
    """

    def __init__(
        self, 
        fall_threshold_frames=20, 
        vertical_speed_threshold: float = 0.03,    # tune: ~3cm/frame downward
        elevation_heights: Tuple[float, ...] = (0.0, 0.7, 1.4),
        centroid_history_len: int = 30,
        valid_zone=(-25, 25, 5, 95)
    ):
        self.fall_threshold = fall_threshold_frames
        self.vert_speed_thresh       = vertical_speed_threshold
        self.heights                 = np.array(elevation_heights, dtype=float)  # (3,)
        self.centroid_history_len    = centroid_history_len
        self.valid_zone = valid_zone  # (x_min, x_max, y_min, y_max)

        # {track_id: int}  consecutive-miss counter
        self.miss_counter: Dict[int, int] = {}
 
        # {track_id: deque of float}  height-centroid history (m)
        self.centroid_history: Dict[int, deque] = {}
 
        # {track_id: float}  peak downward speed seen for this track (m/frame)
        self.recent_downward_speed: Dict[int, float] = {}
 
        # IDs for which an alert was already emitted
        self.alerted_ids: Set[int] = set()
 
        # Full event log
        self.fall_events: List[dict] = []
 
        # Latest known 2-D position  {uid: (x, y)}
        self.last_positions: Dict[int, tuple] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def update_height_centroids(
        self,
        track_id: int,
        track_pos: Tuple[float, float],
        bev_levels: List[np.ndarray],
        phi: np.ndarray,
        r_idxs: np.ndarray,
        radius_m: float = 1.5,
    ) -> Optional[float]:
        """
        Compute the power-weighted height centroid for one track and store it.
 
        For each height level k, we integrate the BEV power inside a disk of
        ``radius_m`` metres around the track's (x, y) position.  The centroid
        is then:
            h_centroid = Σ_k  heights[k] * P_k  /  Σ_k P_k
 
        Parameters
        ----------
        track_id  : int
        track_pos : (x, y) in the same Cartesian frame as r_idxs / phi
        bev_levels: list of 3 normalised BEV arrays, each (num_phi, num_range)
        phi       : azimuth angles (rad), shape (num_phi,)
        r_idxs    : range indices, shape (num_range,)
        radius_m  : integration radius around the track position (index units)
 
        Returns
        -------
        centroid : float (metres) or None if no power found near the track
        """
        tx, ty = track_pos
 
        # Build Cartesian coordinate grids once per call
        # BEV axes: rows = phi, cols = r_idxs
        PHI, R = np.meshgrid(phi, r_idxs, indexing='ij')     # (num_phi, num_range)
        X_grid = R * np.sin(PHI)   # azimuth → x
        Y_grid = R * np.cos(PHI)   # range   → y
 
        # Mask: pixels within radius_m of the track
        dist2 = (X_grid - tx)**2 + (Y_grid - ty)**2
        mask  = dist2 <= radius_m**2  # (num_phi, num_range) bool
 
        if not mask.any():
            return None
 
        level_powers = np.array([
            float(np.sum(bev[mask])) for bev in bev_levels
        ])                                                      # (3,)
 
        total_power = level_powers.sum()
        if total_power < 1e-12:
            return None
 
        centroid = float(np.dot(self.heights, level_powers) / total_power)
 
        # Store in history
        if track_id not in self.centroid_history:
            self.centroid_history[track_id] = deque(maxlen=self.centroid_history_len)
        self.centroid_history[track_id].append(centroid)
 
        # Update peak downward speed (by only looking at the most recent window)
        hist = self.centroid_history[track_id]
        recent = list(hist)[-RECENT_WINDOW_FOR_SPEED:]
        if len(recent) >= 3:
            # Finite-difference estimate: positive = downward (height decreasing)
            speeds = [recent[i-1] - recent[i] for i in range(1, len(recent))]
            self.recent_downward_speed[track_id] = max(speeds)
 
        return centroid

    def update(self, active_track_ids: set[int]) -> list[dict]:
        """
        À appeler à chaque frame avec l'ensemble des IDs de tracks actives.

        Retourne la liste des nouvelles chutes détectées cette frame
        (liste vide si aucune).
        """
        new_falls = []

        # Incrémenter le compteur des tracks manquantes
        missing = set(self.miss_counter.keys()) - active_track_ids
        for tid in missing:
            self.miss_counter[tid] += 1

        # Remettre à zéro les tracks qui sont revenues
        for tid in active_track_ids:
            self.miss_counter[tid] = 0
            # Clear alert so a re-fall can be detected after recovery
            if tid in self.alerted_ids and self.miss_counter.get(tid, 0) == 0:
                self.alerted_ids.discard(tid)

        # Nettoyer les tracks vraiment disparues (> seuil) et alerter
        x_min, x_max, y_min, y_max = self.valid_zone
        for tid, count in list(self.miss_counter.items()):
            print(self.peak_downward_speed.get(tid, 0.0))
            if count >= self.fall_threshold and tid not in self.alerted_ids:

                 # ── Boundary check ───────────────────────────────────────────
                pos = self.last_positions.get(tid)
                if pos is None:
                    continue
                x, y = pos
                if not (x_min <= x <= x_max and y_min <= y <= y_max):
                    # Track exited through boundary → not a fall
                    del self.miss_counter[tid]
                    continue
                
                # ── Vertical-speed gate ──────────────────────────────────────
                peak_speed = self.recent_downward_speed.get(tid, 0.0)
                history_len = len(self.centroid_history.get(tid, []))
                recent = list(self.centroid_history.get(tid, []))[-RECENT_WINDOW_FOR_SPEED:]
                centroid_variance = np.var(recent) if len(recent) >= RECENT_WINDOW_FOR_SPEED else 0.0

                # Confirm fall:
                # Not enough history → trust disappearance alone
                if history_len < RECENT_WINDOW_FOR_SPEED:
                    print(f"[FALL SKIPPED] track_id={tid} not enough history ({history_len} frames) → ghost detection")
                    del self.miss_counter[tid]
                    continue

                # Instant disappearance without enough time to compute vertical speed (very high speed)
                elif centroid_variance < MIN_CENTROID_VARIANCE:
                    print(f"[FALL WARNING] track_id={tid} low variance ({centroid_variance:.4f}) → instant fall")
                        
                # Track disappeared but did NOT show a fast downward motion
                elif peak_speed < self.vert_speed_thresh:
                    print(f"[FALL SKIPPED] track_id={tid} peak_speed={peak_speed:.4f} < threshold={self.vert_speed_thresh:.4f}")
                    del self.miss_counter[tid]
                    continue
                
                # Fall register
                event = {
                    "track_id":      tid,
                    "missing_frames": count,
                    "peak_downward_speed": peak_speed,
                    "timestamp":      time.time(),
                }
                new_falls.append(event)
                self.fall_events.append(event)
                self.alerted_ids.add(tid)
                print(f"[FALL DETECTED] track_id={tid} absent depuis {count} frames avec vert_speed de {peak_speed}")

            # Supprimer les tracks disparues depuis longtemps pour éviter la fuite mémoire
            if count > self.fall_threshold: #+ 30:
                self.miss_counter.pop(tid, None)
                self.centroid_history.pop(tid, None)
                self.peak_downward_speed.pop(tid, None)

        return new_falls