import numpy as np
import time
from collections import deque
from typing import Dict, Set, List, Tuple, Optional

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
        fall_threshold_frames=15, 
        valid_zone=(-25, 25, 5, 95),
        vz_threshold_m_s: float = -0.8,    # m/s — negative = downward
        vz_window_frames: int = 5,         # frames used for derivative
        require_vz: bool = True,           # set False → disappearance-only (original)
    ):
        self.fall_threshold  = fall_threshold_frames
        self.valid_zone      = valid_zone
        self.vz_threshold    = vz_threshold_m_s
        self.vz_window       = vz_window_frames
        self.require_vz      = require_vz

        # {track_id: int}  consecutive-miss counter
        self.miss_counter: Dict[int, int] = {}
 
        # IDs for which an alert was already emitted
        self.alerted_ids: Set[int] = set()
 
        # Full event log
        self.fall_events: List[dict] = []
 
        # Latest known 2-D position  {uid: (x, y)}
        self.last_positions: Dict[int, tuple] = {}
 
        # {uid: deque of (timestamp, height_m)}
        self._height_history: dict[int, deque] = {}
 
        # {uid: vz in m/s}
        self._avg_vz: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Called by Fuser for each active track
    # ------------------------------------------------------------------

    def update_elevation(self, uid: int, height_m: float, timestamp: float = None):
        """
        Record a height estimate for uid.  Differentiates over the sliding
        window to estimate vertical speed.
        """
        if timestamp is None:
            timestamp = time.time()
 
        if uid not in self._height_history:
            self._height_history[uid] = deque(maxlen=self.vz_window)
 
        self._height_history[uid].append((timestamp, height_m))
 
        hist = self._height_history[uid]
        n = len(hist)

        if n >= 2:
            # ------ Regression linéaire ------------
            hist = list(self._height_history[uid])
            times   = np.array([t for t, h in hist])
            heights = np.array([h for t, h in hist])

            # Centrer les temps pour la stabilité numérique
            times -= times[0]

            vz = np.polyfit(times, heights, deg=1)[0]
            self._avg_vz[uid] = float(vz)
            print(f"[SPEED] uid={uid} avg_vz={self._avg_vz[uid]:.3f} m/s")
            # ------ FIN ----------------------------

            # Vitesses instantanées entre chaque paire consécutive
            # vz_values = []
            # for i in range(1, n):
            #     t0, h0 = hist[i - 1]
            #     t1, h1 = hist[i]
            #     dt = t1 - t0
            #     vz_values.append((h1 - h0) / dt)

            # if vz_values:
            #     # Recompute weights for actual number of valid intervals
            #     weights = np.arange(1, len(vz_values) + 1, dtype=float) ** 2
            #     weights /= weights.sum()
            #     self._avg_vz[uid] = float(np.dot(vz_values, weights))
            #     print(f"[SPEED] uid={uid} avg_vz={self._avg_vz[uid]:.3f} m/s")

    def update_with_elevation(
        self,
        tracks,
        sin_el_1,
        sin_el_2,
        r_height,
    ):
        """
        For each active track estimate its height from the elevation map and
        feed it to FallDetector.
    
        Strategy
        --------
        * Collect CFAR detections that GTrack assigned to this track.
        GTrack stores the list of Detection objects in t['points'] if you
        expose them from GTrackUnit2D.report() — see note below.
        * For each detection: height = detection.range * sin_el[d_idx, r_idx]
        * Take the SNR-weighted mean as the track height estimate.
    
        If t['points'] is not available we fall back to using the track's
        range (distance from origin) and the mean sin_el at that range bin.
    
        NOTE: to get t['points'] add this line to GTrackUnit2D.report():
            out['points'] = list(self.associated_points)   # Detection objects
        """
        now = None  # lazy: time.time() called only once if needed
    
        for t in tracks:
            uid  = t['uid']
            x, y = t['pos']
            self.last_positions[uid] = (x, y)
    
            # --- Height estimation ---
            points = t.get('points')   # list[Detection] or None
    
            if points:
                heights = []
                weights = []
                for pt in points:
                    r_bin = int(np.round(pt.range))
                    r_bin = np.clip(r_bin, 0, sin_el_1.shape[1] - 1)
                    d_bin = int(np.round(getattr(pt, 'd_idx', 0)))
                    d_bin = np.clip(d_bin, 0, sin_el_1.shape[0] - 1)
    
                    # Average the two radar elevation maps
                    sin_el = 0.5 * (sin_el_1[d_bin, r_bin] + sin_el_2[d_bin, r_bin])
                    height = r_height + pt.range * sin_el
                    heights.append(height)
                    weights.append(pt.snr)
    
                if weights:
                    height_m = float(np.average(heights, weights=weights))
                else:
                    height_m = 0.0
            else:
                # Fallback: use track range and mean sin_el at that range bin
                track_range = float(np.hypot(x, y))
                r_bin = int(np.round(track_range))
                r_bin = np.clip(r_bin, 0, sin_el_1.shape[1] - 1)
                sin_el = 0.5 * (
                    np.mean(sin_el_1[:, r_bin]) + np.mean(sin_el_2[:, r_bin])
                )
                height_m = r_height + track_range * sin_el
    
            if now is None:
                now = time.time()
    
            self.update_elevation(uid, height_m, timestamp=now)

    # ------------------------------------------------------------------
    # Called by Fuser every frame with the set of currently active IDs
    # ------------------------------------------------------------------

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

        # Nettoyer les tracks vraiment disparues (> seuil) et alerter
        x_min, x_max, y_min, y_max = self.valid_zone
        for tid, count in list(self.miss_counter.items()):

            # Clean up tracks that have been gone way too long
            if count > self.fall_threshold + 30:
                del self.miss_counter[tid]
                self._height_history.pop(tid, None)
                self._avg_vz.pop(tid, None)
                continue
            
            if count >= self.fall_threshold and tid not in self.alerted_ids:

                 # ── Boundary check ───────────────────────────────────────────
                pos = self.last_positions.get(tid)
                if pos is None:
                    continue
                x, y = pos

                if not (x_min <= x <= x_max and y_min <= y <= y_max):
                    # Track exited through boundary → not a fall
                    print(
                        f"[FALL SKIPPED] track_id={tid} | "
                        f"boundary exit at ({x:.1f}, {y:.1f}) — not a fall"
                    )
                    del self.miss_counter[tid]
                    self._height_history.pop(tid, None)
                    self._avg_vz.pop(tid, None)
                    continue
                
                # ── Vertical-speed gate ──────────────────────────────────────
                vz = self._avg_vz.get(tid, 0.0)
                if self.require_vz and vz > self.vz_threshold:
                    # Not moving downward fast enough
                    print(
                        f"[FALL SKIPPED] track_id={tid} | "
                        f"absent {count} frames but vz={vz:.2f} m/s "
                        f"(threshold={self.vz_threshold:.2f}) — not moving down fast enough"
                    )
                    del self.miss_counter[tid]
                    self._height_history.pop(tid, None)
                    self._avg_vz.pop(tid, None)
                    continue
                
                # Fall register
                event = {
                    "track_id":       tid,
                    "missing_frames": count,
                    "vz_m_s":         vz,
                    "timestamp": time.time(),
                }
                new_falls.append(event)
                self.fall_events.append(event)
                self.alerted_ids.add(tid)
                print(
                    f"[FALL DETECTED] track_id={tid} | "
                    f"absent {count} frames | vz={vz:.2f} m/s"
                )

            elif count == 0 and tid in self.alerted_ids:
                self.alerted_ids.discard(tid)

        return new_falls