import numpy as np
import queue
import time

class FallDetector:
    """
    Détecte les chutes en surveillant la disparition prolongée de tracks.
    
    Une track est considérée "chutée" si elle n'apparaît plus dans
    gtrack_output pendant au moins `fall_threshold_frames` frames consécutives.
    Le délai de grâce (grace_frames) absorbe les occultations courtes ou
    les pertes de détection temporaires dues au bruit radar.
    """

    def __init__(self, fall_threshold_frames=15, valid_zone=(-30, 30, 5, 95)):
        self.fall_threshold = fall_threshold_frames
        self.valid_zone = valid_zone  # (x_min, x_max, y_min, y_max)

        # {track_id: nb de frames consécutives sans détection}
        self.miss_counter: dict[int, int] = {}

        # IDs pour lesquels une alerte a déjà été émise (évite les doublons)
        self.alerted_ids: set[int] = set()

        # Historique pour callback externe ou log
        self.fall_events: list[dict] = []

        # {uid: (x, y)}
        self.last_positions: dict[int, tuple] = {}  

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
            if count >= self.fall_threshold and tid not in self.alerted_ids:
                pos = self.last_positions.get(tid)
                if pos is None:
                    continue
                x, y = pos
                if not (x_min <= x <= x_max and y_min <= y <= y_max):
                    # Track exited through boundary → not a fall
                    del self.miss_counter[tid]
                    continue
                
                event = {
                    "track_id":      tid,
                    "missing_frames": count,
                    "timestamp":      time.time(),
                }
                new_falls.append(event)
                self.fall_events.append(event)
                self.alerted_ids.add(tid)
                print(f"[FALL DETECTED] track_id={tid} absent depuis {count} frames")

            elif count == 0 and tid in self.alerted_ids:
                # Track revenue — réinitialiser l'alerte
                self.alerted_ids.discard(tid)

            # Supprimer les tracks disparues depuis longtemps pour éviter la fuite mémoire
            if count > self.fall_threshold: #+ 30:
                del self.miss_counter[tid]

        return new_falls
