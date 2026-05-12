import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import time
import numpy as np

from scipy.interpolate import RegularGridInterpolator
from multiprocessing import Process, Queue
from direct.showbase.ShowBase import ShowBase
from direct.task import Task

import matplotlib
matplotlib.use('Qt5Agg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-dark')

from panda3d.core import loadPrcFileData
loadPrcFileData('', 'window-type none')   # no native GL window

from PyQt5 import QtWidgets

from .prod_dca import producer_real_time_1843
from .fall_detection import FallDetector

from visualization.visualization import configure_ax_bf, configure_ax_db, configure_ax_gtrack, update_ax_gtrack
from utils.utils import cart2pol
from gtrack.config import Detection
from gtrack.module import GTrackModule2D


def consumer(q1, q2, cfg_radar, cfg_gtrack):
    """
    Consumer function that processes data from two queues, performs beamforming,

    Parameters
    ----------
    q1 : Queue
        The first queue containing radar data.
    q2 : Queue
        The second queue containing radar data.
    cfg_radar : dict
        Configuration dictionary for the radar, including parameters like phi, range indices, and offsets.
    cfg_gtrack : dict
        Configuration dictionary for the GTrack module, including parameters like minimum SNR threshold.
    """
    app = MyApp(q1, q2, cfg_radar, cfg_gtrack)
    app.run()


class MyApp(ShowBase):
    """
    MyApp class that extends ShowBase to create a Panda3D application for real-time radar data visualization.
    """
    def __init__(self, queue_1, queue_2, cfg_radar, cfg_gtrack):
        ShowBase.__init__(self)
        self.q1 = queue_1
        self.q2 = queue_2
        self.latest_msg = {}
        self.msg_count = set()

        self.phi = cfg_radar["phi"]
        self.r_idxs = cfg_radar["range_idx"]
        self.treshold = cfg_gtrack.min_snr_threshold

        self.fig = plt.figure(figsize=(12, 6))

        # Utilisation de GridSpec ou add_subplot pour diviser la fenêtre
        # 121 signifie : 1 ligne, 2 colonnes, position 1
        self.ax = self.fig.add_subplot(121, projection='polar')
        self.im = configure_ax_bf(self.ax, self.phi, self.r_idxs)

        # 122 signifie : 1 ligne, 2 colonnes, position 2
        self.ax_3 = self.fig.add_subplot(122)
        configure_ax_gtrack(self.ax_3, cfg_radar["width"], len(self.r_idxs))

        self.last_frame_time = time.time()
        self.frame_counter = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_text = self.ax_3.text(0.00, 1.05, "", transform=self.ax_3.transAxes, fontsize=10, color='blue')

        self.taskMgr.add(self.updateTask, "updateTask")

        self.x1, self.y1 = cfg_radar["offset_x_1"], cfg_radar["offset_y_1"]
        self.x2, self.y2 = cfg_radar["offset_x_2"], cfg_radar["offset_y_2"]

        self.x = np.arange(-cfg_radar["width"], cfg_radar["width"], 1)
        self.y = self.r_idxs
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')

        self.cart2pol = cart2pol(self.X.ravel(), self.Y.ravel())

        self.tracker = GTrackModule2D(cfg_gtrack)
        
        # Instantiation of the fall detector
        self.fall_detector = FallDetector(fall_threshold_frames=20)

        # Historique de chutes
        self.fall_log_text = self.ax_3.text(
            1.02, 1.0, "Chutes :\n—",
            transform=self.ax_3.transAxes,
            fontsize=8, color='red', verticalalignment='top'
        )

        self.last_artists = []

        #==================================
        # Background removal 
        self.CLUTTER_LEARN = 50          # number of frames to accumulate (~2.5 s at 20 fps)
        self.clutter_frames = []         # rolling buffer of normalized frames
        self.clutter_map = None          # learned static background, set after CLUTTER_LEARN frames

    def _on_fall_detected(self, event: dict):
        """Appelé une fois par chute détectée."""
        tid   = event["track_id"]
        ts    = time.strftime("%H:%M:%S", time.localtime(event["timestamp"]))

        # Affichage sur la figure
        self.ax_3.set_title(
            f"⚠ CHUTE détectée — track {tid} à {ts}",
            color="red", fontweight="bold"
        )

        # Historique complet dans le coin
        lines = [
            f"{time.strftime('%H:%M:%S', time.localtime(e['timestamp']))} — track {e['track_id']}"
            for e in self.fall_detector.fall_events
        ]
        self.fall_log_text.set_text("Chutes :\n" + "\n".join(lines))


    def updateTask(self, task):
        """
        Update task that processes the radar data from the queues, performs beamforming,

        Parameters
        ----------
        task : Task (unused)
            The task object provided by Panda3D's task manager.
        """

        try:
            for pid, q in enumerate((self.q1, self.q2)):
                while not q.empty():
                    msg = q.get_nowait()
                    if msg[0] == 'bev':
                        self.latest_msg[pid] = msg[1]
                        self.msg_count.add(pid)

        except:
            pass

        # Check if we have received a new messages from both producers
        if self.msg_count == {0, 1}:
            # Unpack the latest message
            bf_1 = self.latest_msg[0]
            bf_2 = self.latest_msg[1]

            phi1 = np.arctan2((self.Y - self.y1).ravel(), (self.X - self.x1).ravel())
            r1 = np.hypot(self.X.ravel() - self.x1, self.Y.ravel() - self.y1)
            cart2pol1 = np.column_stack((phi1, r1))

            phi2 = np.arctan2((self.Y - self.y2).ravel(), (self.X - self.x2).ravel())
            r2 = np.hypot(self.X.ravel() - self.x2, self.Y.ravel() - self.y2)
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
            Z_cart = (Z1 * Z2)

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

            # Normalize the output
            to_plot = np.abs(Z_polar)
            max_val = np.max(to_plot)
            if max_val > 0:
                to_plot /= max_val
            to_plot = to_plot ** 8

            #======================================================================

            # --- Background learning phase (first CLUTTER_LEARN frames) ---
            if len(self.clutter_frames) < self.CLUTTER_LEARN:
                self.clutter_frames.append(to_plot.copy())
                self.clutter_map = np.mean(self.clutter_frames, axis=0)
                remaining = self.CLUTTER_LEARN - len(self.clutter_frames)
                self.ax.set_title(f"Learning background... ({remaining} frames left)")
 
                # Show raw (squared for contrast) during learning so display is not blank
                self.im.set_array((to_plot ** 8).ravel())
                self.fig.canvas.draw_idle()
                QtWidgets.QApplication.processEvents()
                self.msg_count.clear()
                plt.pause(0.001)
                return Task.cont  # skip gtrack until background is learned
 
            # --- Background subtraction ---
            # Subtract the learned static clutter map; clip negatives to 0
            # so only dynamic (moving/changed) parts remain
            to_plot = np.clip(to_plot - self.clutter_map, 0, None)

            #======================================================================

            # Update the beamforming plot
            self.im.set_array(to_plot.ravel())

            # Convert map to Detections points
            detections = [
                Detection(r=self.r_idxs[i], az=self.phi[j], v=0, snr=to_plot[j, i])
                for i in range(len(self.r_idxs))
                for j in range(len(self.phi))
                if to_plot[j, i] >= self.treshold
            ]

            # Run GTrack
            gtrack_output = self.tracker.step(detections)

            # Update known positions
            for t in gtrack_output['tracks']:
                self.fall_detector.last_positions[t['uid']] = (t['pos'][0], t['pos'][1])

            # Fall detection
            active_ids = {t['uid'] for t in gtrack_output['tracks']}
            fall_events = self.fall_detector.update(active_ids)

            for event in fall_events:
                self._on_fall_detected(event)

            # Update the gtrack plot
            tracks = gtrack_output['tracks']
            update_ax_gtrack(self.ax_3, tracks, self.last_artists)

            # FPS tracking
            current_time = time.time()
            self.frame_counter += 1
            if current_time - self.last_fps_time >= 1.0:  # Every 1 second
                self.fps = self.frame_counter / (current_time - self.last_fps_time)
                self.last_fps_time = current_time
                self.frame_counter = 0
                # Update the threshold with fps mesured
                self.fall_detector.fall_threshold = max(5, int(self.fps * 2.5))

            # Update FPS text on the polar plot
            self.fps_text.set_text(f"FPS: {self.fps:.2f}")

            # Update the figure
            self.fig.canvas.draw_idle()
            #self.fig_3.canvas.draw_idle()

            # Redraw the canvas
            QtWidgets.QApplication.processEvents()

            # Reset the message count
            self.msg_count.clear()

            plt.pause(0.001)

        return Task.cont


def main(cfg_radar, cfg_gtrack, cfg_cfar):
    """
    Main function to start the real-time radar streaming and processing.

    Parameters
    ----------
    cfg_radar : dict
        Configuration dictionary for the radar, including parameters like phi, range indices, and offsets.
    cfg_gtrack : dict
        Configuration dictionary for the GTrack module, including parameters like minimum SNR threshold.
    cfg_cfar : dict
        Configuration dictionary for the CFAR processing, including parameters like window size and guard size.
    """

    # Set up the queues
    q_main_1 = Queue(maxsize=1)  # ❗️ Only keep latest
    q_main_2 = Queue(maxsize=1)

    # Create producer and consumer processes
    producers = [
        Process(target=producer_real_time_1843,args=(q_main_1, cfg_radar, cfg_cfar, 4096, 4098, "192.168.33.30", "192.168.33.180"), daemon=True),
        Process(target=producer_real_time_1843, args=(q_main_2, cfg_radar, cfg_cfar, 4099, 5000, "192.168.33.32", "192.168.33.182"), daemon=True)
        ]
    consumers = [Process(target=consumer, args=(q_main_1, q_main_2, cfg_radar, cfg_gtrack), daemon=True)]

    # Start the producer and consumer processes
    for p in producers: p.start()
    for c in consumers: c.start()

    print("✅ Streaming started.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for p in producers: p.terminate()
        for c in consumers: c.terminate()
        for p in producers: p.join()
        for c in consumers: c.join()
        print("✅ Shutdown complete.")

