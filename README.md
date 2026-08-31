# Radar Tracking

This project was developed as part of the Intelligent Systems (COM-304) bachelor course, where the core objective was to design and implement an intelligent system through a hands-on engineering project. 

We chose to explore a radar-based approach for human tracking because it provides a reliable foundation for building smart, privacy-preserving monitoring systems. Specifically, the project addresses elderly fall detection—a critical application where continuous oversight is essential, yet traditional alternatives like camera surveillance and wearable devices often present privacy concerns and low user compliance. 

To resolve these challenges, our system utilizes millimeter-wave radar to track indoor movement and detect motion seamlessly, while an Arduino Uno integrated with an LED and a buzzer establishes a straightforward hardware layer for immediate visual and auditory alerts.

## Project deliverables & presentation

This repository includes the academic documents produced during the project.
For a quick overview of the project, see the presentation.

- **📄 [Project Proposal](./docs/Project_Proposal.pdf)**
- **📄 [Project Report](./docs/Project_Report.pdf)** – details on the processing pipeline and evaluation.
- **📊 [Project Presentation Slides](./docs/Project_Presentation.pdf)** – final project presentation.

## System overview

The system uses two FMCW TI 77 GHz radars to observe the same indoor scene. Using two sensors improves coverage and robustness by reducing blind spots and making the fused map more reliable than a single-radar setup.

Each radar generates range-Doppler data that is converted into a polar representation, then synchronized and projected into a common Cartesian frame so that both sensors contribute to the same occupancy map.

The pipeline then works as follows:

- the two radar streams are synchronized and fused with a maximum-intensity strategy to build a single 2D occupancy map;
- temporal smoothing and background subtraction are applied to suppress static clutter and keep only moving targets;
- CFAR-like thresholding is used to extract candidate detections above the signal level;
- the GTrack module performs clustering and Kalman-based tracking to maintain and update target trajectories over time;
- the fall detection block monitors track disappearance and vertical motion to flag possible falls while reducing false alarms;
- the Arduino interface receives the alarm state and triggers a visual/auditory feedback signal.

The result is a real-time tracking and alert pipeline that can be visualized live and reused as a base for safety-oriented motion monitoring.

## Hardware

- 2 FMCW TI 77 GHz radars
- Texas Instruments AWR1843BOOST platform
- DCA1000 acquisition chain
- Arduino Uno with LED and buzzer for visual/auditory feedback

## Usage

### 1. Make the launcher executable

```bash
chmod +x ./stream.sh
```

### 2. Initialize the radars

This starts the radar acquisition process:

```bash
./stream.sh -init
```

### 3. Start the streaming and processing pipeline

Once the radars are already running, you can launch the system normally with:

```bash
./stream.sh
```

This starts the stream and processing workflow without needing to restart the radars unless a new initialization is required.

## References

- AWR1843BOOST documentation: https://www.ti.com/lit/ug/tidueo9/tidueo9.pdf?ts=1779243139006

## 👥 Authors

- **[@mthoux](https://github.com/mthoux)**
- **[@Romain-du-25](https://github.com/Romain-du-25)**
- **[@DrMoebius1](https://github.com/DrMoebius1)**

---

## 🤝 Acknowledgements / Credits

This project builds on previous academic and open-source work:

- **COM-304 Previous Year Project:** continuation of the radar tracking pipeline developed by the previous cohort at [COM-304-Group-2/COM-304-Radars](https://github.com/COM-304-Group-2/COM-304-Radars).
- **Texas Instruments mmWave GTrack:** the tracking logic in `src/processing/consumer/gtrack` is based on Texas Instruments multi-target tracking algorithms.
- **OpenRadar Platform:** the DCA1000 parsing modules rely on algorithms from the [OpenRadar Project](https://github.com/OpenRadar/OpenRadar), originally licensed under the **Apache License 2.0**.
- **mmwavecapture-std:** the hardware abstraction and communication layer in `src/mmwave/mmwavecapture` are adapted from the `mmwavecapture` library by **Louie Lu (<louielu@cs.unc.edu>)**, originally licensed under the **BSD 3-Clause License**.

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to fork, experiment, share, or use it as educational material.