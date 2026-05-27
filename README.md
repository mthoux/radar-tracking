# Radar Tracking

A radar tracking project using radar data processing techniques.

## Hardware 

2 FMCW TI 77GHz Radars.

## Getting Started

Clone the repository:

git clone https://github.com/mthoux/radar-tracking.git
cd radar-tracking

Install dependencies:

pip install -r requirements.txt

Run the project:

c
```
radar-tracking/
└── src/            # Core source code
```

## 💡 Context & Motivation

### The Public Health Challenge
* [cite_start]**Critical Risk:** 1/3 of adults aged 65+ fall each year, making it the leading cause of accidental death among older adults[cite: 4, 5, 8].
* [cite_start]**The "Long Lie" Risk:** Lying on the floor for more than an hour drastically increases complications, leading to a 50% mortality rate within 6 months[cite: 32, 34].

### Why mmWave Radars over Cameras?
[cite_start]Privacy preservation is a major concern: **80% to 90% of older adults refuse optical cameras** in private spaces like bedrooms or bathrooms[cite: 36, 38, 40]. 

[cite_start]Our system uses **TI mmWave Radars** to guarantee 100% anonymized, passive, and fully automatic sensing[cite: 48, 66, 68, 78, 86]:

| Metric | Optical Cameras | Wearables (Bracelets) | Radar mmWave (Ours) |
| :--- | :---: | :---: | :---: |
| **Privacy Preservation** | [cite_start]Low [cite: 75] | [cite_start]High [cite: 67] | [cite_start]**Superior (Anonymized)** [cite: 68] |
| **Auto-Detection** | [cite_start]Yes [cite: 73] | [cite_start]No (Manual Action) [cite: 80] | [cite_start]**Fully Automatic** [cite: 78] |
| **Low Light / Bathroom** | [cite_start]Non [cite: 70] | [cite_start]Yes [cite: 83] | [cite_start]**Yes (RF Sensing)** [cite: 84] |
| **User Compliance** | [cite_start]Low [cite: 75] | [cite_start]Medium (Often forgotten) [cite: 85] | [cite_start]**High (Passive sensing)** [cite: 86] |

## 📂 Project Deliverables & Reports

This repository contains the complete documentation, academic reports, and presentations for this Bachelor project:

* **📄 [Project Proposal](./docs/Project_Proposal.pdf)**
* **📄 [Project Report](./docs/Project_Report.pdf)** – A comprehensive deep dive into the asynchronous signal processing pipeline, architectural choices, and performance evaluation.
* **📊 [Project Presentation Slides](./docs/Project_Presentation.pdf)** – The defense slide deck covering the public health challenges, technology comparisons, and live tracking results.

# Usage

Start streaming with executing ./stream.sh 

Start giving it the rights : chmod +x ./stream.sh

Launch first with -init flag to launch the data acquisition of the radars : ./stream.sh -init

When radars started you can then simply type ./stream.sh to launch streaming and processing again without the need of starting radar if already running.

# References 

AWR1843BOOST DOC : https://www.ti.com/lit/ug/tidueo9/tidueo9.pdf?ts=1779243139006

## 👥 Authors
* **[@mthoux](https://github.com/mthoux)**
* **[@Romain-du-25](https://github.com/Romain-du-25)**
* **[@DrMoebius1](https://github.com/DrMoebius1)**

---

## 🤝 Acknowledgements / Credits
This project is built upon and extends the work of previous academic and open-source contributors:

* **COM-304 Previous Year Project:** This repository is a direct continuation and evolution of the radar pipeline developed by the previous student cohort at [COM-304-Group-2/COM-304-Radars](https://github.com/COM-304-Group-2/COM-304-Radars).
* **Texas Instruments mmWave GTrack:** The tracking submodule inside `src/processing/consumer/gtrack` is implemented based on Texas Instruments' multi-target tracking algorithms.
* **OpenRadar Platform:** The base DCA1000 parsing modules utilize algorithms from the [OpenRadar Project](https://github.com/OpenRadar/OpenRadar), originally licensed under the **Apache License 2.0**.
* **mmwavecapture-std:** The hardware abstraction layer and advanced socket communication configurations in `src/mmwave/mmwavecapture` are adapted from the `mmwavecapture` library by **Louie Lu (<louielu@cs.unc.edu>)**, originally licensed under the **BSD 3-Clause License**.

---

## 📄 License
This project itself is licensed under the **MIT License**. Feel free to fork it, experiment with it, share it, or use it as educational material!