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