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

## Authors
* **[@mthoux](https://github.com/mthoux)**
* **[@Romain-du-25](https://github.com/Romain-du-25)**
* **[@DrMoebius1](https://github.com/DrMoebius1)**

---

## 📄 License
This project is licensed under the **MIT License**. Feel free to fork it, experiment with it, share it, or use it as educational material!