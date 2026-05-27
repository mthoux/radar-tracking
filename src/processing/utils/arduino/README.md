# Arduino Radar Alarm Subsystem

This folder contains the embedded micro-controller firmware and the diagnostic test bench used to interface the Texas Instruments mmWave radar tracking application with an external physical alarm system.

## 📂 Folder Structure

```text
arduino/
├── firmware/
│   └── firmware.ino      # C++ Arduino sketch with case-sensitive protocol
└── test_arduino.py       # Python standalone serial CLI test script
```
---

## Hardware Layout & Pin Mapping

Connect your LEDs and Buzzer to the following digital pins on your Arduino Uno board:

| Component | Arduino Pin | Protocol Turn ON | Protocol Turn OFF | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Green LED** | **Pin 12** | 'G' | 'g' | Targets tracked status indicator |
| **Red LED + Buzzer** | **Pin 11** | 'R' | 'r' | Fall Alarm |
| **Blue LED** | **Pin 10** | 'B' | 'b' | Background removal indicator |

> **Note:** Ensure proper current-limiting resistors (e.g., 220 ohms) are placed in series with your LEDs to protect the Arduino digital output pins.

---

## Deployment Guide

### 1. Flash the Arduino Firmware
You only need to perform this step once, as the program remains permanently in the Arduino's non-volatile flash memory.

1. Open the **Arduino IDE**.
2. Open the file located at firmware/firmware.ino.
3. Connect your **Arduino Uno** to your computer via USB.
4. Navigate to **Tools > Board** and select **Arduino Uno**.
5. Navigate to **Tools > Port** and select your active port:
   * **macOS/Linux:** /dev/tty.usbmodemXXXX or /dev/tty.usbmodemXXXX
   * **Windows:** COM3, COM4, etc.
6. Click the **Upload** (Right Arrow) button.

---

## Hardware Test Bench Automation

A standalone script is provided to manually toggle the outputs and troubleshoot hardware issues without booting up the entire heavy radar data processing pipeline.

### Prerequisites
Ensure pyserial is installed within your active development environment

### Configuration
Open test_arduino.py and modify the SERIAL_PORT variable at the top of the script to match your operating system's detected address
```text
SERIAL_PORT = "/dev/tty.usbmodem1401"  # Replace with your actual port
```

### Execution
Run the interactive CLI command directly from this directory:
```text
python test_arduino.py
```

### Command Reference (Case-Sensitive)
Once running, type any of the following characters followed by :
*  G / g : Toggle **Green LED** ON / OFF
*  R / r : Toggle **Red LED + Buzzer** ON / OFF
*  B / b : Toggle **Blue LED** ON / OFF
*  Q / q : Exit the interactive CLI test utility cleanly