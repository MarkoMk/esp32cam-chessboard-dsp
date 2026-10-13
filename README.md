# ESP32-CAM Real-Time Chessboard Localization

Real-time image processing and chessboard localization project implemented on an ESP32-CAM using ESP-IDF.

The project was developed as part of a university course focused on **real-time techniques and digital signal processing (DSP)**.

The main goal is to capture a chessboard using an ESP32-CAM, process the image directly on the ESP32, detect the chessboard structure, and determine the position of the board inside the camera frame.

The project also benchmarks the computational cost of the different processing stages in order to evaluate whether the system can later be extended into a complete embedded chess assistant.

---

## Hardware

- ESP32-CAM
- OV2640 camera
- 2 MB PSRAM
- microSD card
- ESP32 dual-core Xtensa LX6
- Current CPU frequency: 160 MHz

The camera framebuffer is stored in PSRAM while the main DSP working buffers are allocated in internal RAM for better performance.

---

## Software

- ESP-IDF v5.5.1
- `espressif/esp32-camera`
- FATFS
- SDMMC
- ESP Timer
- ESP-IDF heap capabilities API

Target:

```text
ESP32
