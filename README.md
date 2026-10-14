cat > README.md <<'EOF'
# ESP32-CAM Chessboard Localization

Embedded computer-vision and DSP project for chessboard localization using an **ESP32-CAM** and **OV2640 camera**.

The system captures a chessboard, extracts the luminance channel, performs image preprocessing, detects the complete 9×9 chessboard grid using a Harris-based method, determines the four board corners, and applies perspective correction to produce a normalized **96×96 grayscale chessboard image**.

The project was developed in **ESP-IDF** for the university course **Real-Time and DSP Techniques**.

---

## Final Result

The final implementation performs:

- OV2640 camera acquisition
- Y-channel extraction from YUV422
- 3×3 Gaussian filtering
- Harris corner detection
- Non-maximum suppression
- Grid-direction estimation
- 9×9 chessboard grid reconstruction
- Board-corner ordering
- Projective perspective correction
- Bilinear interpolation
- 96×96 normalized chessboard generation
- Per-stage performance benchmarking
- Diagnostic image saving to microSD

### Final Benchmark

| Metric | Result |
|---|---:|
| Input resolution | 160×120 |
| Output resolution | 96×96 |
| CPU frequency | 240 MHz |
| Grid detection success | 30 / 30 |
| Perspective warp success | 30 / 30 |
| Average localization time | 268.083 ms |
| Minimum localization time | 265.591 ms |
| Maximum localization time | 269.480 ms |
| Effective localization rate | 3.730 FPS |

The final benchmark completed with:

```text
RESULT: PASS