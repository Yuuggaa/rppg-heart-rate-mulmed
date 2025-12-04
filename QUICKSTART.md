# Quick Start Guide - rPPG System

## Instalasi Cepat

1. **Install Python Dependencies**

```bash
pip install -r requirements.txt
```

2. **Run System**

```bash
python run.py
```

## Struktur Folder

```
├── src/           # Source code utama
├── logs/          # Log files
├── run.py         # Launcher utama (jalankan ini!)
├── requirements.txt
├── README.md
└── QUICKSTART.md
```

## Checklist Sebelum Running

- [ ] Webcam terhubung dan berfungsi
- [ ] Pencahayaan ruangan cukup (tidak terlalu gelap/terang)
- [ ] Posisi wajah di depan kamera (frontal view)
- [ ] Close aplikasi lain yang menggunakan webcam
- [ ] Install semua dependencies

## Expected Behavior

### Startup (0-5 detik)

- Camera initialization dengan FPS detection
- Log file dibuat di folder `logs/`
- Visualization window muncul

### Running (5-30 detik)

- Video feed menampilkan wajah dengan ROI overlay (Blue forehead, Green/Red cheeks)
- Signal plot mulai menampilkan waveform
- BPM estimate muncul setelah ~10 detik (Fast window)
- BPM stabil setelah ~30 detik (Slow window converged)

### Normal Operation

- BPM display: 60-100 (typical resting heart rate)
- SQI: > 0.5 (good signal quality)
- Motion: STABLE (no motion detected)
- Light: STABLE (stable lighting)
- Face: YES (face detected)

## Common First-Run Issues

### Issue: ModuleNotFoundError

**Fix:**

```bash
pip install opencv-python mediapipe numpy scipy matplotlib
```

### Issue: Camera not found

**Fix:**

- Check camera connection
- Try different CAMERA_ID in config.py (0, 1, 2)

### Issue: No face detected

**Fix:**

- Position face directly in front of camera
- Improve lighting
- Remove obstructions (mask, hands)

### Issue: Window not responding

**Fix:**

- Press ESC to stop
- Force close and restart

## Performance Tips

### For Low-End PC

Edit `config.py`:

```python
RESOLUTION = (640, 480)  # Lower resolution
TARGET_FPS = [30, 20, 15]  # Skip 60 FPS
```

### For Better Accuracy

- Sit still for 30 seconds
- Ensure good lighting
- Keep face centered
- Wait for SQI > 0.7

## Keyboard Controls

- **ESC**: Stop and exit application
- **Window Close**: Stop and exit application

## Log Files Location

`logs/rppg_session_YYYYMMDD_HHMMSS.log`

Check this file for detailed information and errors.

## Demo Video Guide

1. Start application
2. Position face in frame
3. Wait 10 seconds for initial estimate
4. Wait 30 seconds for accurate estimate
5. Observe:
   - ROI overlay on face (colored boxes)
   - Signal waveform (cyan=raw, green=filtered)
   - Frequency spectrum with peak marker
   - BPM history graph
   - Status panel metrics

## Support

If issues persist:

1. Check log file for errors
2. Review README.md troubleshooting section
3. Verify all dependencies installed correctly
4. Test camera with other application first
