<div align="center">

# LifeBeat
### Because Every Beat Matters

**A contactless vital signs monitoring application using smartphone camera and machine learning**

[![Download APK](https://img.shields.io/badge/Download%20APK-v2.0.0-red?style=for-the-badge&logo=android)](https://github.com/muhmdfarhan0/lifebeat-apk/releases/download/v2.0.0/app-release.apk)

## [⬇️ Click here to download and install LifeBeat on your Android phone](https://github.com/muhmdfarhan0/lifebeat-apk/releases/download/v2.0.0/app-release.apk)

> Tap the link above on your Android device → tap the downloaded file → tap Install

</div>

---

## Abstract

LifeBeat is a Final Year Project that addresses the growing need for accessible, real-time health monitoring without expensive medical equipment. Cardiovascular diseases remain the leading cause of death globally, yet routine vital sign monitoring is largely limited to clinical settings.

LifeBeat enables anyone with an Android smartphone to measure their **Heart Rate, Blood Oxygen (SpO2), Systolic Blood Pressure, and Diastolic Blood Pressure** in under 20 seconds — with no additional hardware. The user simply places their finger over the rear camera, and the app uses **Photoplethysmography (PPG)** — the same optical technique used in hospital pulse oximeters — extracted from a 15-second video recording.

The extracted PPG signal is processed through a signal processing pipeline (FFT, peak detection, autocorrelation) and fed into trained **scikit-learn machine learning models** deployed on a cloud backend. Results are stored in Firebase Firestore and the user receives personalised health advice from an **AI Therapist powered by Groq LLaMA**.

---

## Repositories

| Component | Repository | Description |
|---|---|---|
| 📱 **Frontend (Flutter App)** | [muhmdfarhan0/life-beat-final](https://github.com/muhmdfarhan0/life-beat-final) | Flutter Android app — UI, scan flow, Firebase, AI Chat |
| ⚙️ **Backend (FastAPI)** | [muhmdfarhan0/lifeeeebeat](https://github.com/muhmdfarhan0/lifeeeebeat) | Python REST API — PPG processing, ML inference, deployed on Render |
| 📦 **APK Distribution** | [muhmdfarhan0/lifebeat-apk](https://github.com/muhmdfarhan0/lifebeat-apk) | This repo — public APK releases for direct install |

**Live Backend URL:** `https://lifeeeebeat.onrender.com`

---

## How to Install

### On Android Phone
1. Open this link on your phone: **[Download LifeBeat v2.0.0](https://github.com/muhmdfarhan0/lifebeat-apk/releases/download/v2.0.0/app-release.apk)**
2. Wait for the APK file to download
3. Tap the downloaded file in your notification bar
4. If prompted, tap **"Install anyway"** or enable **"Install from unknown sources"** in Settings
5. Open **LifeBeat**, sign up with your email, and verify it
6. Start your first scan!

> **Requires:** Android 6.0 (Marshmallow) or higher

---

## What We Built

### The Problem
- Vital sign monitoring requires expensive dedicated hardware (pulse oximeters, BP cuffs)
- These devices are unavailable in remote areas and inconvenient for daily use
- There is no accessible, real-time, contactless solution for everyday health monitoring

### Our Solution
A smartphone app that uses the rear camera as a medical sensor. When light from the phone's flashlight passes through the fingertip, blood volume changes cause subtle variations in the reflected light. We capture these changes as a video and extract the PPG (Photoplethysmography) signal.

### What It Measures

| Vital Sign | Normal Range | Method |
|---|---|---|
| Heart Rate | 60–100 BPM | FFT + Peak Detection + Autocorrelation |
| SpO2 (Blood Oxygen) | 95–100% | ML regression on PPG features |
| Systolic Blood Pressure | 90–120 mmHg | ML regression on PPG + biometrics |
| Diastolic Blood Pressure | 60–80 mmHg | ML regression on PPG + biometrics |

---

## How It Works — Full Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     FLUTTER ANDROID APP                      │
│  Login → Home Dashboard → Tap "Start Scan"                  │
└─────────────────────┬───────────────────────────────────────┘
                      │ Pings /health to wake Render server
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     SCAN FLOW (15 seconds)                   │
│  1. Torch (flashlight) turned ON                            │
│  2. Camera records 15-second video at low resolution        │
│  3. Real-time redness check — aborts if finger removed      │
│  4. Torch turned OFF                                        │
│  5. Video file uploaded to backend via multipart POST       │
└─────────────────────┬───────────────────────────────────────┘
                      │ POST /predict (video + age/gender/height/weight)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              FASTAPI BACKEND (Render Cloud)                  │
│                                                             │
│  Step 1 — OpenCV reads video frame by frame                 │
│  Step 2 — Extracts mean GREEN channel value per frame       │
│           (Green wavelength is most sensitive to blood)     │
│  Step 3 — Applies bandpass filter (0.7–3.5 Hz)             │
│           removes noise and motion artifacts                │
│  Step 4 — Signal Processing:                               │
│           • FFT → dominant frequency → FFT_HR              │
│           • Peak Detection → inter-peak interval → Peak_HR │
│           • Autocorrelation → periodicity → AC_HR          │
│           • Hybrid_HR = mean(FFT_HR, Peak_HR, AC_HR)       │
│  Step 5 — Feature Engineering (13 features):               │
│           PPG: Hybrid_HR, FFT_HR, Peak_HR, AC_HR,          │
│                peaks_count, skewness, kurtosis,             │
│                mean_amp, rr_mean                            │
│           Bio: Age, Gender, Height_cm, Weight_kg            │
│  Step 6 — HR Model (hr_model.joblib):                      │
│           Residual correction model                         │
│           Final HR = Hybrid_HR + model.predict(features)   │
│  Step 7 — Vitals Model (multivariate_spo2_bp.joblib):      │
│           MultiOutputRegressor(HistGradientBoosting)        │
│           Outputs: SpO2, Systolic BP, Diastolic BP         │
│  Step 8 — Safety clamp (prevents extreme outlier values)   │
│  Step 9 — Therapy text generated from rule-based engine    │
└─────────────────────┬───────────────────────────────────────┘
                      │ JSON: HR, SpO2, SBP, DBP, therapy, quality_score
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    FLUTTER APP (cont.)                       │
│  Results saved to Firebase Firestore (users/{uid}/scans)   │
│  Results Screen shows animated vitals                       │
│  AI Therapist (Groq LLaMA) auto-analyzes last 3 scans      │
│  Home Dashboard updates weekly health score chart          │
└─────────────────────────────────────────────────────────────┘
```

---

## Machine Learning Models

### HR Model — `hr_model.joblib`
A **residual-based correction model**. Instead of predicting heart rate from scratch, it uses the Hybrid_HR (average of three signal processing estimates) as a strong baseline and predicts only the correction offset.

```
Hybrid_HR = mean(FFT_HR, Peak_HR, AC_HR)   ← signal processing baseline
Residual  = model.predict(9 PPG features)   ← ML correction
Final HR  = Hybrid_HR + Residual            ← combined output
```

### Vitals Model — `multivariate_spo2_bp.joblib`
A **multi-output regression model** using `HistGradientBoostingRegressor` (scikit-learn) wrapped in `MultiOutputRegressor`. Predicts SpO2, Systolic BP and Diastolic BP simultaneously from 13 input features.

**Why HistGradientBoosting?**
- Handles missing values natively (important for variable-quality PPG signals)
- No feature scaling required (tree-based model)
- Robust on small datasets — PPG training data is hard to collect with ground truth
- Faster than standard GradientBoosting via histogram binning

---

## App Features

- **Contactless scan** — no extra hardware, just your smartphone
- **15-second PPG scan** with live countdown and redness validation
- **Scan history** with trend charts per vital sign
- **Weekly health score** dashboard — one averaged score per day
- **AI Therapy Chat** — Groq LLaMA powered, context-aware of your vitals
- **Automated post-scan analysis** — AI compares your last 3 scans automatically
- **Email verification** — Firebase Auth with mandatory email confirmation
- **Dark / Light theme** support
- **PDF export** — share your health report

---

## Tech Stack

### Frontend (Flutter App)
| Package | Purpose |
|---|---|
| Flutter 3.x (Dart) | Cross-platform mobile framework |
| flutter_riverpod | Reactive state management |
| firebase_auth | Email authentication + verification |
| cloud_firestore | Real-time scan history database |
| camera | Video recording for PPG scan |
| fl_chart | Health score and history charts |
| permission_handler | Camera and microphone permissions |
| http | REST API calls to FastAPI backend |

### Backend (FastAPI — Python)
| Library | Purpose |
|---|---|
| FastAPI | REST API framework |
| OpenCV (cv2) | Video processing and green channel extraction |
| SciPy | Bandpass filter, FFT, peak detection |
| NumPy | Signal array operations |
| scikit-learn 1.7.2 | ML models (HistGradientBoosting, MultiOutputRegressor) |
| joblib | Model serialization and loading |
| uvicorn | ASGI server |

### Cloud and Services
| Service | Purpose |
|---|---|
| Firebase Auth | User authentication and email verification |
| Cloud Firestore | Scan results and user profile storage |
| Supabase | Session management abstraction layer |
| Render | Backend hosting — auto-deploy from GitHub |
| Groq AI (LLaMA 3) | AI Therapy Chat LLM |
| GitHub Releases | APK distribution (this repo) |

---

## Releases

| Version | Date | Changes |
|---|---|---|
| **v2.0.0** | 2026-05-12 | Real backend connected, video scan flow, permissions fixed, chart improved, AI auto-analysis added |
| v1.0.0 | 2026-05-11 | Initial release |

---

## Team

**Final Year Project — Air University, Islamabad (2025–2026)**

| Name | GitHub | Role |
|---|---|---|
| Muhammad Farhan | [@muhmdfarhan0](https://github.com/muhmdfarhan0) | Full Stack — Flutter + FastAPI + ML Pipeline |
| Nusama | [@us4321](https://github.com/us4321) | Frontend — Flutter UI + Firebase Integration |
| Hassaan Abdullah | [@hassaanabdullah107-netizen](https://github.com/hassaanabdullah107-netizen) | Backend — API + Signal Processing + ML Models |

---

<div align="center">

**LifeBeat — Because Every Beat Matters**

*PPG-based Contactless Vital Signs Monitoring using Smartphone Camera and Machine Learning*

</div>
