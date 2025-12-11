# Intelligent Spatiotemporal Vibration Patch for PD Gait Rehabilitation

> Flexible single-IMU e-skin patch + on-device gait recognition + personalized spatiotemporal vibration for Parkinson’s disease gait rehabilitation.

---

## System overview

<p align="center">
  <img src="https://github.com/user-attachments/assets/21729f51-f8d4-4e71-aa59-ac0dd11b6b24" alt="System overview of the intelligent spatiotemporal vibration patch" width="800">

</p>

The system consists of:

- A **48 mm × 38 mm flexible vibration patch** with:
  - Single 6-axis IMU
  - Four vibration motors (VIB1–VIB4)
  - MCU, BLE, power management, coil and battery
- A **Gait Phase Adaptive single-IMU algorithm** that reconstructs thigh angle in real time and labels stance / swing phases.
- A **dynamic gait database** that:
  - Uses a DBSCAN-inspired strategy to reject outlier / pathological cycles.
  - Maintains a time-weighted personalized template.
- A **fusion module** that merges the personalized template with a **normative gait template** and computes a **Gait Normality** index.
- A **spatiotemporal vibration controller** that triggers four-channel leg-lifting cues just before swing phase.
- A **Rehealthy smartphone app (Android)** for:
  - BLE connection and real-time visualization
  - Display of Gait Normality and stimulation mode
  - Local gait-cycle database management
- **Python analysis scripts** to reproduce the main results:
  - Kinematic accuracy vs. optical motion capture and commercial IMU
  - Spatiotemporal gait variability
  - Hip / knee joint Dynamic Time Warping (DTW) distances
  - EMG-based fatigue indices
  - Repeated-measures statistics across stimulation conditions

---

## Repository structure

```text
.
├── README.md                  # This file
├── core/                      # Real-time algorithms and controllers
│   ├── __init__.py
│   ├── gait_phase_adaptive.py # Single-IMU Gait Phase Adaptive filter
│   ├── gait_database.py       # Dynamic DBSCAN-style gait database + weighting
│   ├── template_fusion.py     # Normative–personal fusion + Gait Normality
│   ├── vibro_controller.py    # Spatiotemporal vibration triggering
│   └── signal_utils.py        # IMU preprocessing & gait-cycle segmentation
├── analysis/                  # Offline analysis and statistics
│   ├── gait_metrics.py        # Stride length, gait-cycle duration, stance ratio
│   ├── dtw_tools.py           # DTW distance for joint trajectories
│   ├── emg_processing.py      # EMG band-pass, RMS, iEMG, %-reduction
│   └── stats_anova.py         # One-way repeated-measures ANOVA + post-hoc
└── mobile_app/                # Android app project (Kotlin)
    ├── README.md              # App description
    └── RehealthyApp/          # Import this folder into Android Studio
        ├── settings.gradle
        ├── build.gradle
        └── app/
            ├── build.gradle
            ├── src/main/
            │   ├── AndroidManifest.xml
            │   ├── java/com/rehealthy/pdpatch/
            │   │   ├── MainActivity.kt
            │   │   ├── ble/BleManager.kt
            │   │   └── ui/GaitDashboardFragment.kt
            │   └── res/layout, values, ...
            └── ...
```

Main dependencies:

* `numpy` – numerical computing
* `scipy` – signal processing
* `pandas` – data handling
* `matplotlib` – plotting
* `scikit-learn` – optional clustering / ML
* `statsmodels` – statistics and ANOVA

### 1.2 Gait Phase Adaptive filter

`core/gait_phase_adaptive.py` implements a **phase-dependent complementary filter**:

* Uses accelerometer inclination when transient accelerations are small (stance).
* Uses integrated gyroscope when dynamic motion dominates (swing).
* Classifies gait phase using vertical acceleration variance and angle trend.
* Outputs smoothed **thigh flexion–extension angle** time series and phase labels.

The filter matches and slightly exceeds the accuracy of a commercial multi-IMU system when compared with optical motion capture in steady walking.

### 1.3 Dynamic gait database and fusion

`core/gait_database.py` and `core/template_fusion.py` implement:

* Resampling each gait cycle to a 0–100% phase vector.
* **Dynamic DBSCAN-style inclusion**:

  * Adaptive neighborhood radius based on the current database.
  * Cycles inconsistent with the main dense cluster are rejected.
* **Time-weighted averaging** to emphasize recent stable cycles.
* Fusion with a **normative template**:

  * Reliability-dependent weights between patient template and normative curve.
  * Output: fused target curve and scalar **Gait Normality** index.

### 1.4 Spatiotemporal vibration controller

`core/vibro_controller.py` converts thigh angle trajectories into **four-channel vibration commands**:

* Detects rapid leg-lifting initiation using angle and slope thresholds.
* Schedules short high-frequency bursts (e.g., 40 Hz, 0.2 s) at key gait phases.
* Supports:

  * **SpaVib**: phase-locked spatiotemporal pattern.
  * **Const**: constant vibration mode.
* Provides a helper to convert commands into time-stamped packets that can be sent over BLE to the patch MCU.

### 1.5 Minimal example

```python
import numpy as np
from core.gait_phase_adaptive import GaitPhaseAdaptiveFilter
from core.signal_utils import segment_gait_cycles
from core.gait_database import GaitDatabase
from core.template_fusion import TemplateFusion
from core.vibro_controller import VibrationController

# Load raw IMU data: acc (N,3), gyro (N,3), t (N,)
acc = np.load("examples/acc.npy")
gyro = np.load("examples/gyro.npy")
t = np.load("examples/time.npy")

# 1) Reconstruct thigh angle
gpa = GaitPhaseAdaptiveFilter()
theta, phases = gpa.run_batch(acc, gyro, t)

# 2) Segment gait cycles
cycles = segment_gait_cycles(theta, t)

# 3) Build personalized gait database
db = GaitDatabase()
for c in cycles:
    db.add_cycle(c.theta, c.time)

# 4) Fuse with normative template
norm_template = np.load("examples/norm_template.npy")
fusion = TemplateFusion(norm_template=norm_template)
target, (w_patient, w_norm) = fusion.get_fused_template(db)

# 5) Compute Gait Normality of the last cycle
from core.template_fusion import TemplateFusion as TF
gn_score = TF.gait_normality(cycles[-1].theta, target)
print("Gait Normality:", gn_score)

# 6) Generate vibration commands over the whole recording
controller = VibrationController()
commands = controller.generate_commands(theta, t, mode="SpaVib")
print("Generated", len(commands), "commands.")
```

---

## 2. Analysis scripts

The `analysis/` folder contains scripts to reproduce the quantitative results of the study.

### 2.1 Gait metrics

`gait_metrics.py` provides utilities to compute:

* **Stride length**
* **Gait-cycle duration**
* **Stance-phase ratio**
* Inter-cycle variance of the above parameters, used to quantify spatiotemporal gait stability under different stimulation conditions.

### 2.2 DTW analysis of joint kinematics

`dtw_tools.py` implements a classic Dynamic Time Warping (DTW) distance:

* Resamples patient hip / knee joint trajectories and normative curves to 0–100% gait cycle.
* Computes DTW distance to quantify waveform discrepancy.
* Used for:

  * Comparing patch vs. optical vs. commercial IMU trajectories.
  * Quantifying how spatiotemporal vibration drives the joint angles closer to the normative pattern.

### 2.3 EMG processing and iEMG

`emg_processing.py` implements the pipeline described in the paper:

* Band-pass filter (10–500 Hz) + 50 Hz notch.
* Full-wave rectification.
* Optional sliding-window RMS envelope.
* Integrated EMG (**iEMG**) over each walking trial.
* Percentage reduction in iEMG between conditions (e.g., NoVib vs. SpaVib).

### 2.4 Repeated-measures ANOVA

`stats_anova.py` wraps `statsmodels` and `scipy` to perform:

* One-way repeated-measures ANOVA across the **five stimulation conditions**.
* Bonferroni-corrected paired t-tests for post-hoc comparisons.

Example:

```python
import pandas as pd
from analysis.stats_anova import repeated_measures_anova, bonferroni_posthoc

df = pd.read_csv("results/stance_ratio_variance.csv")  # columns: subject, condition, value
anova_res = repeated_measures_anova(df, "subject", "condition", "value")
print(anova_res)

posthoc = bonferroni_posthoc(df, "subject", "condition", "value")
print(posthoc)
```

---

## 3. Android mobile app

The `mobile_app/RehealthyApp` folder is a minimal Android Studio project (Kotlin, AndroidX).

### 3.1 Features (skeleton)

* Scan and connect to the patch via BLE (default name: **`ReHealthyDevice`**).
* Subscribe to IMU / gait metrics notifications (placeholders).
* Display:

  * Connection status
  * Basic text representation of gait metrics
* Provide a starting point for:

  * Real-time thigh angle plots
  * Gait Normality gauge
  * Stimulation mode control and logs

### 3.2 Quick start

1. Open Android Studio → **Open an Existing Project**.
2. Select `mobile_app/RehealthyApp`.
3. Sync Gradle and build.

You will see:

* `MainActivity.kt` – hosts a simple Fragment container.
* `GaitDashboardFragment.kt` – button to connect + status / metrics text views.
* `BleManager.kt` – BLE scanning, connection and notification callbacks.
* Basic layouts (`activity_main.xml`, `fragment_gait_dashboard.xml`) and theme resources.

> **Important:**
> Replace the placeholder UUIDs and data parsing logic in
> `BleManager.kt` with the actual BLE service / characteristic UUIDs and data format used by your patch firmware.

---

## 4. Data formats (recommended)

Although not fixed, we recommend logging data in the following format to facilitate use with the provided scripts:

### 4.1 IMU + angle log (CSV)

Columns:

* `t` – time stamp (s)
* `ax, ay, az` – accelerometer in g
* `gx, gy, gz` – gyroscope in deg/s
* *(optional)* `theta_patch` – on-device thigh angle (for debugging)

### 4.2 Gait-cycle level metrics (CSV)

Per subject × condition × trial:

* `subject_id`
* `condition` (e.g., `NoVib`, `SpaVib_40Hz`, `ConstVib`, …)
* `stride_length_var`
* `gait_cycle_duration_var`
* `stance_ratio_var`
* `hip_dtw`
* `knee_dtw`
* `iemg_TA`, `iemg_RF`, `iemg_BF`, `iemg_GAS`, …

These tables can be directly fed into `stats_anova.py` for statistical analysis.

---

## 5. How to reproduce the paper’s main results

1. **Kinematic validation**

   * Collect synchronous patch, optical motion capture, and commercial IMU data.
   * Use `GaitPhaseAdaptiveFilter` to reconstruct thigh angle from the patch.
   * Use `dtw_tools.py` to compute DTW distances vs. optical and commercial IMU trajectories.
2. **PD gait intervention study**

   * For each subject and condition:

     * Use `core/` to process IMU and generate gait cycles.
     * Use `analysis/gait_metrics.py` for spatiotemporal variability.
     * Use `analysis/emg_processing.py` for iEMG and %-reduction.
   * Combine all subjects into long-format tables and run `stats_anova.py`.
3. **Visualization**

   * Plot thigh angle trajectories, variability bar plots, DTW distances, and EMG results using `matplotlib` or your preferred tool.

---

## 6. License

This repository is released under the **MIT License** (or any other license you choose).
Create a `LICENSE` file at the project root and specify the exact terms.

---
