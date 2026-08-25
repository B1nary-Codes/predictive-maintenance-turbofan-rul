# NASA C-MAPSS Predictive Maintenance: Engineering Log

## Plain-English Executive Summary (The "Explain Like I'm 5" Guide)

**1. The Big Goal**
* **The Analogy:** Imagine driving a car where the fuel light doesn't tell you how much gas is left—it just tells you when you're completely stranded on the highway. That is bad. 
* **What We Are Doing:** We are building an AI warning system for airplane engines that looks at sensor data (temperature, pressure, fan speed) and accurately predicts: *"You have exactly 15 flights left before this part fails—schedule maintenance now."*

**2. Cleaning the Gauges (Data Pruning)**
* **The Analogy:** You are looking at a dashboard with 26 different dials. You notice 7 of those dials are frozen, turned off, or glued to zero.
* **What We Did:** We threw away those 7 useless dials so the AI doesn't waste energy watching gauges that never move.

**3. Smoothing the Jitter (Rolling Features)**
* **The Analogy:** If you measure your heart rate while sprinting up a flight of stairs, it spikes for a second. That doesn't mean you are having a heart attack; it just means you hit a temporary bump. But if your average heart rate stays high over a full hour—and starts shaking wildly—you're in trouble.
* **What We Did:** Instead of looking at split-second sensor spikes (throttle bumps), we averaged sensor data over 10-flight windows to track real, underlying wear and tear.

**4. The "Brand-New Engine" Rule (Target Clipping)**
* **The Analogy:** A 20-year-old human and a 20-and-a-half-year-old human are physically identical in health. A brand-new engine doesn't start breaking down on flight #2. 
* **What We Did:** We told the AI: *"Treat all early-life, perfectly healthy engines as 100% healthy (capped at 125 cycles left). Only start predicting degradation when real wear actually begins."*

**5. No Cheating on the Test (GroupKFold)**
* **The Analogy:** If a teacher gives you a study guide containing the exact 10 questions on tomorrow's exam, you didn't learn the material—you just memorized the answers. 
* **What We Did:** We made sure the AI never saw flight cycles from the same engine in both its study guide (training set) and its final exam (validation set). It has to learn how *any* jet engine ages, not just memorize specific engine serial numbers.

**6. Choosing the Model (Random Forest vs. XGBoost)**
* **The Analogy:** 
  * **Random Forest:** Asking a committee of 100 independent mechanics to inspect an engine and taking the average of their guesses.
  * **XGBoost:** Having one expert make a guess, a second expert inspect where the first expert made a mistake, a third expert inspect the second expert's mistakes, and repeating that 100 times until the team gets it right.

---

## Technical Architectural Log

### Phase 1: Infrastructure & Software Architecture
Developing industrial machine learning systems requires strict decoupling between data processing, feature engineering, and model training routines. Mixing data pre-processing scripts directly into model training loops leads to code duplication, testing difficulty, and subtle validation leakage. 

We established a modular repository layout (`src/data_loader.py`, `src/features.py`, `src/train.py`). Using relative `Path` definitions via Python's `pathlib` ensures cross-platform portability across Linux Codespaces containers and local execution environments.

### Phase 2: Ingestion & Schema Enforcement
The raw NASA C-MAPSS dataset consists of unformatted, space-delimited text files lacking headers. To establish structured inputs, we defined a 26-column schema mapping operational settings and 21 thermodynamic sensor channels.

### Phase 3: Variance Profiling & Low-Variance Channel Pruning
Standard deviation profiling revealed seven non-informative channels ($\sigma < 10^{-4}$): `setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, and `sensor_19`. In physical turbines, these parameters reflect locked control variables or uncalibrated channels that remain static throughout all 20,631 cycles.

Retaining static channels increases matrix memory overhead and injects non-informative dimensions into decision tree splits. Dropping these zero-variance attributes reduced matrix width from 26 to 19 clean features without stripping dynamic degradation signals.

### Phase 4: Degradation Target Construction & Piecewise Target Clipping
We calculated initial Remaining Useful Life ($RUL_{i,t} = T_{i,\text{max}} - t$) grouped by engine unit (`unit_nr`). 

In real physical systems, turbofans suffer negligible wear during initial operating cycles. A component at cycle 10 exhibits identical physical integrity to cycle 40. Linear target structures ($RUL = T_{\text{max}} - t$) force models to attempt predicting structural degradation during early healthy states where sensor signals remain completely static.

To align training targets with physical wear dynamics, we implemented a piecewise linear target capped at $RUL = 125$ cycles:

$$RUL_{\text{piecewise}} = \min(125, T_{i,\text{max}} - t)$$

Capping target values prevents the regression algorithm from attempting to separate indistinguishable healthy operational states, significantly sharpening model convergence near true degradation inflection points.

### Phase 5: Time-Series Feature Engineering & Signal Smoothing
Raw sensor streams contain high-frequency noise induced by transient operational environmental changes. To separate momentary sensor fluctuations from structural component wear, we engineered two distinct time-series features per active sensor using a 10-cycle window:

$$\text{Rolling Mean}_{i,t} = \frac{1}{W} \sum_{k=0}^{W-1} S_{i, t-k}$$

$$\text{Rolling Std}_{i,t} = \sqrt{\frac{1}{W-1} \sum_{k=0}^{W-1} (S_{i, t-k} - \bar{S}_{i,t})^2}$$

* **Rolling Mean:** Acts as a low-pass filter to reveal long-term thermodynamic drift.
* **Rolling Standard Deviation:** Measures physical degradation instability, capturing escalating sensor jitter as turbofan components lose structural integrity near failure.
* **Grouping Constraints:** Computations are grouped strictly by `unit_nr` (`groupby("unit_nr")`). This prevents rolling calculations at cycle 1 of Engine #2 from pulling historical sensor values from the tail end of Engine #1.

### Phase 6: Validation Strategy & Group Cross-Validation
Standard random $K$-fold cross-validation is strictly invalid for time-series predictive maintenance. Shuffling rows at random places cycle $t$ of an engine in the training set and cycle $t+1$ of that exact same engine in the validation set. Decision trees can easily memorize individual engine signatures, creating unrealistically optimistic validation metrics that fail in real-world deployment.

We implemented `GroupKFold` cross-validation grouped by `unit_nr`. This structure guarantees that all operational cycles of a given engine are assigned exclusively to either the training split or the validation split per fold. Evaluating out-of-sample prediction metrics exclusively on unseen engine units accurately simulates real-world production deployment on newly deployed aircraft.

### Phase 7: Metric Evaluation Framework
* **Root Mean Squared Error (RMSE):**

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

Squaring residual errors penalizes large prediction errors severely. Overestimating remaining useful life on a degrading engine can lead to catastrophic component failure. RMSE acts as our primary metric to heavily penalize over-predictions near engine failure boundaries.

* **Mean Absolute Error (MAE):**

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

Provides an unweighted linear measure of average prediction error across all flight cycles.

---

## Architecture Matrix Summary

| Phase | Input Shape | Output Shape | Primary Operation | Key Module |
| :--- | :--- | :--- | :--- | :--- |
| **01. Ingestion** | Raw text file | `(20631, 26)` | Schema Mapping & Parsing | `src/data_loader.py` |
| **02. Pruning** | `(20631, 26)` | `(20631, 19)` | Variance Filtering ($\sigma < 10^{-4}$) | `src/data_loader.py` |
| **03. RUL Target** | `(20631, 19)` | `(20631, 20)` | Grouped Max Cycle Calculation | `src/data_loader.py` |
| **04. Feature Eng.** | `(20631, 20)` | `(20631, 51)` | Rolling Stats + Piecewise RUL Clipping | `src/features.py` |
| **05. Cross-Val** | `(20631, 51)` | Splits by `unit_nr` | 5-Fold GroupKFold Engine Isolation | `src/train.py` |

---

## Phase 8: Experiment Tracking Log

| Exp ID | Model Architecture | Feature Matrix | Validation RMSE | Validation MAE | Key Takeaway / Technical Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | Random Forest (100 Trees) | Active Sensors + 10-cycle Rolling Stats | 19.33 cycles | 13.87 cycles | Solid baseline. Stable cross-validation variance ($\pm 0.49$). |
| **EXP-02** | XGBoost Regressor | Active Sensors + 10-cycle Rolling Stats | TBD | TBD | Testing sequential residual error optimization vs. bagging. |