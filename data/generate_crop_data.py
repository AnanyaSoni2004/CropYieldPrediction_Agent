"""Script to generate synthetic crop recommendation dataset."""
import pandas as pd
import numpy as np

np.random.seed(42)

# Crop profiles: (N_mean, N_std, P_mean, P_std, K_mean, K_std,
#                 temp_mean, temp_std, humidity_mean, humidity_std,
#                 ph_mean, ph_std, rainfall_mean, rainfall_std)
crop_profiles = {
    "rice":        (80, 10, 40, 5,  40, 5,  23, 3, 82, 5,  6.2, 0.3, 200, 30),
    "maize":       (85, 10, 58, 5,  41, 5,  22, 3, 65, 5,  6.2, 0.3, 100, 15),
    "chickpea":    (40,  8, 67, 5,  79, 8,  18, 3, 17, 4,  7.2, 0.3,  73, 10),
    "kidneybeans": (20,  5, 67, 5,  20, 5,  20, 3, 22, 4,  5.7, 0.3, 105, 15),
    "pigeonpeas":  (20,  5, 67, 5,  20, 5,  27, 3, 49, 5,  5.8, 0.3, 149, 20),
    "mothbeans":   (21,  5, 47, 5,  20, 5,  28, 3, 53, 5,  6.9, 0.3,  51, 10),
    "mungbean":    (20,  5, 47, 5,  19, 5,  28, 3, 86, 5,  6.4, 0.3,  49, 10),
    "blackgram":   (40,  8, 67, 5,  19, 5,  30, 3, 65, 5,  7.1, 0.3,  68, 10),
    "lentil":      (18,  5, 68, 5,  19, 5,  24, 3, 65, 5,  6.9, 0.3,  45, 8),
    "pomegranate": (18,  5, 18, 3,  40, 5,  21, 3, 90, 5,  6.0, 0.3, 108, 15),
    "banana":      (100,10, 82, 8, 50, 5,  27, 3, 80, 5,  6.0, 0.3, 105, 15),
    "mango":       (20,  5, 27, 4,  30, 5,  31, 3, 50, 5,  5.8, 0.3,  95, 15),
    "grapes":      (23,  5, 132,10, 200,15, 24, 3, 82, 5,  6.0, 0.3,  70, 10),
    "watermelon":  (99,  10,17, 3, 50, 5,  25, 3, 85, 5,  6.0, 0.3,  50, 8),
    "muskmelon":   (100,10, 17, 3, 50, 5,  29, 3, 92, 5,  6.4, 0.3,  25, 5),
    "apple":       (21,  5, 134,10,199,15, 22, 3, 92, 5,  5.9, 0.3, 113, 15),
    "orange":      (20,  5, 16, 3, 10, 3,  23, 3, 92, 5,  7.0, 0.3, 110, 15),
    "papaya":      (49,  8, 59, 5, 50, 5,  33, 3, 92, 5,  6.7, 0.3, 145, 20),
    "coconut":     (22,  5, 16, 3, 30, 5,  27, 3, 95, 5,  5.9, 0.3, 176, 25),
    "cotton":      (117,12, 46, 5, 19, 5,  23, 3, 80, 5,  6.6, 0.3,  80, 12),
    "jute":        (78,  10,46, 5, 39, 5,  25, 3, 80, 5,  6.7, 0.3, 174, 25),
    "coffee":      (101,10, 28, 4, 29, 5,  25, 3, 58, 5,  6.5, 0.3, 159, 20),
}

samples_per_crop = 100
rows = []

for crop, params in crop_profiles.items():
    (N_m, N_s, P_m, P_s, K_m, K_s,
     T_m, T_s, H_m, H_s, pH_m, pH_s, R_m, R_s) = params
    for _ in range(samples_per_crop):
        rows.append({
            "N":          max(0, np.random.normal(N_m, N_s)),
            "P":          max(0, np.random.normal(P_m, P_s)),
            "K":          max(0, np.random.normal(K_m, K_s)),
            "temperature":np.random.normal(T_m, T_s),
            "humidity":   np.clip(np.random.normal(H_m, H_s), 10, 100),
            "ph":         np.clip(np.random.normal(pH_m, pH_s), 3.5, 10),
            "rainfall":   max(0, np.random.normal(R_m, R_s)),
            "label":      crop,
        })

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("crop_data.csv", index=False)
print(f"Generated {len(df)} rows across {df['label'].nunique()} crops.")
print(df["label"].value_counts())
