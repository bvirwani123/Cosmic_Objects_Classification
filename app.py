import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # or import pickle
import os

# --- STEP 1: INITIAL VERIFICATION ---
st.write("✨ App script initialized successfully!")

# --- STEP 2: CHECK FILE EXISTENCE & SIZE ---
model_filename = "cosmic_objects_rf_pipeline.pkl"  # ⚠️ Change this to your exact pickle filename!

if os.path.exists(model_filename):
    file_size_mb = os.path.getsize(model_filename) / (1024 * 1024)
    st.write(f"📂 File detected! Detected size: `{file_size_mb:.2f} MB`")
    
    # CRITICAL CHECK FOR GIT LFS POINTER BUG:
    if file_size_mb < 0.01:
        st.error("🚨 BUG DETECTED: Streamlit is reading a 130-byte Git LFS pointer text file instead of the actual 79 MB model file.")
else:
    st.error(f"❌ ERROR: File `{model_filename}` cannot be found in this directory.")

# --- STEP 3: TRY LOADING THE MODEL ---
st.write("🔄 Attempting to load the ML model into memory...")

try:
    # This is the line where your code is currently freezing:
    model = joblib.load(model_filename)  # Or pickle.load(open(model_filename, "rb"))
    st.write("✅ SUCCESS! Model fully loaded into RAM.")
except Exception as e:
    st.error(f"💥 LOADING FAILED: {e}")

# --- REST OF YOUR APP CODE ---
st.title("Cosmic Objects Classification")
# (Your inputs, buttons, and prediction code continue down here...)



import streamlit as st
st.write("The app is starting up successfully!")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Read and parse the target table from the shared notebook cell
data = {"class": ["GALAXY", "STAR", "QSO"], "count": [59445, 21594, 18961]}

df_counts = pd.DataFrame(data)

# Calculate percentages for proper annotations
total_counts = df_counts["count"].sum()
df_counts["percentage"] = (df_counts["count"] / total_counts) * 100

# 2. Plotting Configuration
plt.figure(figsize=(10, 6), dpi=100)

# Set the style used in the notebook
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 18,
        "patch.edgecolor": "none",
    }
)

# 3. Create the Bar Plot
# Custom color palette aligning with astronomical color schemes
palette = ["#2b5c8f", "#e69f00", "#8b4513"]

ax = sns.barplot(
    data=df_counts,
    x="class",
    y="count",
    hue="class",
    palette=palette,
    legend=False,
)

# 4. Add Proper Annotations (Values and Percentages)
for p in ax.patches:
    count = int(p.get_height())
    percentage = (count / total_counts) * 100
    annotation_text = f"{count:,}\n({percentage:.1f}%)"

    # Place text dynamically over each bar
    ax.annotate(
        annotation_text,
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 15),
        textcoords="offset points",
        fontweight="bold",
        fontsize=12,
    )

# 5. Labels and Titles Setup
plt.title("Distribution of Cosmic Object Classes", pad=25, fontweight="bold")
plt.xlabel("Cosmic Object Class", labelpad=15)
plt.ylabel("Observation Count", labelpad=15)

# Adjust y-limit dynamically to prevent layout clippings from top annotations
plt.ylim(0, df_counts["count"].max() * 1.15)
plt.tight_layout()

# 6. Render Plot
plt.show()