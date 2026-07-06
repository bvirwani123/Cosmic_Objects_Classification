import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  
import os

# Set page title layout
st.title("Cosmic Objects Classification")

# --- MODEL LOADING LOGIC ---
model_filename = "cosmic_objects_rf_pipeline.pkl"

if os.path.exists(model_filename):
    try:
        model = joblib.load(model_filename)
    except Exception as e:
        st.error(f"💥 Failed to load model. Error detail: {e}")
else:
    st.error(f"❌ Error: File `{model_filename}` cannot be found.")

# --- CHART AND UI DISPLAY ---
st.subheader("Data Insights: Class Distributions")

# 1. Read and parse the target table from the shared notebook cell
data = {"class": ["GALAXY", "STAR", "QSO"], "count":}
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

plt.ylim(0, df_counts["count"].max() * 1.15)
plt.tight_layout()

# 6. RENDER PLOT ON WEB PAGE (Fixed Line)
st.pyplot(plt)
