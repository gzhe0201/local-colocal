import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def photon_carbon_comparison(path):
    dose = "1Gy"
    time = "2h"
    let = 120
    metric = "Lacunarity_Coloc_norm"
    """
    Compare Carbon, and Photon conditions for a given dose/time/LET and plot a bar graph.
    """
    df = pd.read_csv(path)

    # --- Subsets ---
    # Carbon and photon have defined dose/time
    carbon = df[
        (df["Carbon_or_Photon"] == "carbon") &
        (df["Radiation_Dose"] == dose) &
        (df["Time"].str.contains(time, case=False, na=False)) &
        (df["Carbon_LET"] == let)
    ]

    photon = df[
        (df["Carbon_or_Photon"] == "photon") &
        (df["Radiation_Dose"] == dose) &
        (df["Time"].str.contains(time, case=False, na=False))
    ]

    # --- Compute averages ---
    data = {
        f"Carbon ({let} LET)": carbon[metric].mean() if not carbon.empty else None,
        "Photon": photon[metric].mean() if not photon.empty else None
    }

    # --- Drop missing ones ---
    data = {k: v for k, v in data.items() if v is not None}

    if not data:
        print(f"⚠️ No matching data found for {dose}, {time}, LET={let}.")
        return None

    # --- Plot ---
    plt.figure(figsize=(6, 4))
    plt.bar(data.keys(), data.values(), color=["red", "blue"][:len(data)])
    plt.ylabel(metric.replace("_", " "))
    plt.title(f"{metric.replace('_', ' ')} — {dose}, {time}, {let} LET")
    plt.tight_layout()
    plt.show()

    print("✅ Data used for plot:", data)
    return data


def plot_metric_by_condition(csv_path):

    metric="Mean_Red_Coloc_Intensity"

    """
    Plot any metric across all combinations of Gy, LET, Time, and Carbon/Photon.

    Parameters
    ----------
    csv_path : str
        Path to Overall_Image_Summary.csv
    metric : str
        Column name for the y-axis (e.g., 'Total_Voxels', 'Num_Green_Puncta')
    """
    # --- Load data ---
    df = pd.read_csv(csv_path)

    # Clean up missing / text consistency
    for col in ["Radiation_Dose", "Time", "Carbon_LET"]:
        if col in df.columns:
            df[col] = df[col].fillna("N/A")

    # --- Combine metadata into a single "Condition" label ---
    def make_label(row):
        
        if row["Carbon_or_Photon"] == "baseline":
            return "Baseline"
        elif row["Carbon_or_Photon"] == "carbon":
            return f"Carbon {row['Radiation_Dose']}, {row['Time']}, LET{row['Carbon_LET']}"
        elif row["Carbon_or_Photon"] == "photon":
            return f"Photon {row['Radiation_Dose']}, {row['Time']}"
        else:
            return "Unknown"

    df["Condition"] = df.apply(make_label, axis=1)

    # --- Sort conditions by logical order ---
    df["Condition"] = pd.Categorical(df["Condition"], categories=sorted(df["Condition"].unique(), key=str))

    # --- Plot using seaborn for nice style ---
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="Condition", y=metric, hue="Carbon_or_Photon", ci="sd", errorbar="se")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric.replace("_", " "))
    plt.xlabel("Condition (Dose x Time x LET)")
    plt.title(f"{metric.replace('_', ' ')} across all conditions")
    plt.tight_layout()
    plt.show()

    return df[["Condition", metric, "Carbon_or_Photon", "Radiation_Dose", "Time", "Carbon_LET"]]

# === MAIN EXECUTION ===
if __name__ == "__main__":
    
    path = "C:/Users/m319725/Desktop/Test python/Overall_Image_Summary.csv"
    #data = photon_carbon_comparison(path)
    datum = plot_metric_by_condition(path)