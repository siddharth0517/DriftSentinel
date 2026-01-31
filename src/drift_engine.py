import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

class DriftDetector:
    def __init__(self):
        self.reference_profile = None
        self.reference_batch = None
        
    def set_reference(self, batch_df):
        """
        Locks the 'Normal' behavior based on the first batch.
        We store two things:
        1. Basic Stats (Mean, Std) for fast checking.
        2. The actual column data (optional, but needed for advanced tests like KS-Test).
        """
        print("🔒 Locking Reference Profile...")
        
        # We only care about numerical columns for now
        numerical_cols = batch_df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.reference_profile = {}
        self.reference_batch = batch_df[numerical_cols].copy() # Keep a copy for statistical tests
        
        for col in numerical_cols:
            self.reference_profile[col] = {
                "mean": batch_df[col].mean(),
                "std": batch_df[col].std(),
                "min": batch_df[col].min(),
                "max": batch_df[col].max()
            }
            
        print(f"Reference locked with {len(numerical_cols)} numerical features.")
        return self.reference_profile

    def detect_drift(self, current_batch_df):
        if self.reference_profile is None:
            raise ValueError("Reference not set!")

        drift_report = {}
        numerical_cols = current_batch_df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numerical_cols:
            if col not in self.reference_batch.columns:
                continue

            # --- METHOD 1: Z-Score (Simple Mean Shift) ---
            ref_stats = self.reference_profile[col]
            curr_mean = current_batch_df[col].mean()
            std = ref_stats['std'] if ref_stats['std'] > 0 else 1e-6
            z_score = abs(curr_mean - ref_stats['mean']) / std

            # --- METHOD 2: KS Test (Distribution Shape) ---
            # This compares the ACTUAL data arrays, not just the mean
            ref_data = self.reference_batch[col]
            curr_data = current_batch_df[col]
            
            # ks_2samp returns (statistic, p-value)
            # statistic is the "Distance" between distributions (0 to 1)
            ks_stat, p_value = ks_2samp(ref_data, curr_data)

            # --- DECISION LOGIC ---
            # We flag drift if EITHER the mean shifted OR the shape changed
            is_drifting = z_score > 0.5 or ks_stat > 0.1 # Thresholds (tune these)

            drift_report[col] = {
                "z_score": round(z_score, 4),
                "ks_statistic": round(ks_stat, 4), # 0.0 = same, 1.0 = different
                "p_value": round(p_value, 4),      # Low p-value (<0.05) means "Statistically Significant"
                "is_drifting": is_drifting
            }

        return drift_report