from data_simulator import FinancialDataStreamer
from drift_engine import DriftDetector

# Initialize
streamer = FinancialDataStreamer("data/PS_20174392719_1491204439457_log.csv")
detector = DriftDetector()

# 1. Get Reference Batch (Day 0)
reference_batch = streamer.get_next_batch(inject_drift=True)
detector.set_reference(reference_batch)

print("\n--- STARTING LIVE MONITORING ---\n")

# 2. Simulate Production (Next 3 batches)
for i in range(3):
    print(f"Checking Batch {i+1}...")
    current_batch = streamer.get_next_batch(inject_drift=False) # No drift yet
    
    report = detector.detect_drift(current_batch)
    
    # Just print the 'amount' column to keep output clean
    if 'amount' in report:
        stats = report['amount']
        print(f"  [Amount] Z-Score: {stats['z_score']} | KS-Stat: {stats['ks_statistic']} | Status: {'🔴 DRIFT' if stats['is_drifting'] else '🟢 OK'}")
    print("-" * 30)