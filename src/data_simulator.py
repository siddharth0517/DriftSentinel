import pandas as pd
import time

class FinancialDataStreamer:
    def __init__(self, file_path, chunk_size=100_000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        
        print(f"Initializing stream from {file_path}...")
        
        # CSV iterator (this is the key fix)
        self.chunk_iterator = pd.read_csv(
            file_path,
            chunksize=self.chunk_size
        )

    def get_next_batch(self, inject_drift=False):
        try:
            batch = next(self.chunk_iterator)
        except StopIteration:
            return None  # End of stream
        
        batch = batch.copy()

        # --- DRIFT INJECTION HOOK (future use) ---
        if inject_drift:
            # Example (keep commented for now)
            batch["amount"] *= 1.5
            pass

        print(f"Streamed batch with {len(batch)} rows (Drift={inject_drift})")
        return batch
