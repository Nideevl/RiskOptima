from riskoptima import db, lapse_model as L

conn = db.build()
r = L.run(conn)
m = r["metrics"]
print(f"Train: {r['train_rows']:,} policy-years from {r['train_policies']:,} policies | "
      f"Test: {r['test_rows']:,} policy-years from {r['test_policies']:,} held-out policies | base lapse rate {m['base_rate']:.4f}")
print(f"\n{'':24s}{'AUC':>8s}{'log-loss':>10s}")
print(f"{'gradient boosting':24s}{m['auc']:8.4f}{m['logloss']:10.5f}")
print(f"{'policy-year lookup':24s}{m['auc_year_baseline']:8.4f}{m['logloss_year_baseline']:10.5f}")
print(f"{'oracle (true probs)':24s}{m['auc_oracle']:8.4f}{m['logloss_oracle']:10.5f}")
print(f"\nBrier score {m['brier']:.5f}")
print("\nCalibration by decile of predicted lapse")
print(m["calibration"].round(4).to_string(index=False))
print("\nPermutation importance (drop in held-out AUC)")
print(r["importance"].round(4).to_string(index=False))
