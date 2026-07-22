"""
E-MR mini-rig tooling: capture + analysis pipeline for the real-silicon lead-time
experiment (see thermalos-vault wiki/experiments/E-MR.md).

Built AHEAD of the hardware (the E-LT precedent: build the whole analysis so that
when the rig is assembled, the only new thing is turning the degradation knob). The
analysis core reuses the SHIPPED detector + calibration modules (theta.agent), so the
lab and the product share one implementation and a rig-validated result maps straight
onto the running agent.

Modules:
  throttle.py            -- NVML clock-reason bitmask -> thermal-throttle event (t_throttle)
  analyze.py             -- RunTrace -> lead_time = t_throttle - t_anomaly (shipped CUSUM)
  calibrate_from_runs.py -- labeled run-to-throttle runs -> calibration + survival RUL fit
  colab_capture.py       -- log the full NVML surface to CSV (run on Colab/host today)
"""
