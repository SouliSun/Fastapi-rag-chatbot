"""Root calibrate_threshold module proxy to scripts.calibrate_threshold."""
import sys
import scripts.calibrate_threshold
sys.modules[__name__] = scripts.calibrate_threshold

if __name__ == "__main__":
    scripts.calibrate_threshold.main()
