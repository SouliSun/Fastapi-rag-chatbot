"""Root ingest module proxy to app.ingest."""
import sys
import app.ingest
sys.modules[__name__] = app.ingest

if __name__ == "__main__":
    app.ingest.main()