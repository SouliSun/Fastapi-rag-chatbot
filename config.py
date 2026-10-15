"""Root config module proxy to app.config."""
import sys
import app.config
sys.modules[__name__] = app.config
