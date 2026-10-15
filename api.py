"""Root api module proxy to app.api."""
import sys
import app.api
sys.modules[__name__] = app.api
