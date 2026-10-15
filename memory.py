"""Root memory module proxy to app.memory."""
import sys
import app.memory
sys.modules[__name__] = app.memory
