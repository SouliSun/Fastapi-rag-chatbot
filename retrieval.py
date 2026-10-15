"""Root retrieval module proxy to app.retrieval."""
import sys
import app.retrieval
sys.modules[__name__] = app.retrieval
