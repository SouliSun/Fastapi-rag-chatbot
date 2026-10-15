"""Root generation module proxy to app.generation."""
import sys
import app.generation
sys.modules[__name__] = app.generation
