import os
import tempfile

# App configuration
DEBUG = True
SECRET_KEY = os.environ.get("SESSION_SECRET", "default_secret_key")

# Database configuration
DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///voice_biometric.db')

# Audio processing configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'wav', 'opus'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size

# Voice verification configuration
# Completely revised voice biometric system with better voice discrimination
# This threshold is calibrated for our new content-adaptive voice similarity algorithm
# Increased threshold for stricter verification while still handling content variation
SIMILARITY_THRESHOLD = 0.75  # Higher threshold to improve security and reduce false positives

# New content-adaptive similarity scale interpretation (reference)
# 0.0 - 0.2: Definitely different speakers
# 0.2 - 0.4: Very likely different speakers
# 0.4 - 0.55: Possibly different speakers (or same speaker with very different content)
# 0.55 - 0.75: Likely same speaker (even with different content)
# 0.75 - 1.0: Very likely same speaker (high confidence)
