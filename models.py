from app import db
import numpy as np
import json
import datetime
from sqlalchemy.dialects.postgresql import BYTEA

class VoiceEmbedding(db.Model):
    """Model for storing enrollment voice embeddings and audio files"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(64), nullable=False, index=True)
    embedding = db.Column(db.Text, nullable=False)  # Stored as JSON string
    audio_data = db.Column(BYTEA, nullable=True)  # Binary audio data storage
    audio_format = db.Column(db.String(10), nullable=True)  # Format of the stored audio (wav, opus)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, 
                          default=datetime.datetime.utcnow,
                          onupdate=datetime.datetime.utcnow)
    # Flag to indicate this is an enrollment embedding, not a verification embedding
    is_enrollment = db.Column(db.Boolean, default=True, nullable=False)
    
    # Add a composite unique constraint - only one enrollment per user_id
    __table_args__ = (
        db.UniqueConstraint('user_id', 'is_enrollment', name='uix_user_enrollment'),
    )
    
    def __init__(self, user_id, embedding_array, audio_data=None, audio_format=None, is_enrollment=True):
        self.user_id = user_id
        self.set_embedding(embedding_array)
        self.audio_data = audio_data
        self.audio_format = audio_format
        self.is_enrollment = is_enrollment
    
    def set_embedding(self, embedding_array):
        """Convert numpy array to JSON string for storage"""
        if isinstance(embedding_array, np.ndarray):
            self.embedding = json.dumps(embedding_array.tolist())
        else:
            self.embedding = json.dumps(embedding_array)
    
    def get_embedding(self):
        """Convert stored JSON string back to numpy array"""
        return np.array(json.loads(self.embedding))
    
    def __repr__(self):
        return f'<VoiceEmbedding {self.user_id}>'
    
    @classmethod
    def update_or_create(cls, user_id, embedding_array, audio_data=None, audio_format=None):
        """Update an existing enrollment embedding or create a new one"""
        # Always look for an enrollment embedding
        embedding = cls.query.filter_by(user_id=user_id, is_enrollment=True).first()
        if embedding:
            embedding.set_embedding(embedding_array)
            if audio_data:
                embedding.audio_data = audio_data
                embedding.audio_format = audio_format
            embedding.updated_at = datetime.datetime.utcnow()
        else:
            embedding = cls(user_id, embedding_array, audio_data, audio_format, is_enrollment=True)
            db.session.add(embedding)
        db.session.commit()
        return embedding
