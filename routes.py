import logging
import os
import datetime
import sys
from flask import Blueprint, request, jsonify, current_app, render_template
from werkzeug.exceptions import BadRequest, NotFound
from models import VoiceEmbedding
from utils import save_audio_file, extract_embedding, calculate_similarity
from app import db

# Configure logging for better debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a blueprint for API routes
api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/', methods=['GET'])
def index():
    """Render the API documentation page"""
    return render_template('index.html')

@api_bp.route('/enroll', methods=['POST'])
def enroll():
    """
    Endpoint to enroll a user's voice
    
    Expects:
    - user_id: A unique identifier for the user
    - audio: A WAV file containing the user's voice
    """
    try:
        # Check if required parameters are present
        if 'user_id' not in request.form:
            return jsonify({'success': False, 'error': 'Missing user_id parameter'}), 400
        
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
        
        user_id = request.form['user_id']
        audio_file = request.files['audio']
        
        # Save the audio file temporarily
        audio_path = save_audio_file(audio_file)
        if not audio_path:
            return jsonify({'success': False, 'error': 'Invalid audio file format. Only WAV and OPUS files are supported.'}), 400
        
        # Extract voice embedding
        embedding = extract_embedding(audio_path)
        if embedding is None:
            return jsonify({'success': False, 'error': 'Failed to extract voice embedding'}), 400
        
        # Read the audio file data for secure storage
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Get the file extension for format information
        audio_format = request.files['audio'].filename.rsplit('.', 1)[1].lower()
        
        # Use the new update_or_create class method
        VoiceEmbedding.update_or_create(
            user_id=user_id, 
            embedding_array=embedding,
            audio_data=audio_data,
            audio_format=audio_format
        )
        
        # Clean up the temporary file after successful processing
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logging.debug(f"Successfully removed temporary file: {audio_path}")
        except Exception as e:
            logging.error(f"Error removing temporary file after enrollment: {str(e)}")
        
        response = {'success': True, 'message': 'Voice profile enrolled successfully'}
        
        return jsonify(response), 200
        
    except Exception as e:
        logging.error(f"Error in enrollment: {str(e)}")
        # Rollback database session if there was an error
        db.session.rollback()
        
        # Clean up any temporary files in case of error
        if 'audio_path' in locals() and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logging.debug(f"Cleaned up temporary file after error: {audio_path}")
            except Exception as file_error:
                logging.error(f"Failed to remove temporary file after error: {str(file_error)}")
        
        return jsonify({'success': False, 'error': f'Internal server error: {str(e)}'}), 500

@api_bp.route('/verify', methods=['POST'])
def verify():
    """
    Endpoint to verify a user's voice
    
    Expects:
    - user_id: The unique identifier for the user to verify
    - audio: A WAV file containing the voice to verify
    
    Returns:
    - match: Boolean indicating if the voice matches
    - similarity: A score between 0 and 1 indicating the similarity
    """
    try:
        # Check if required parameters are present
        if 'user_id' not in request.form:
            return jsonify({'success': False, 'error': 'Missing user_id parameter'}), 400
        
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
        
        user_id = request.form['user_id']
        audio_file = request.files['audio']
        
        # Check if user's enrollment voice exists - always compare with enrollment voice
        enrolled_voice = VoiceEmbedding.query.filter_by(user_id=user_id, is_enrollment=True).first()
        if not enrolled_voice:
            return jsonify({'success': False, 'error': 'User not enrolled, please enroll first'}), 404
        
        # Log the found enrollment for debugging
        logging.debug(f"Found enrollment for user_id={user_id}, created at {enrolled_voice.created_at}")
        
        # Save the audio file temporarily
        audio_path = save_audio_file(audio_file)
        if not audio_path:
            return jsonify({'success': False, 'error': 'Invalid audio file format. Only WAV and OPUS files are supported.'}), 400
        
        # Extract voice embedding from the provided sample
        new_embedding = extract_embedding(audio_path)
        if new_embedding is None:
            return jsonify({'success': False, 'error': 'Failed to extract voice embedding'}), 400
        
        # Get the stored embedding for the user (from enrollment, not previous verification)
        stored_embedding = enrolled_voice.get_embedding()
        
        # Log embedding shapes for debugging
        logging.debug(f"Verification embedding shape: {new_embedding.shape}")
        logging.debug(f"Enrolled embedding shape: {stored_embedding.shape}")
        
        # Calculate similarity between the new verification embedding and the original enrollment embedding
        similarity = calculate_similarity(stored_embedding, new_embedding)
        logging.debug(f"Calculated similarity: {similarity}")
        
        # Determine if it's a match based on the threshold
        threshold = current_app.config['SIMILARITY_THRESHOLD']
        is_match = similarity >= threshold
        logging.debug(f"Match result: {is_match} (threshold: {threshold})")
        
        # Read the audio file data for secure storage of verification attempt
        with open(audio_path, "rb") as audio_file_data:
            audio_data = audio_file_data.read()
        
        # Get the file extension for format information
        audio_format = request.files['audio'].filename.rsplit('.', 1)[1].lower()
        
        # Use a timestamp in the user_id to make each verification record unique
        # This way we can store multiple verification attempts for the same user
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        verification_id = f"{user_id}_verify_{timestamp}"
        
        # Store verification attempt in database with is_enrollment=False and unique ID
        verification_record = VoiceEmbedding(
            user_id=verification_id,  # Use a unique ID for each verification 
            embedding_array=new_embedding,
            audio_data=audio_data,
            audio_format=audio_format,
            is_enrollment=False  # Mark as verification attempt, not enrollment
        )
        db.session.add(verification_record)
        db.session.commit()
        
        # Clean up the temporary file after successful processing
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logging.debug(f"Successfully removed temporary file: {audio_path}")
        except Exception as e:
            logging.error(f"Error removing temporary file after verification: {str(e)}")
        
        # Calculate similarity percentage for easier user understanding
        similarity_percentage = round(similarity * 100, 2)
        threshold_percentage = round(threshold * 100, 2)
        
        # Create interpretation based on similarity score with our new content-adaptive ranges
        interpretation = "Unknown"
        if similarity < 0.2:
            interpretation = "Definitely different speakers"
        elif similarity < 0.4:
            interpretation = "Very likely different speakers"
        elif similarity < 0.55:
            interpretation = "Possibly different speakers (or same speaker with very different content)"
        elif similarity < 0.75:  # Changed from 0.8 to better indicate confidence levels
            interpretation = "Likely same speaker (even with different content)"
        else:
            interpretation = "Very likely same speaker (high confidence)"
            
        response = {
            'success': True,
            'match': is_match,
            'similarity': similarity,
            'similarity_percentage': similarity_percentage,
            'threshold': threshold,
            'threshold_percentage': threshold_percentage,
            'user_id': user_id,
            'interpretation': interpretation,
            'details': {
                'enrolled_at': enrolled_voice.created_at.isoformat(),
                'threshold_description': {
                    '0.0 - 0.2': 'Definitely different speakers',
                    '0.2 - 0.4': 'Very likely different speakers',
                    '0.4 - 0.55': 'Possibly different speakers (or same speaker with very different content)',
                    '0.55 - 0.75': 'Likely same speaker (even with different content)',
                    '0.75 - 1.0': 'Very likely same speaker (high confidence)'
                }
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logging.error(f"Error in verification: {str(e)}")
        
        # Clean up any temporary files in case of error
        if 'audio_path' in locals() and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logging.debug(f"Cleaned up temporary file after verification error: {audio_path}")
            except Exception as file_error:
                logging.error(f"Failed to remove temporary file after verification error: {str(file_error)}")
        
        return jsonify({'success': False, 'error': f'Internal server error: {str(e)}'}), 500

@api_bp.route('/audio/<user_id>', methods=['GET'])
def get_audio(user_id):
    """
    Endpoint to retrieve a user's stored audio file
    
    Expects:
    - user_id: The unique identifier for the user
    
    Returns:
    - Audio file if found
    """
    try:
        # Check if user's enrollment audio exists
        enrolled_voice = VoiceEmbedding.query.filter_by(user_id=user_id, is_enrollment=True).first()
        if not enrolled_voice or not enrolled_voice.audio_data:
            return jsonify({'success': False, 'error': 'Enrollment audio not found for this user'}), 404
        
        # Return the audio file with appropriate content type
        from flask import send_file, Response
        import io
        
        audio_format = enrolled_voice.audio_format
        mimetype = 'audio/wav' if audio_format == 'wav' else 'audio/ogg' if audio_format == 'opus' else 'application/octet-stream'
        
        return Response(
            io.BytesIO(enrolled_voice.audio_data),
            mimetype=mimetype,
            headers={
                'Content-Disposition': f'attachment; filename="{user_id}.{audio_format}"'
            }
        )
        
    except Exception as e:
        logging.error(f"Error retrieving audio: {str(e)}")
        return jsonify({'success': False, 'error': f'Internal server error: {str(e)}'}), 500

# Error handlers
@api_bp.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify({'success': False, 'error': str(e)}), 400

@api_bp.errorhandler(NotFound)
def handle_not_found(e):
    return jsonify({'success': False, 'error': 'Resource not found'}), 404

@api_bp.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {str(e)}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500
