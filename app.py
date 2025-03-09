import json
import os
import tempfile
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import openai
import streamlit as st
import torch
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pydub import AudioSegment
from st_audiorec import st_audiorec

# Load environment variables
load_dotenv()

# Set up Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT or ""

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", "")

API_VERSION = os.getenv("API_VERSION", "2023-05-15")
os.environ["API_VERSION"] = API_VERSION

# Configure OpenAI client for Azure
if AZURE_OPENAI_ENDPOINT and os.environ.get("AZURE_OPENAI_API_KEY"):
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    openai.api_type = "azure"
    openai.api_version = API_VERSION
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-35-turbo")

# Model cache to avoid reloading models
@st.cache_resource
def get_whisper_model(model_size, device, compute_type):
    """
    Load and cache the Whisper model for faster subsequent use
    
    Args:
        model_size (str): Size of the model to load (tiny, base, small, medium, large)
        device (str): Device to use for computation
        compute_type (str): Compute type for the model
        
    Returns:
        WhisperModel: Loaded model instance
    """
    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=tempfile.gettempdir()  # Use temp dir to prevent permission issues
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise

def format_timestamp(seconds):
    """Convert seconds to hh:mm:ss format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def process_large_audio(audio_path, model, chunk_duration=600, language="ko", progress_bar=None):
    """
    Process large audio files by splitting them into manageable chunks
    
    Args:
        audio_path (str): Path to the audio file
        model: WhisperModel instance
        chunk_duration (int): Duration of each chunk in seconds
        language (str): Language code
        progress_bar: Streamlit progress bar
        
    Returns:
        list: List of transcription segments
    """
    import os
    import tempfile

    import librosa
    import numpy as np
    from pydub import AudioSegment
    
    try:
        # Get audio duration using librosa (handles various formats)
        audio_duration = librosa.get_duration(path=audio_path)
        st.write(f"Audio duration: {format_timestamp(audio_duration)}")
        
        # If audio is small enough, process it directly
        if audio_duration <= chunk_duration:
            segments, info = model.transcribe(
                audio_path,
                beam_size=5,
                language=language,
                condition_on_previous_text=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            return list(segments)
        
        # For large files, process in chunks
        all_segments = []
        
        # Load audio using pydub (handles more formats than librosa for manipulation)
        audio = AudioSegment.from_file(audio_path)
        
        # Calculate total chunks
        total_chunks = int(audio_duration / chunk_duration) + (1 if audio_duration % chunk_duration > 0 else 0)
        
        if progress_bar:
            progress_bar.progress(0, text="Preparing chunks...")
        
        # Process each chunk
        for i in range(total_chunks):
            start_ms = i * chunk_duration * 1000
            end_ms = min((i + 1) * chunk_duration * 1000, len(audio))
            
            # Extract chunk
            chunk = audio[start_ms:end_ms]
            
            # Save chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                chunk_path = temp_file.name
                chunk.export(chunk_path, format="wav")
            
            # Update progress
            if progress_bar:
                progress_bar.progress((i / total_chunks) * 0.9, 
                                    text=f"Transcribing chunk {i+1}/{total_chunks}...")
            
            # Process chunk
            chunk_segments, info = model.transcribe(
                chunk_path,
                beam_size=5,
                language=language,
                condition_on_previous_text=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Adjust timestamps and add to results
            time_offset = i * chunk_duration
            for segment in chunk_segments:
                segment.start += time_offset
                segment.end += time_offset
                all_segments.append(segment)
            
            # Clean up temp file
            os.unlink(chunk_path)
        
        if progress_bar:
            progress_bar.progress(1.0, text="Transcription complete!")
            
        return all_segments
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        raise

def transcribe_audio(audio_path, output_path=None, model_size="base", device="cuda" if torch.cuda.is_available() else "cpu", include_timestamps=False):
    """
    Transcribe audio to text using faster-whisper with support for large files
    
    Args:
        audio_path (str): Path to the audio file
        output_path (str, optional): Path to save the transcription
        model_size (str): Whisper model size to use (tiny, base, small, medium, large)
        device (str): Device to use for computation ("cuda" for GPU, "cpu" for CPU)
        include_timestamps (bool): Whether to include timestamps in the output
    
    Returns:
        tuple: (transcription_text, segments_data)
    """
    # Create a progress bar
    progress_text = "Initializing model..."
    progress_bar = st.progress(0, text=progress_text)
    
    try:
        # Determine compute type based on device
        compute_type = "float16" if device == "cuda" else "int8"
        
        # Load the model using cached function
        progress_bar.progress(0.1, text=f"Loading Whisper model ({model_size})...")
        model = get_whisper_model(model_size, device, compute_type)
        
        progress_bar.progress(0.2, text="Processing audio...")
        
        # Process audio (with chunking for large files)
        segments_list = process_large_audio(
            audio_path=audio_path, 
            model=model, 
            language="ko",
            progress_bar=progress_bar
        )
        
        progress_bar.progress(0.95, text="Formatting results...")
        
        # Create transcription text based on user preference
        transcription_with_timestamps = ""
        transcription_clean = ""
        segments_data = []
        
        for segment in segments_list:
            # Save both formats
            timestamp = f"[{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}] "
            transcription_with_timestamps += timestamp + segment.text + "\n"
            transcription_clean += segment.text + "\n"
            
            # Store segment data for generating meeting notes
            segments_data.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
        
        # Choose which format to return based on user preference
        transcription = transcription_with_timestamps if include_timestamps else transcription_clean
        
        # Save to file if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcription)
            st.success(f"Transcription saved to {output_path}")
        
        progress_bar.progress(1.0, text="Transcription complete!")
        return transcription, segments_data
        
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        progress_bar.empty()
        raise

def generate_meeting_notes(segments_data):
    """
    Generate meeting notes from transcription using Azure OpenAI
    
    Args:
        segments_data (list): List of transcription segments with timestamps and text
    
    Returns:
        str: Generated meeting notes
    """
    if not AZURE_OPENAI_ENDPOINT or not os.environ.get("AZURE_OPENAI_API_KEY"):
        return "Azure OpenAI credentials not configured. Please set environment variables."
    
    try:
        # Prepare the transcription as context - use only text without timestamps for meeting notes
        full_transcript = "\n".join([seg['text'] for seg in segments_data])
        
        # Create the prompt
        prompt = f"""You are an expert meeting note summarizer. Below is a transcript of a meeting in Korean.
        Please create comprehensive meeting notes with the following sections:
        1. Summary of the meeting (simple paragraph)
        2. Key discussion points (bulleted list)
        3. Action items and owners (if any)
        4. Decisions made
        5. Follow-up tasks
        
        Make the notes professional, clear, and well-organized. Format the output in markdown.
        
        Here's the meeting transcript:
        
        {full_transcript}
        """
        
        # Call Azure OpenAI
        response = openai.ChatCompletion.create(
            engine=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an AI assistant that creates professional meeting notes from transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=2000
        )
        
        meeting_notes = response.choices[0].message.content
        return meeting_notes
        
    except Exception as e:
        st.error(f"Error generating meeting notes: {str(e)}")
        return f"Failed to generate meeting notes: {str(e)}"

def main():
    st.set_page_config(page_title="Korean Meeting Transcription & Notes", layout="wide")
    
    st.title("Korean Speech-to-Text & Meeting Notes Generator")
    st.write("Upload a Korean audio file or use a file path to transcribe and generate meeting notes")
    
    # Set Streamlit to handle larger file uploads (this won't affect server limits)
    # The actual limit is controlled by the server configuration
    st.markdown(f"""
        <style>
            /* Increase the maximum width to accommodate more content */
            .reportview-container .main .block-container {{
                max-width: 1200px;
            }}
            
            /* Make text areas larger */
            textarea {{
                min-height: 300px;
            }}
        </style>
    """, unsafe_allow_html=True)
    
    # Check Azure OpenAI configuration
    azure_configured = AZURE_OPENAI_ENDPOINT and os.environ.get("AZURE_OPENAI_API_KEY")
    if not azure_configured:
        st.warning("⚠️ Azure OpenAI is not configured. Meeting notes generation will not be available. Please set the environment variables.")
    
    # Model settings
    col1, col2 = st.columns(2)
    with col1:
        model_size = st.selectbox(
            "Select Whisper model size",
            options=["tiny", "base", "small", "medium", "large-v2"],
            index=1,  # Default to "base"
            help="Larger models are more accurate but slower. For long recordings, 'base' or 'small' is recommended for balance."
        )
    
    with col2:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_display = st.selectbox(
            "Processing device",
            options=["GPU", "CPU"],
            index=0 if device == "cuda" else 1,
            disabled=True,
            help=f"Using {'GPU' if device == 'cuda' else 'CPU'} for processing"
        )
    
    # Tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["File Path", "Upload File", "Record Audio"])
    
    with tab1:
        # Fixed file path option (default tab)
        file_path = st.text_input(
            "Enter the path to the audio file",
            value="/data/chats/5qwtx/workspace/uploads/recorded_audio.mp3"
        )
        output_path = st.text_input(
            "Enter the path for the output text file",
            value="/data/chats/5qwtx/workspace/uploads/output_text.txt"
        )
        notes_output = st.text_input(
            "Enter the path for meeting notes output (optional)",
            value="/data/chats/5qwtx/workspace/uploads/meeting_notes.md",
            help="Leave blank if you don't want to save meeting notes to a file"
        )
    
    with tab2:
        # File upload option
        uploaded_file = st.file_uploader("Upload an audio file (mp3, wav, m4a, etc.)", 
                                       type=["mp3", "wav", "m4a", "ogg", "flac"])
        st.info("Note: There may be server-side upload limits for large files. If you encounter errors with large files, use the File Path tab instead.")
    
    with tab3:
        # Audio recording using streamlit-audiorec
        st.subheader("Record Audio")
        st.write("Click the microphone button below to start recording your audio directly in the browser.")
        
        # Get audio data from the recorder component
        wav_audio_data = st_audiorec()
        
        if wav_audio_data is not None:
            # Display success and audio player
            st.success("✅ Recording complete! Click play below to preview.")
            st.audio(wav_audio_data, format='audio/wav')
            
            # Save the recorded audio
            if 'recorded_file_path' not in st.session_state:
                st.session_state.recorded_file_path = None
            
            # Save to temporary file for processing
            temp_dir = Path(tempfile.gettempdir())
            recorded_file_path = temp_dir / f"recorded_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            
            # Write the WAV bytes to file
            with open(recorded_file_path, "wb") as f:
                f.write(wav_audio_data)
            
            st.session_state.recorded_file_path = recorded_file_path
            
            # Display file details
            try:
                audio_segment = AudioSegment.from_wav(recorded_file_path)
                duration_seconds = len(audio_segment) / 1000
                st.write(f"Recording length: {duration_seconds:.2f} seconds")
                st.write(f"Sample rate: {audio_segment.frame_rate} Hz")
            except Exception as e:
                st.write("Audio file saved successfully")
            
            # Add transcribe button directly in the recording tab
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Transcribe Recording"):
                    with st.spinner('Transcribing...'):
                        try:
                            # Use the recorded file path for transcription
                            transcription, segments_data = transcribe_audio(
                                audio_path=str(recorded_file_path),
                                model_size=model_size,
                                device=device,
                                include_timestamps=include_timestamps
                            )
                            st.success("✨ Transcription complete!")
                            st.text_area("Korean Transcript:", transcription, height=150)
                            
                            # Generate meeting notes if requested and Azure is configured
                            if generate_notes and azure_configured:
                                with st.spinner('Generating meeting notes...'):
                                    meeting_notes = generate_meeting_notes(segments_data)
                                    st.markdown(meeting_notes)
                        except Exception as e:
                            st.error(f"Error during transcription: {str(e)}")
            
            with col2:
                # Add a button to clear the recording
                if st.button("Reset Recording"):
                    st.session_state.recorded_file_path = None
                    st.experimental_rerun()
        
        else:
            st.info("👆 Click the microphone button above to start recording")
            st.write("Note: For best results, record in a quiet environment and use a good microphone.")
        

    
    # Add chunking options for large files
    with st.expander("Advanced Options"):
        chunk_duration = st.slider(
            "Chunk duration (seconds)",
            min_value=60,  # 1 minute
            max_value=900,  # 15 minutes
            value=600,     # Default 10 minutes
            step=60,
            help="For large files, the audio will be processed in chunks of this duration"
        )
        
        include_timestamps = st.checkbox(
            "Include timestamps in transcript", 
            value=False,
            help="If checked, timestamps will be included in the transcript in format [HH:MM:SS --> HH:MM:SS]"
        )
        
        generate_notes = st.checkbox(
            "Generate meeting notes with Azure OpenAI", 
            value=True,
            disabled=not azure_configured,
            help="Requires Azure OpenAI configuration"
        )

    # Process file
    if st.button("Transcribe Audio"):
        with st.spinner("Processing..."):
            try:
                # Determine audio path based on input method
                using_file_path = True  # Default to file path tab
                
                if st.session_state.get('active_tab') == "Upload File" and uploaded_file is not None:
                    # Save uploaded file temporarily with a unique name
                    temp_dir = Path(tempfile.gettempdir())
                    audio_path = temp_dir / f"uploaded_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{uploaded_file.name.split('.')[-1]}"
                    
                    # Write in chunks to avoid memory issues
                    with open(audio_path, "wb") as f:
                        for chunk in uploaded_file.chunks(chunk_size=5*1024*1024):  # 5MB chunks
                            f.write(chunk)
                            
                    using_file_path = False
                    temp_output_path = str(temp_dir / "transcript_output.txt")
                
                # Handle recorded audio if available
                elif st.session_state.get('recorded_file_path') and os.path.exists(st.session_state.recorded_file_path):
                    audio_path = st.session_state.recorded_file_path
                    temp_dir = Path(tempfile.gettempdir())
                    temp_output_path = str(temp_dir / "transcript_output.txt")
                    using_file_path = False
                    st.session_state.active_tab = "Record Audio"
                    
                elif os.path.exists(file_path):
                    audio_path = file_path
                    temp_output_path = output_path
                else:
                    st.error(f"File not found: {file_path}")
                    return
                
                # Perform transcription
                transcription, segments_data = transcribe_audio(
                    audio_path=str(audio_path),
                    output_path=temp_output_path,
                    model_size=model_size,
                    device=device,
                    include_timestamps=include_timestamps
                )
                
                # Generate meeting notes if requested and Azure is configured
                if generate_notes and azure_configured:
                    st.write("Generating meeting notes...")
                    meeting_notes = generate_meeting_notes(segments_data)
                    
                    # Display meeting notes
                    st.subheader("Meeting Notes")
                    st.markdown(meeting_notes)
                    
                    # Save meeting notes if path provided
                    if notes_output and notes_output.strip():
                        os.makedirs(os.path.dirname(notes_output) or '.', exist_ok=True)
                        with open(notes_output, "w", encoding="utf-8") as f:
                            f.write(meeting_notes)
                        st.success(f"Meeting notes saved to {notes_output}")
                    
                    # Also provide download option
                    if not using_file_path:
                        temp_notes_path = str(temp_dir / "meeting_notes.md")
                        with open(temp_notes_path, "w", encoding="utf-8") as f:
                            f.write(meeting_notes)
                        
                        with open(temp_notes_path, "rb") as f:
                            st.download_button(
                                label="Download Meeting Notes",
                                data=f,
                                file_name="meeting_notes.md",
                                mime="text/markdown"
                            )
                
                # Display transcription results
                st.subheader("Transcription Result")
                st.text_area("Text output", transcription, height=400)
                
                # Provide download options for uploaded files
                if not using_file_path:
                    with open(temp_output_path, "rb") as f:
                        st.download_button(
                            label="Download Transcription",
                            data=f,
                            file_name="transcription.txt",
                            mime="text/plain"
                        )
                    
                    # Clean up temp files
                    # Don't delete recorded audio file if it's from the recording tab
                    if not using_file_path and st.session_state.get('active_tab') != "Record Audio":
                        os.unlink(audio_path)
                    if not using_file_path:
                        os.unlink(temp_output_path)
            
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
    
    st.markdown("---")
    st.markdown("### Requirements")
    st.code("""
    pip install faster-whisper streamlit torch python-dotenv openai librosa pydub numpy streamlit-audiorec
    """)

# Handle file upload chunks - Added to fix the 413 error
def get_file_chunk_size(filesize):
    """Calculate appropriate chunk size based on file size"""
    if filesize < 10 * 1024 * 1024:  # < 10MB
        return 1 * 1024 * 1024  # 1MB chunks
    elif filesize < 50 * 1024 * 1024:  # < 50MB
        return 5 * 1024 * 1024  # 5MB chunks
    else:  # >= 50MB
        return 10 * 1024 * 1024  # 10MB chunks

# Add chunking support to UploadedFile
def add_chunking_to_uploaded_file():
    from streamlit.runtime.uploaded_file_manager import UploadedFile
    
    def chunks(self, chunk_size=None):
        """Iterator to chunk file data"""
        if chunk_size is None:
            chunk_size = get_file_chunk_size(self.size)
            
        self.file.seek(0)
        while True:
            data = self.file.read(chunk_size)
            if not data:
                break
            yield data
        self.file.seek(0)
    
    # Add method to UploadedFile class if it doesn't exist
    if not hasattr(UploadedFile, 'chunks'):
        UploadedFile.chunks = chunks

# Execute before main app runs
add_chunking_to_uploaded_file()  

# Track active tab
def on_tab_change():
    # Determine which tab is active based on widget values
    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
        st.session_state.active_tab = "Upload File"
    elif 'recorded_file_path' in st.session_state and st.session_state.recorded_file_path is not None:
        st.session_state.active_tab = "Record Audio"
    else:
        st.session_state.active_tab = "File Path"

# Initialize session state
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "File Path"

if __name__ == "__main__":
    main()