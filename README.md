# Project Summary
The Korean Speech-to-Text (STT) application is a tool designed to facilitate the transcription of Korean audio recordings into text format. 

This application is aimed at improving productivity in various business settings, such as meetings, interviews, and lectures. By leveraging advanced speech recognition technology powered by the Faster Whisper model, the application allows users to upload audio files or directly record meetings. In addition to transcription, the application integrates with Azure OpenAI to generate meeting notes, making it a comprehensive solution for teams to document discussions and actions efficiently.

# Project Module Description
## Key Functional Modules
1. **Audio Transcription**: 
   - Upload audio files (mp3, wav, m4a, etc.) or record directly from the browser.
   - Supports large audio files (up to 2 hours) with file chunking for efficient processing.
   - Option to include or exclude timestamps in the transcription.

2. **Meeting Notes Generation**: 
   - After transcription, users have the option to generate meeting notes through integration with Azure OpenAI.
   - Meeting notes are saved as markdown files and can be downloaded.

3. **User Interface**:
   - Web-based interface built with Streamlit for easy interaction.
   - Tabs for different input methods: file path input, file uploads, and audio recording.
   - Clear instructions for users on how to utilize built-in recording software outside the application if needed.

# Directory Tree
```plaintext
/data/chats/067qsd/workspace
+-- streamlit_template
|   +-- app.py
|   +-- requirements.txt

+-- uploads
    +-- output_text.txt
    +-- recorded_audio.mp3
```

# File Description Inventory
- **app.py**: Main application file that handles user input, audio processing, transcription, and note generation.
- **requirements.txt**: Contains a list of Python package dependencies required to run the application.
- **template_config.json**: Holds configuration settings for the Streamlit application (not detailed in this context).
- **uploads/**: Directory containing uploaded recordings and generated outputs (text files).

# Technology Stack
- **Backend**: Python with Faster Whisper for speech recognition.
- **Frontend**: Streamlit for creating the user interface.
- **Audio Processing**: Pydub for handling audio files.
- **Cloud Processing**: Azure OpenAI for generating meeting notes.
- **Environment Management**: dotenv for managing environment variables.

# Usage
## Installation
To set up the project and install the required dependencies, execute the following commands in your terminal:
```bash
pip install -r requirements.txt
```

## Building and Running
1. **Navigate** to the project directory:
   ```bash
   cd /data/chats/067qsd/workspace/streamlit_template
   ```

2. **Run** the application:
   ```bash
   streamlit run app.py
   ```

The application will start and be accessible in your web browser.

Ensure that you have properly configured your environment variables for Azure OpenAI before running the application to enable meeting notes generation.


# INSTRUCTION
- Project Path:`/data/chats/067qsd/workspace/streamlit_template`
- You can search for the file path in the 'Directory Tree';
- After modifying the project files, if this project can be previewed, then you need to reinstall dependencies, restart service and preview;
