# 🎙️ Audio Diarization, Transcription & Summarization Pipeline

This project provides a complete pipeline for processing audio recordings from calls, meetings, interviews, and more. It handles noise reduction, speaker diarization, transcription, and AI-powered summarization.

-----

## ⚙️ Requirements

  * **Python 3.9+**
  * **FFmpeg:** Must be installed and accessible in your system's PATH. You can download it from the official [FFmpeg website](https://ffmpeg.org/download.html).
  * **Hugging Face Access Token:** Required for accessing the models. You can get one from your [Hugging Face settings](https://huggingface.co/settings/tokens).

To install the necessary Python libraries, run:

```bash
pip install -r requirements.txt
```

-----

## 🚀 Quick Start

Run the entire pipeline with a single command:

```bash
python main.py --audio path/to/input_file.mp3 --token YOUR_HF_TOKEN
```

### Options

  * `--audio`: Path to your input audio file (supports `.mp3`, `.wav`, `.m4a`, etc.).
  * `--token`: Your Hugging Face access token.
  * `--model`: The size of the Whisper model to use (`small`, `medium` [default], `large-v2`).

-----

## 📝 Step-by-Step Usage (Inside Python)

If you prefer to integrate each step into your own Python script, you can import and run the functions manually:

```python
from main import (
    setup_huggingface,
    convert_to_wav,
    reduce_noise,
    run_diarization,
    run_transcription,
    summarize_transcription,
    cleanup,
)

# 1. Set up Hugging Face
setup_huggingface("YOUR_HF_TOKEN")

# 2. Convert input audio to WAV (mono, 16kHz)
wav_file = convert_to_wav("input.mp3")

# 3. Apply noise reduction
reduce_noise(wav_file)

# 4. Perform speaker diarization (who spoke when)
diarization = run_diarization(wav_file, "YOUR_HF_TOKEN")

# 5. Transcribe diarized segments using Whisper
results = run_transcription(diarization, wav_file, model_size="medium")

# 6. Summarize the conversation using an AI model
summary = summarize_transcription(results)

# 7. Clean up temporary files
cleanup(["converted_audio.wav", "diarization.rttm", "diarized_transcription.json"])
```

-----

## 📂 Outputs

The pipeline generates the following files in the project directory:

  * `diarized_transcription.json`: A JSON file containing the full transcript with speaker labels and timestamps.
  * `diarization.rttm`: A standard RTTM file detailing speaker segments.
  * `call_summary.txt`: An AI-generated summary of the entire conversation.

-----

## 🔑 Getting a Hugging Face Token

1.  Navigate to [Hugging Face Tokens](https://huggingface.co/settings/tokens).
2.  Click the **"New Token"** button.
3.  Give it a name and select **"Read"** permissions.
4.  Copy the generated token and use it with the `--token` flag.

-----

## ☁️ Running on Google Colab (Optional)

If you prefer to run this without a local setup, Google Colab is a great option.

1.  Open a new Colab notebook.

2.  Install dependencies:

    ```python
    !pip install -r requirements.txt
    !apt-get install -qq ffmpeg
    ```

3.  Upload your audio file:

    ```python
    from google.colab import files
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    ```

4.  Run the pipeline:

    ```python
    !python main.py --audio $filename --token YOUR_HF_TOKEN
    ```

5.  Download your results:

    ```python
    from google.colab import files
    files.download("diarized_transcription.json")
    files.download("call_summary.txt")
    ```
