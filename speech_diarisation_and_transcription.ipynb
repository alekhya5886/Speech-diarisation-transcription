import os
import ffmpeg
import librosa
import noisereduce as nr
import soundfile as sf
import json
import glob
from datetime import timedelta

from huggingface_hub import login
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from transformers import pipeline as hf_pipeline


def setup_huggingface(token: str):
    """Login to HuggingFace."""
    login(token)
    print("✅ HuggingFace token set up.")


def convert_to_wav(input_file: str, output_file: str = "converted_audio.wav"):
    """Convert any audio to mono 16kHz WAV."""
    ffmpeg.input(input_file).output(output_file, ac=1, ar="16000").overwrite_output().run()
    print(f"✅ Audio converted to {output_file}")
    return output_file


def reduce_noise(input_wav: str):
    """Apply noise reduction to WAV file."""
    y, sr = librosa.load(input_wav, sr=16000)
    noise_sample = y[: int(sr * 0.5)]
    reduced_noise = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
    sf.write(input_wav, reduced_noise, sr)
    print("✅ Noise reduction complete.")


def run_diarization(input_wav: str, token: str, rttm_path: str = "diarization.rttm"):
    """Run speaker diarization and save RTTM file."""
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    diarization = pipeline(input_wav)

    with open(rttm_path, "w") as f:
        diarization.write_rttm(f)

    print(f"✅ Diarization complete. RTTM saved as {rttm_path}")
    return diarization


def run_transcription(diarization, input_wav: str, model_size="medium"):
    """Run Whisper transcription per diarized segment."""
    whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start, end = turn.start, turn.end
        if end - start >= 0.5:
            segments.append({"speaker": speaker, "start": start, "end": end})

    results = []
    for i, seg in enumerate(segments):
        start, end, speaker = seg["start"], seg["end"], seg["speaker"]
        segment_path = f"segment_{i}.wav"
        ffmpeg.input(input_wav, ss=start, to=end).output(segment_path, ac=1, ar="16000").overwrite_output().run()

        transcription, _ = whisper_model.transcribe(segment_path, beam_size=5)
        text = " ".join([s.text for s in transcription])

        results.append(
            {"speaker": speaker, "start": float(start), "end": float(end), "transcript": text}
        )

    with open("diarized_transcription.json", "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Transcription complete. Results saved to diarized_transcription.json")
    return results


def summarize_transcription(results, max_chars=3000):
    """Summarize full transcription text using HuggingFace summarizer."""
    summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn")

    full_text = " ".join([seg["transcript"] for seg in results if seg["transcript"]])
    text_for_summary = full_text[:max_chars]

    summary = summarizer(text_for_summary, max_length=200, min_length=50, do_sample=False)[0][
        "summary_text"
    ]

    with open("call_summary.txt", "w") as f:
        f.write(summary)

    print("\n📋 Call Summary:")
    print(summary)
    return summary


def cleanup(temp_files: list):
    """Remove temporary files."""
    for f in temp_files + glob.glob("segment_*.wav"):
        if os.path.exists(f):
            os.remove(f)
    print("✅ Temporary files removed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio Diarization & Transcription Pipeline")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--token", type=str, required=True, help="HuggingFace token")
    parser.add_argument("--model", type=str, default="medium", help="Whisper model size")
    args = parser.parse_args()

    setup_huggingface(args.token)

    wav_file = convert_to_wav(args.audio)
    reduce_noise(wav_file)

    diarization = run_diarization(wav_file, args.token)
    results = run_transcription(diarization, wav_file, model_size=args.model)
    summarize_transcription(results)

    cleanup(["converted_audio.wav", "diarization.rttm", "diarized_transcription.json"])
    print("🎉 All done!")
