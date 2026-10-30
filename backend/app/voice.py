"""Browser audio -> the WAV Muse Voice Transcribe accepts (RIFF/WAVE, mono, 16-bit PCM at 16 or 24 kHz; at most 10 min and 32 MB).

WAV input is converted in memory with the standard library and scipy (any PCM sample width, channels mixed down, resampled to 24 kHz).
Other containers (webm/ogg/mp4 from MediaRecorder) are converted with ffmpeg only if it is installed; otherwise they are refused with a
plain message. Audio is never written to job storage; an ffmpeg temp file is deleted at once."""
import io
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

MAX_S = 600
KEYWORDS = ["slot", "slots", "bump", "ridge", "lock ridge", "plate", "depth", "height", "width", "flat", "remove", "rounded", "sharp",
            "underside", "opposite side", "millimetre", "millimetres", "mm"]


class AudioError(ValueError):
    pass


def to_wav(data: bytes, max_bytes: int) -> tuple[bytes, float]:
    """(WAV bytes ready for transcription, duration s)."""
    if len(data) > max_bytes:
        raise AudioError(f"The recording is larger than {max_bytes // 2**20} MB.")
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        pcm, rate = _read_wav(data)
    elif shutil.which("ffmpeg"):
        pcm, rate = _read_wav(_ffmpeg(data))
    else:
        raise AudioError("Please send the recording as WAV (this server cannot convert other audio formats).")
    dur = len(pcm) / rate
    if dur > MAX_S:
        raise AudioError("The recording is longer than 10 minutes.")
    if dur < 0.2:
        raise AudioError("The recording is too short.")
    if rate not in (16000, 24000):
        from scipy.signal import resample_poly
        g = np.gcd(rate, 24000)
        pcm = resample_poly(pcm, 24000 // g, rate // g)
        rate = 24000
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.clip(np.round(pcm * 32767), -32768, 32767).astype("<i2").tobytes())
    return out.getvalue(), dur


def _read_wav(data: bytes) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            ch, sw, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            if n / max(rate, 1) > MAX_S + 1:
                raise AudioError("The recording is longer than 10 minutes.")
            raw = w.readframes(n)
    except wave.Error:
        raise AudioError("This is not a readable PCM WAV file.") from None
    if sw == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif sw == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        x = ((b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)) << 8 >> 8).astype(np.float32) / 2 ** 23
    elif sw == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2 ** 31
    else:
        raise AudioError("Unsupported WAV sample size.")
    if not 8000 <= rate <= 96000 or ch < 1:
        raise AudioError("Unsupported WAV sample rate.")
    return x.reshape(-1, ch).mean(1), rate


def _ffmpeg(data: bytes) -> bytes:
    fd, src = tempfile.mkstemp(prefix="snappart-voice-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", src, "-t", str(MAX_S + 1), "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
                            "-map_metadata", "-1", "-f", "wav", "pipe:1"], capture_output=True, timeout=60)
        if r.returncode != 0 or not r.stdout:
            raise AudioError("This recording could not be read.")
        return r.stdout
    finally:
        os.unlink(src)                                                # the audio never stays on disk
