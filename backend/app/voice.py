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
KEYWORDS = ["slot", "slots", "depth", "deep", "ridge", "plate", "millimetres", "mm", "point", "bump", "lock ridge", "height", "width",
            "flat", "remove", "rounded", "sharp", "underside", "opposite side", "millimetre"]   # hints for the speech recogniser


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
    """Mono float samples and the rate, from PCM (8, 16, 24 or 32 bit) or float WAV, plain or WAVE_FORMAT_EXTENSIBLE (common from phone and
    desktop recorders)."""
    import warnings
    from scipy.io import wavfile
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", wavfile.WavFileWarning)  # unknown extra chunks (LIST, fact) are skipped
            rate, x = wavfile.read(io.BytesIO(data))
    except Exception:                                                   # scipy raises several kinds on broken files
        raise AudioError("This is not a readable PCM WAV file.") from None
    if not 8000 <= rate <= 96000:
        raise AudioError("Unsupported WAV sample rate.")
    if len(x) / rate > MAX_S + 1:
        raise AudioError("The recording is longer than 10 minutes.")
    if x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128) / 128
    elif x.dtype == np.int16:
        x = x.astype(np.float32) / 32768
    elif x.dtype == np.int32:                                          # 24-bit arrives left-justified in int32
        x = x.astype(np.float32) / 2 ** 31
    elif x.dtype.kind == "f":
        x = x.astype(np.float32)
    else:
        raise AudioError("Unsupported WAV sample size.")
    return x.reshape(len(x), -1).mean(1), rate


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
