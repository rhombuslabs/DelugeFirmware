#!/usr/bin/env python3
"""Generate MALLET.WAV, a 256-frame exciter wavetable of soft, mallet- and hammer-like shapes.

These shapes are mostly fundamental (by the 8th harmonic they are 27-40 dB down), so they strike rather than pick:
they suit a future mallet model better than the plucked string, which uses generate_pluck_wavetable.py. On the
STRING oscillator they make a dull, struck tone: load the file on OSC1 (type Wavetable) with OSC2 on String, and
OSC1's WAVETBL knob picks the shape. Copy it to the SD card as SAMPLES/_WAVTBL/PhysMod/MALLET.WAV.

The file carries a Serum-style "clm " chunk ("<!>2048 ..."), which is how the Deluge recognises a WAV as
a wavetable and learns its 2048-sample cycle size (see AudioFile::loadFile in storage/audio/audio_file.cpp).

Frame zones (64 frames each; frames within a zone are identical):
  0-63    soft felt / nylon pluck with a sigmoid touch
  64-127  piano-hammer-like, asymmetric
  128-191 plectrum snap on a cubic (x - x^3) shape
  192-255 harpsichord quill / sitar-jawari-like clipping with high-harmonic sizzle

Usage: python generate_mallet_wavetable.py [output.wav]   (default: MALLET.WAV)
"""

import math
import struct
import sys

SAMPLE_RATE = 44100
FRAME_SIZE = 2048
NUM_FRAMES = 256


def generate_frame(frame_idx: int) -> list[float]:
    frame = [0.0] * FRAME_SIZE
    for i in range(FRAME_SIZE):
        t = i / float(FRAME_SIZE)  # phase 0..1
        angle = t * 2.0 * math.pi

        if frame_idx < 64:
            # Soft felt hammer / nylon pluck: decaying sine plus a little sigmoid shaping
            val = math.sin(angle) * math.exp(-t * 3.0)
            val += 0.15 * math.tanh(2.0 * math.sin(angle))
        elif frame_idx < 128:
            # Piano hammer: sharper, asymmetrically clipped, with a decaying 2nd harmonic
            val = math.sin(angle) * (1.0 - t * 0.5)
            if val > 0.3:
                val = 0.3 + (val - 0.3) * 0.5
            val += 0.2 * math.sin(2.0 * angle) * math.exp(-t * 4.0)
        elif frame_idx < 192:
            # Plectrum snap on a cubic air-jet-like curve
            x = math.sin(angle)
            val = x - (x * x * x) * 0.4
            if t < 0.15:
                val += (0.5 - t * 3.3) * math.cos(t * 20.0 * math.pi)
        else:
            # Quill / jawari: flattened negative half-cycle plus decaying 6th-harmonic sizzle
            x = math.sin(angle)
            val = -0.1 - ((-x - 0.1) * 0.25) if x < -0.1 else x
            val += 0.25 * math.sin(6.0 * angle) * math.exp(-t * 2.0)

        frame[i] = max(-1.0, min(1.0, val))
    return frame


def chunk(name: bytes, data: bytes) -> bytes:
    padding = b"\x00" if len(data) % 2 else b""
    return name + struct.pack("<I", len(data)) + data + padding


def build_wav() -> bytes:
    samples = bytearray()
    for f in range(NUM_FRAMES):
        for sample in generate_frame(f):
            samples += struct.pack("<h", int(sample * 32767.0))

    fmt = struct.pack(
        "<HHIIHH", 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16
    )  # PCM, mono, 16-bit
    clm = f"<!>{FRAME_SIZE} 00000000 wavetable (Deluge mallet exciter)".encode("ascii")
    body = (
        b"WAVE"
        + chunk(b"fmt ", fmt)
        + chunk(b"clm ", clm)
        + chunk(b"data", bytes(samples))
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "MALLET.WAV"
    data = build_wav()
    with open(path, "wb") as f:
        f.write(data)
    print(
        f"Wrote {path}: {NUM_FRAMES} frames x {FRAME_SIZE} samples, {len(data)} bytes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
