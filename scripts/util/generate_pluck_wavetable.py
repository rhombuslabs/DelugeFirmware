#!/usr/bin/env python3
"""Generate PLUCK.WAV, a 256-frame exciter wavetable for plucking the STRING oscillator, from fingertip to hard pick.

Load it on OSC1 (type Wavetable) of a synth whose OSC2 is String, with ENV2 at 0 (attack, decay, sustain, release) so
the exciter is a burst of about one cycle. OSC1's WAVETBL knob then morphs the pluck from a fingertip (0) to a hard
pick (50). Copy the file to the SD card as SAMPLES/_WAVTBL/PhysMod/PLUCK.WAV.

Each frame is one cycle holding a single raised-cosine pulse. A plucked string is released from a corner, and how
sharp that corner is decides the brightness: a soft, wide fingertip rounds it off, a hard, narrow pick doesn't. So the
pulse narrows across the table, from 30% of the cycle down to 1%:
  - Every pulse has the same area, so the fundamental and low harmonics stay level across the table: the knob changes
    brightness, not loudness.
  - Narrowing the pulse moves its roll-off up: the fingertip end falls away from the 3rd or 4th harmonic, the pick
    end is flat to about the 50th, like a noise burst but without its randomness or low thump.
  - The width changes exponentially, so the knob sweeps brightness evenly.
Pluck position is not part of the table: that is the STRING oscillator's PW knob.

The file carries a Serum-style "clm " chunk ("<!>2048 ..."), which is how the Deluge recognises a WAV as a wavetable
(see AudioFile::loadFile in storage/audio/audio_file.cpp). For mallet-like shapes see generate_mallet_wavetable.py.

Usage: python generate_pluck_wavetable.py [output.wav] [--spectrum]
"""

import math
import struct
import sys

SAMPLE_RATE = 44100
FRAME_SIZE = 2048
NUM_FRAMES = 256
WIDEST = 0.30  # Fingertip pulse width, as a fraction of the cycle
NARROWEST = 0.01  # Hard pick
PEAK = 0.95  # The narrowest pulse's peak; the others are scaled to its area


def pulse_width(frame_idx: int) -> float:
    """Pulse width (fraction of the cycle) of a frame, exponential from WIDEST to NARROWEST."""
    position = frame_idx / (NUM_FRAMES - 1)
    return WIDEST * (NARROWEST / WIDEST) ** position


def generate_frame(frame_idx: int) -> list[float]:
    width = pulse_width(frame_idx)
    # Equal area: a raised-cosine pulse's area is peak * width / 2, matched to the narrowest pulse's
    peak = PEAK * NARROWEST / width
    start = 0.5 - width / 2  # Centred in the cycle
    frame = [0.0] * FRAME_SIZE
    for i in range(FRAME_SIZE):
        t = (i / FRAME_SIZE - start) / width
        if 0.0 <= t <= 1.0:
            frame[i] = peak * 0.5 * (1.0 - math.cos(2.0 * math.pi * t))
    # Remove the DC, which would otherwise reach the string as a low thump
    mean = sum(frame) / FRAME_SIZE
    return [v - mean for v in frame]


def harmonic_levels(frame: list[float], harmonics: list[int]) -> list[float]:
    """Magnitudes of the given harmonics of one cycle."""
    levels = []
    for h in harmonics:
        re = sum(
            v * math.cos(2 * math.pi * h * i / FRAME_SIZE) for i, v in enumerate(frame)
        )
        im = sum(
            v * math.sin(2 * math.pi * h * i / FRAME_SIZE) for i, v in enumerate(frame)
        )
        levels.append(math.hypot(re, im))
    return levels


def chunk(name: bytes, data: bytes) -> bytes:
    padding = b"\x00" if len(data) % 2 else b""
    return name + struct.pack("<I", len(data)) + data + padding


def build_wav() -> bytes:
    samples = bytearray()
    for f in range(NUM_FRAMES):
        for sample in generate_frame(f):
            samples += struct.pack("<h", round(max(-1.0, min(1.0, sample)) * 32767.0))
    fmt = struct.pack(
        "<HHIIHH", 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16
    )  # PCM, mono, 16-bit
    clm = f"<!>{FRAME_SIZE} 00000000 wavetable (Deluge string pluck: fingertip to pick)".encode(
        "ascii"
    )
    body = (
        b"WAVE"
        + chunk(b"fmt ", fmt)
        + chunk(b"clm ", clm)
        + chunk(b"data", bytes(samples))
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def print_spectrum() -> None:
    harmonics = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
    print(
        "frame  width  "
        + " ".join(f"h{h:<4}" for h in harmonics)
        + "  (dB re the narrowest frame's h1)"
    )
    reference = harmonic_levels(generate_frame(NUM_FRAMES - 1), [1])[0]
    for f in (0, 64, 128, 192, 255):
        levels = harmonic_levels(generate_frame(f), harmonics)
        row = " ".join(f"{20 * math.log10(v / reference + 1e-9):5.0f}" for v in levels)
        print(f"{f:5}  {pulse_width(f) * 100:4.1f}%  {row}")


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--spectrum"]
    if "--spectrum" in sys.argv:
        print_spectrum()
    path = args[0] if args else "PLUCK.WAV"
    data = build_wav()
    with open(path, "wb") as f:
        f.write(data)
    print(
        f"Wrote {path}: {NUM_FRAMES} frames x {FRAME_SIZE} samples, {len(data)} bytes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
