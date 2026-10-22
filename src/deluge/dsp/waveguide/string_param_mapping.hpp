/*
 * Copyright © 2026 Synthstrom Audible Limited
 *
 * This file is part of The Synthstrom Audible Deluge Firmware.
 *
 * The Synthstrom Audible Deluge Firmware is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program.
 * If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once

#include "util/fixedpoint.h"

#include <algorithm>
#include <cstdint>
#include <span>

namespace deluge::dsp::waveguide {

/// Final value of LOCAL_CARRIER_x_FEEDBACK at knob 50 with no cables:
/// getFinalParameterValueLinear(5931642, 536870911) = lshiftAndSaturate<3>((1073741823 * 5931642) >> 32).
constexpr int32_t kFeedbackFinalMax = 11863280;

/// Hybrid params (PW, wave index) have final values in [-2^30, 2^30].
constexpr float kHybridFinalMax = 1073741824.f;

/// OSC PW: knob 0..50 gives final values 0..2^30.
inline float pulseWidthToUnit(int32_t finalValue) {
	return std::clamp(static_cast<float>(finalValue) / kHybridFinalMax, 0.f, 1.f);
}

/// OSC wave index: knob 0..50 gives final values -2^30..2^30.
inline float waveIndexToUnit(int32_t finalValue) {
	return std::clamp((static_cast<float>(finalValue) / kHybridFinalMax + 1.f) * 0.5f, 0.f, 1.f);
}

/// Carrier feedback: knob 0..50 gives final values 0..kFeedbackFinalMax; cables can overshoot, so clamp.
inline float feedbackToUnit(int32_t finalValue) {
	return std::clamp(static_cast<float>(finalValue) / static_cast<float>(kFeedbackFinalMax), 0.f, 1.f);
}

inline float velocityToUnit(int32_t velocity) {
	return static_cast<float>(std::clamp<int32_t>(velocity, 0, 127)) / 127.f;
}

/// Envelope output (INT32_MIN = silent, INT32_MAX = full) to a q31 gain in [0, ONE_Q31].
constexpr q31_t envelopeToGain(int32_t envelopeValue) {
	return (envelopeValue >> 1) + (1 << 30);
}

/// Multiplies `buffer` by a gain ramping linearly from `from` to `to` (both in [0, ONE_Q31]).
inline void applyGainRamp(std::span<q31_t> buffer, q31_t from, q31_t to) {
	if (buffer.empty()) {
		return;
	}
	const int32_t step = (to - from) / static_cast<int32_t>(buffer.size());
	q31_t gain = from;
	for (q31_t& sample : buffer) {
		gain += step;
		sample = q31_mult(sample, gain);
	}
}

} // namespace deluge::dsp::waveguide
