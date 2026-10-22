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

#include <cstddef>
#include <cstdint>

namespace deluge::dsp::waveguide {

/// String output (internal level ~2^27 at nominal, ceiling 0.75 * 2^29) to osc-buffer level
constexpr int32_t kStringOutputShift = 8;

/// How a voice mixes its unison strings into its oscillator buffer
struct StringMix {
	const q31_t* in;   ///< The strings' output: string k's sample i at in[i * stride + k]
	size_t stride;     ///< 1 for a single string, KarplusString::kGroupSize for a group's interleaved output
	size_t count;      ///< Strings to mix, at most stride
	size_t numSamples; ///< Samples to mix
	int32_t amplitude; ///< Before the first sample; each sample first adds amplitudeIncrement
	int32_t amplitudeIncrement;
	/// Per string, stereo only: pan gains (at most 2^30). Null for mono.
	const int32_t* panLeft;
	const int32_t* panRight;
};

/// Adds the strings into `out` (interleaved left/right when panning), each one clipped to int32 by the saturating
/// shift and summed with add_saturate, so loud strings or many of them clip rather than wrap. On NEON, four samples at
/// a time; bit-identical to mixStringsReference().
void mixStrings(const StringMix& mix, q31_t* out);

/// The scalar mix, one string after another
inline void mixStringsReference(const StringMix& mix, q31_t* out) {
	for (size_t k = 0; k < mix.count; ++k) {
		int32_t amplitude = mix.amplitude;
		for (size_t i = 0; i < mix.numSamples; ++i) {
			amplitude += mix.amplitudeIncrement;
			const int32_t amplified =
			    signed_saturate<32 - kStringOutputShift>(multiply_32x32_rshift32(mix.in[i * mix.stride + k], amplitude))
			    << kStringOutputShift;
			if (mix.panLeft != nullptr) {
				// Pan gains are at most 2^30, so each term is below 2^29 and the << 2 cannot overflow
				out[i << 1] = add_saturate(out[i << 1], multiply_32x32_rshift32(amplified, mix.panLeft[k]) << 2);
				out[(i << 1) + 1] =
				    add_saturate(out[(i << 1) + 1], multiply_32x32_rshift32(amplified, mix.panRight[k]) << 2);
			}
			else {
				out[i] = add_saturate(out[i], amplified);
			}
		}
	}
}

} // namespace deluge::dsp::waveguide
