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
#include "dsp/waveguide/string_mix.hpp"
#include "dsp/waveguide/karplus_string.hpp"

#include <arm_simd.hpp> // <arm_neon.h> on the Deluge, SIMDe's NEON on a host
#include <utility>

// Each step is the exact NEON twin of mixStringsReference's: smmul(a, b) is vqdmulh(a, b) >> 1 (vqdmulh saturates only
// for INT32_MIN * INT32_MIN; the amplitude and pan gains are never INT32_MIN), and ssat to 24 bits is a clamp. Four
// samples go at once, adding the strings in the same order, so every output element sees the same sequence of
// saturating adds.

namespace deluge::dsp::waveguide {

namespace {
/// signed_saturate<32 - kStringOutputShift>(smmul(x, amplitude)) << kStringOutputShift
inline int32x4_t amplify(int32x4_t x, int32x4_t amplitude) {
	constexpr int32_t kTop = (1 << (31 - kStringOutputShift)) - 1;
	const int32x4_t product = vshrq_n_s32(vqdmulhq_s32(x, amplitude), 1);
	return vshlq_n_s32(vmaxq_s32(vminq_s32(product, vdupq_n_s32(kTop)), vdupq_n_s32(-kTop - 1)), kStringOutputShift);
}

/// smmul(amplified, pan) << 2. A widening multiply rather than vqdmulh: a clipped string is amplified to exactly
/// INT32_MIN, which SIMDe's 32-bit vqdmulh mishandles (the hardware's is right), so the host specs couldn't check it.
inline int32x4_t pan(int32x4_t amplified, int32x2_t gain) {
	const int32x2_t low = vshrn_n_s64(vmull_s32(vget_low_s32(amplified), gain), 32);
	const int32x2_t high = vshrn_n_s64(vmull_s32(vget_high_s32(amplified), gain), 32);
	return vshlq_n_s32(vcombine_s32(low, high), 2);
}

/// kCount is a template parameter, and the strings are unrolled with an index sequence, so each string's vector and
/// pan gains stay in fixed registers (a plain loop over k was left rolled, with the vectors bounced through the stack)
template <size_t kStride, bool kStereo, size_t kCount>
void mixBlocks(const StringMix& mix, q31_t* out, size_t numBlocks) {
	const int32_t increment = mix.amplitudeIncrement;
	// Amplitudes for samples 0..3, as the reference's running sum (with its wrap-around)
	static constexpr uint32_t kSteps[4] = {1, 2, 3, 4};
	const int32x4_t startAmplitude = vreinterpretq_s32_u32(vmlaq_n_u32(
	    vdupq_n_u32(static_cast<uint32_t>(mix.amplitude)), vld1q_u32(kSteps), static_cast<uint32_t>(increment)));
	const int32x4_t blockIncrement = vdupq_n_s32(static_cast<int32_t>(static_cast<uint32_t>(increment) * 4));
	// Each gain in both lanes of a NEON register, once, rather than moved over from the core every block
	int32x2_t panLeft[kCount];
	int32x2_t panRight[kCount];
	if constexpr (kStereo) {
		for (size_t k = 0; k < kCount; ++k) {
			panLeft[k] = vdup_n_s32(mix.panLeft[k]);
			panRight[k] = vdup_n_s32(mix.panRight[k]);
		}
	}
	int32x4_t amplitude = startAmplitude;
	for (size_t block = 0; block < numBlocks; ++block) {
		const size_t i = block * 4;
		int32x4x4_t strings;
		if constexpr (kStride == 1) {
			strings.val[0] = vld1q_s32(mix.in + i);
		}
		else {
			strings = vld4q_s32(mix.in + i * kStride);
		}
		if constexpr (kStereo) {
			const int32x4x2_t before = vld2q_s32(out + 2 * i);
			int32x4_t left = before.val[0];
			int32x4_t right = before.val[1];
			[&]<size_t... k>(std::index_sequence<k...>) {
				((left = vqaddq_s32(left, pan(amplify(strings.val[k], amplitude), panLeft[k])),
				  right = vqaddq_s32(right, pan(amplify(strings.val[k], amplitude), panRight[k]))),
				 ...);
			}(std::make_index_sequence<kCount>{});
			// Named first: SIMDe's vst2q_s32 is a macro, which would split a braced argument at its comma
			const int32x4x2_t after{{left, right}};
			vst2q_s32(out + 2 * i, after);
		}
		else {
			int32x4_t sum = vld1q_s32(out + i);
			[&]<size_t... k>(std::index_sequence<k...>) {
				((sum = vqaddq_s32(sum, amplify(strings.val[k], amplitude))), ...);
			}(std::make_index_sequence<kCount>{});
			vst1q_s32(out + i, sum);
		}
		amplitude = vaddq_s32(amplitude, blockIncrement);
	}
}
} // namespace

void mixStrings(const StringMix& mix, q31_t* out) {
	const size_t numBlocks = mix.numSamples / 4;
	const bool stereo = mix.panLeft != nullptr;
	if (mix.stride == 1) {
		stereo ? mixBlocks<1, true, 1>(mix, out, numBlocks) : mixBlocks<1, false, 1>(mix, out, numBlocks);
	}
	else {
		static_assert(KarplusString::kGroupSize == 4);
		switch (mix.count) {
		case 1:
			stereo ? mixBlocks<4, true, 1>(mix, out, numBlocks) : mixBlocks<4, false, 1>(mix, out, numBlocks);
			break;
		case 2:
			stereo ? mixBlocks<4, true, 2>(mix, out, numBlocks) : mixBlocks<4, false, 2>(mix, out, numBlocks);
			break;
		case 3:
			stereo ? mixBlocks<4, true, 3>(mix, out, numBlocks) : mixBlocks<4, false, 3>(mix, out, numBlocks);
			break;
		default:
			stereo ? mixBlocks<4, true, 4>(mix, out, numBlocks) : mixBlocks<4, false, 4>(mix, out, numBlocks);
			break;
		}
	}
	// The last few samples, one at a time
	const size_t done = numBlocks * 4;
	if (done < mix.numSamples) {
		StringMix tail = mix;
		tail.in = mix.in + done * mix.stride;
		tail.numSamples = mix.numSamples - done;
		tail.amplitude = static_cast<int32_t>(static_cast<uint32_t>(mix.amplitude)
		                                      + static_cast<uint32_t>(mix.amplitudeIncrement) * done);
		mixStringsReference(tail, out + (stereo ? 2 * done : done));
	}
}

} // namespace deluge::dsp::waveguide
