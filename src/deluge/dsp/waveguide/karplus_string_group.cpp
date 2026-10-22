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
#include "dsp/waveguide/karplus_string.hpp"

#include <algorithm>
#include <arm_simd.hpp> // <arm_neon.h> on the Deluge, SIMDe's NEON on a host

// KarplusString::renderPlucked() for a voice's unison strings, one per NEON lane. Every step is the exact NEON twin of
// the scalar code's, so the results are bit-identical (karplus_string_spec checks this):
// - q31_mult(a, b) is smmul(a, b) * 2, i.e. floor(a b / 2^32) * 2: vqdmulh gives floor(a b / 2^31), and clearing its
//   lowest bit gives the same. (It saturates only for INT32_MIN * INT32_MIN, which can't happen here: one operand is
//   always a positive constant or fraction.)
// - multiply32x32(a, b) << 1 is vqdmull(a, b), which saturates under the same condition only.
// - smmul(t, t) is vqdmulh(t, t) >> 1.
// - (y + 2^31) >> 32 is vrshrn(y, 32), and y >> 32 is vshrn(y, 32).

namespace deluge::dsp::waveguide {

namespace {
using LoopLine = FractionalDelay<KarplusString::kLoopSize>;

/// A Lagrange read at delay d reaches back to the sample written d - 1 samples ago, so with d >= 3 the next sample's
/// read never needs the current sample's write
constexpr int32_t kMinPairedDelay = 3;

/// q31_mult on each lane
inline int32x4_t q31Mult(int32x4_t a, int32x4_t b) {
	return vandq_s32(vqdmulhq_s32(a, b), vdupq_n_s32(~1));
}
inline int32x4_t q31Mult(int32x4_t a, int32_t b) {
	return vandq_s32(vqdmulhq_n_s32(a, b), vdupq_n_s32(~1));
}

/// Two int64x2 halves to one int32x4, each (y + 2^31) >> 32
inline int32x4_t roundHigh(int64x2_t low, int64x2_t high) {
	return vcombine_s32(vrshrn_n_s64(low, 32), vrshrn_n_s64(high, 32));
}

/// Per-lane state, gathered from the strings before a block and scattered back after it
struct alignas(16) Lanes {
	int64_t lowpassY[KarplusString::kGroupSize];
	int64_t highpassY[KarplusString::kGroupSize];
	int32_t lowpassA[KarplusString::kGroupSize];
	int32_t highpassH[KarplusString::kGroupSize];
	int32_t highpassX[KarplusString::kGroupSize];
	int32_t gain[KarplusString::kGroupSize];
	int32_t delay[KarplusString::kGroupSize];
	int32_t step[KarplusString::kGroupSize];
};

/// Where a Lagrange read at `delay` finds its oldest tap, as in FractionalDelay::readLagrange
inline const q31_t* oldestTap(const q31_t* buffer, uint32_t position, int32_t delay) {
	int32_t index = static_cast<int32_t>(position) - static_cast<int32_t>((static_cast<uint32_t>(delay) >> 16) + 2);
	if (index < 0) {
		index += KarplusString::kLoopSize;
	}
	return buffer + index;
}

/// Each lane's four taps, oldest first: taps.val[j] lane k is tap j of lane k. Every lane is loaded, so the vectors
/// start undefined rather than carrying the last sample's taps, which would hold four NEON registers across the loop.
inline int32x4x4_t loadTaps(const q31_t* lane0, const q31_t* lane1, const q31_t* lane2, const q31_t* lane3) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
	int32x4x4_t taps;
	taps = vld4q_lane_s32(lane0, taps, 0);
#pragma GCC diagnostic pop
	taps = vld4q_lane_s32(lane1, taps, 1);
	taps = vld4q_lane_s32(lane2, taps, 2);
	return vld4q_lane_s32(lane3, taps, 3);
}

/// The fraction of a Q16.16 delay as a Q31 interpolation weight, as in FractionalDelay::readLagrange
inline uint32_t fraction(int32_t delay) {
	return (static_cast<uint32_t>(delay) & 0xFFFF) << 15;
}

/// Renders `n` samples of up to four strings, one per lane. With `paired`, two samples are interleaved: the second's
/// taps are read before the first's result is written, which is only right when every lane's delay stays at least
/// kMinPairedDelay samples, so neither sample reads the other's write. The A9's NEON unit issues in order, and each
/// sample is one long chain of dependent operations, so two independent chains fill each other's latency.
template <bool kExcited>
void renderLanes(q31_t* const* buffers, uint32_t& position, Lanes& lanes, const q31_t* in, q31_t* out, size_t n,
                 bool paired) {
	int64x2_t lowpassLow = vld1q_s64(&lanes.lowpassY[0]);
	int64x2_t lowpassHigh = vld1q_s64(&lanes.lowpassY[2]);
	int64x2_t highpassLow = vld1q_s64(&lanes.highpassY[0]);
	int64x2_t highpassHigh = vld1q_s64(&lanes.highpassY[2]);
	int32x4_t highpassX = vld1q_s32(lanes.highpassX);
	// The lowpass output, kept from one sample to the next rather than recomputed from the state
	int32x4_t lowpassed = roundHigh(lowpassLow, lowpassHigh);
	const int32x4_t a = vld1q_s32(lanes.lowpassA);
	const int32x4_t h = vld1q_s32(lanes.highpassH);
	const int32x4_t gain = vld1q_s32(lanes.gain);
	// The delays, tap addresses and fractions are worked out on the ARM core, alongside the NEON work: moving values
	// from NEON to the core stalls the Cortex-A9 (the other way is cheap)
	int32_t delay0 = lanes.delay[0];
	int32_t delay1 = lanes.delay[1];
	int32_t delay2 = lanes.delay[2];
	int32_t delay3 = lanes.delay[3];

	const int64x2_t roundTowardZero = vdupq_n_s64((int64_t{1} << 30) - 1);
	const int32x4_t knee = vdupq_n_s32(SoftSaturator::kKnee);
	const int32x4_t limit = vdupq_n_s32(SoftSaturator::kLimit);
	const int32x4_t maxLevel = vdupq_n_s32(SoftSaturator::kMax);
	const int32x4_t minLevel = vdupq_n_s32(-SoftSaturator::kMax);
	const int32x4_t zero = vdupq_n_s32(0);

	// The next sample's FractionalDelay::readLagrange on every lane, for the sample to be written at `pos`
	const auto read = [&](uint32_t pos) __attribute__((always_inline)) {
		delay0 += lanes.step[0];
		delay1 += lanes.step[1];
		delay2 += lanes.step[2];
		delay3 += lanes.step[3];
		// Keeps GCC from vectorising this into NEON registers, which would need NEON-to-core moves for the addresses
		asm("" : "+r"(delay0), "+r"(delay1), "+r"(delay2), "+r"(delay3));
		const int32x4x4_t taps = loadTaps(oldestTap(buffers[0], pos, delay0), oldestTap(buffers[1], pos, delay1),
		                                  oldestTap(buffers[2], pos, delay2), oldestTap(buffers[3], pos, delay3));
		const int32x4_t f = vreinterpretq_s32_u32(
		    vcombine_u32(vcreate_u32(fraction(delay0) | (static_cast<uint64_t>(fraction(delay1)) << 32)),
		                 vcreate_u32(fraction(delay2) | (static_cast<uint64_t>(fraction(delay3)) << 32))));
		const int32x4_t x2 = taps.val[0];
		const int32x4_t x1 = taps.val[1];
		const int32x4_t x0 = taps.val[2];
		const int32x4_t xm1 = taps.val[3];
		const int32x4_t halfX0 = vshrq_n_s32(x0, 1);
		const int32x4_t halfX1 = vshrq_n_s32(x1, 1);
		const int32x4_t c1 =
		    vsubq_s32(vsubq_s32(vsubq_s32(x1, q31Mult(xm1, LoopLine::kThird)), halfX0), q31Mult(x2, LoopLine::kSixth));
		const int32x4_t c2 = vsubq_s32(vaddq_s32(vshrq_n_s32(xm1, 1), halfX1), x0);
		const int32x4_t c3 = vaddq_s32(q31Mult(vsubq_s32(x2, xm1), LoopLine::kSixth), vsubq_s32(halfX0, halfX1));
		return vaddq_s32(q31Mult(vaddq_s32(q31Mult(vaddq_s32(q31Mult(c3, f), c2), f), c1), f), x0);
	};

	// The rest of the loop for sample i, from its delay-line read; also writes it to `out`
	const auto process = [&](int32x4_t delayed, size_t i) __attribute__((always_inline)) {
		// LoopLowpass::process
		const int32x4_t lowpassDiff = vsubq_s32(delayed, lowpassed);
		lowpassLow = vaddq_s64(lowpassLow, vqdmull_s32(vget_low_s32(lowpassDiff), vget_low_s32(a)));
		lowpassHigh = vaddq_s64(lowpassHigh, vqdmull_s32(vget_high_s32(lowpassDiff), vget_high_s32(a)));
		lowpassed = roundHigh(lowpassLow, lowpassHigh);

		// LoopHighpass::process
		const int32x4_t d = vsubq_s32(lowpassed, highpassX);
		const int32x4_t y = vcombine_s32(vshrn_n_s64(highpassLow, 32), vshrn_n_s64(highpassHigh, 32));
		highpassLow = vsubq_s64(vsubq_s64(vaddq_s64(highpassLow, vshll_n_s32(vget_low_s32(d), 32)),
		                                  vshrq_n_s64(vmull_s32(vget_low_s32(d), vget_low_s32(h)), 1)),
		                        vmull_s32(vget_low_s32(y), vget_low_s32(h)));
		highpassHigh = vsubq_s64(vsubq_s64(vaddq_s64(highpassHigh, vshll_n_s32(vget_high_s32(d), 32)),
		                                   vshrq_n_s64(vmull_s32(vget_high_s32(d), vget_high_s32(h)), 1)),
		                         vmull_s32(vget_high_s32(y), vget_high_s32(h)));
		highpassX = lowpassed;
		const int32x4_t damped = roundHigh(highpassLow, highpassHigh);

		// Loop gain, rounded toward zero
		int64x2_t productLow = vmull_s32(vget_low_s32(damped), vget_low_s32(gain));
		int64x2_t productHigh = vmull_s32(vget_high_s32(damped), vget_high_s32(gain));
		productLow = vaddq_s64(productLow, vandq_s64(vshrq_n_s64(productLow, 63), roundTowardZero));
		productHigh = vaddq_s64(productHigh, vandq_s64(vshrq_n_s64(productHigh, 63), roundTowardZero));
		int32x4_t sample = vcombine_s32(vshrn_n_s64(productLow, 30), vshrn_n_s64(productHigh, 30));
		if constexpr (kExcited) {
			sample = vqaddq_s32(sample, vld1q_dup_s32(in + i));
		}

		// SoftSaturator::process, branch-free: below the knee t is 0 and the magnitude passes through, above the
		// limit t is (kLimit - kKnee) and the curve gives exactly kMax
		const int32x4_t magnitude = vqabsq_s32(sample); // INT32_MIN saturates, and lands above the limit as before
		const int32x4_t t = vmaxq_s32(vsubq_s32(vminq_s32(magnitude, limit), knee), zero);
		const int32x4_t curve =
		    vsubq_s32(vaddq_s32(vminq_s32(magnitude, knee), t), vshlq_n_s32(vshrq_n_s32(vqdmulhq_s32(t, t), 1), 3));
		const int32x4_t sign = vshrq_n_s32(sample, 31);
		sample = vsubq_s32(veorq_s32(curve, sign), sign);
		sample = vmaxq_s32(vminq_s32(sample, maxLevel), minLevel);
		vst1q_s32(out + i * KarplusString::kGroupSize, sample);
		return sample;
	};

	// Padding lanes repeat the last string's buffer and state exactly, so they store the same value to the same place,
	// and need no branches
	const auto store = [&](uint32_t pos, int32x4_t sample) __attribute__((always_inline)) {
		vst1q_lane_s32(buffers[0] + pos, sample, 0);
		vst1q_lane_s32(buffers[1] + pos, sample, 1);
		vst1q_lane_s32(buffers[2] + pos, sample, 2);
		vst1q_lane_s32(buffers[3] + pos, sample, 3);
	};

	uint32_t pos = position;
	size_t i = 0;
	while (i < n) {
		if (pos < LoopLine::kGuard) {
			// The first few samples of the line are mirrored past its end (see FractionalDelay::write)
			const int32x4_t sample = process(read(pos), i);
			store(pos, sample);
			store(KarplusString::kLoopSize + pos, sample);
			++i;
			++pos;
			continue;
		}
		// Up to where the line wraps, with no wrap checks or guard writes
		size_t run = std::min<size_t>(n - i, KarplusString::kLoopSize - pos);
		if (paired) {
			for (; run >= 2; run -= 2, i += 2, pos += 2) {
				int32x4_t first = read(pos);
				int32x4_t second = read(pos + 1);
				// Left to itself GCC sinks the second read below the first sample's filters, back into one long chain.
				// Both reads must now finish before either sample's filters start, so the two reads overlap, and so do
				// the filters (sample two's only wait on sample one's filter states, not its whole chain).
#if defined(__arm__)
				asm("" : "+w"(first), "+w"(second));
#endif
				store(pos, process(first, i));
				store(pos + 1, process(second, i + 1));
			}
		}
		for (; run > 0; --run, ++i, ++pos) {
			store(pos, process(read(pos), i));
		}
		if (pos == KarplusString::kLoopSize) {
			pos = 0;
		}
	}

	vst1q_s64(&lanes.lowpassY[0], lowpassLow);
	vst1q_s64(&lanes.lowpassY[2], lowpassHigh);
	vst1q_s64(&lanes.highpassY[0], highpassLow);
	vst1q_s64(&lanes.highpassY[2], highpassHigh);
	vst1q_s32(lanes.highpassX, highpassX);
	position = pos;
}
} // namespace

bool KarplusString::renderPluckedGroup(std::span<KarplusString* const> strings, std::span<const q31_t> plucked,
                                       std::span<q31_t> out) {
	const size_t count = std::min(strings.size(), kGroupSize);
	const size_t n = std::min(out.size() / kGroupSize, kMaxBlock);
	if (count == 0 || n == 0) {
		return true;
	}
	uint32_t position = strings[0]->loop_.writePosition();
	for (size_t k = 1; k < count; ++k) {
		if (strings[k]->loop_.writePosition() != position) {
			return false;
		}
	}

	Lanes lanes;
	q31_t* buffers[kGroupSize];
	for (size_t lane = 0; lane < kGroupSize; ++lane) {
		// Padding lanes repeat the last string, so their reads stay in bounds
		KarplusString& string = *strings[std::min(lane, count - 1)];
		if (lane < count) {
			// As renderPlucked() does
			string.loop_.ensureZeroedFor((std::max(string.delayQ16_, string.targetDelayQ16_) >> 16) + 2,
			                             static_cast<uint32_t>(n));
		}
		buffers[lane] = string.loop_.data();
		lanes.lowpassY[lane] = string.damping_.y_;
		lanes.lowpassA[lane] = string.damping_.a_;
		lanes.highpassY[lane] = string.loopHp_.y1_;
		lanes.highpassX[lane] = string.loopHp_.x1_;
		lanes.highpassH[lane] = string.loopHp_.h_;
		lanes.gain[lane] = string.loopGain_;
		lanes.delay[lane] = static_cast<int32_t>(string.delayQ16_);
		lanes.step[lane] = (static_cast<int32_t>(string.targetDelayQ16_) - static_cast<int32_t>(string.delayQ16_))
		                   / static_cast<int32_t>(n);
	}

	// See renderLanes: two samples at a time unless a lane's delay comes within kMinPairedDelay samples
	int32_t shortestDelay = INT32_MAX;
	for (size_t lane = 0; lane < count; ++lane) {
		const int32_t end = lanes.delay[lane] + lanes.step[lane] * static_cast<int32_t>(n);
		shortestDelay = std::min({shortestDelay, lanes.delay[lane], end});
	}
	const bool paired = (shortestDelay >> 16) >= kMinPairedDelay;
	if (plucked.empty()) {
		renderLanes<false>(buffers, position, lanes, nullptr, out.data(), n, paired);
	}
	else {
		renderLanes<true>(buffers, position, lanes, plucked.data(), out.data(), n, paired);
	}

	for (size_t lane = 0; lane < count; ++lane) {
		KarplusString& string = *strings[lane];
		string.damping_.y_ = lanes.lowpassY[lane];
		string.loopHp_.y1_ = lanes.highpassY[lane];
		string.loopHp_.x1_ = lanes.highpassX[lane];
		string.loop_.setWritePosition(position);
		string.delayQ16_ = string.targetDelayQ16_;
	}
	return true;
}

} // namespace deluge::dsp::waveguide
