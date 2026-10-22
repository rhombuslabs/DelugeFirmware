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
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>

namespace deluge::dsp::waveguide {

/// Full 32 x 32 -> 64-bit signed product as one `smull`. GCC otherwise tends to hoist a sign-extended copy of a
/// loop-invariant operand and then emit a 64 x 64 multiply (three multiplies plus spills) inside the loop.
inline int64_t multiply32x32(int32_t a, int32_t b) {
#if defined(__arm__)
	int32_t lo;
	int32_t hi;
	asm("smull %0, %1, %2, %3" : "=&r"(lo), "=&r"(hi) : "r"(a), "r"(b));
	return static_cast<int64_t>((static_cast<uint64_t>(static_cast<uint32_t>(hi)) << 32) | static_cast<uint32_t>(lo));
#else
	return static_cast<int64_t>(a) * b;
#endif
}

/// Converts a 0..1 float to q31, saturating at ONE_Q31.
inline q31_t toQ31(float unit) {
	unit = std::clamp(unit, 0.f, 1.f);
	if (unit >= 1.f) {
		return ONE_Q31;
	}
	return static_cast<q31_t>(unit * 2147483648.f);
}

/// y += a * (x - y). a = ONE_Q31 is (very nearly) a pass-through.
class OnePoleLowpass {
public:
	void setCoefficient(q31_t a) { a_ = a; }
	void reset() { y_ = 0; }
	q31_t process(q31_t x) {
		y_ += q31_mult(x - y_, a_);
		return y_;
	}
	/// True once the output has decayed below `threshold` (rounding can leave a small residue that never reaches 0)
	[[nodiscard]] bool isQuiet(q31_t threshold) const { return y_ < threshold && y_ > -threshold; }
	static q31_t coefficientForCutoff(float hz, float sampleRate) {
		return toQ31(1.f - std::exp(-2.f * std::numbers::pi_v<float> * hz / sampleRate));
	}

private:
	q31_t a_ = ONE_Q31;
	q31_t y_ = 0;
};

/// One-pole high-pass at ~5 Hz: y = x - x[n-1] + R * y[n-1].
class DcBlocker {
public:
	void reset() {
		x1_ = 0;
		y1_ = 0;
	}
	q31_t process(q31_t x) {
		const q31_t y = x - x1_ + q31_mult(y1_, kPole);
		x1_ = x;
		y1_ = y;
		return y;
	}
	/// True once the last input was 0 and the output has decayed below `threshold` (truncation leaves a residue)
	[[nodiscard]] bool isQuiet(q31_t threshold) const { return x1_ == 0 && y1_ < threshold && y1_ > -threshold; }

private:
	static constexpr q31_t kPole = 2145954558; // 1 - 2*pi*5/44100
	q31_t x1_ = 0;
	q31_t y1_ = 0;
};

/// Odd-symmetric soft clip: identity up to kKnee, then a quadratic knee reaching kMax (zero slope) at kLimit.
struct SoftSaturator {
	static constexpr q31_t kKnee = 1 << 28;
	static constexpr q31_t kLimit = 1 << 29;
	static constexpr q31_t kMax = kKnee + (kLimit - kKnee) / 2; // 0.75 * kLimit

	static q31_t process(q31_t x) {
		const bool negative = x < 0;
		const q31_t magnitude = negative ? (x == std::numeric_limits<q31_t>::min() ? kLimit : -x) : x;
		if (magnitude <= kKnee) {
			return x;
		}
		q31_t y = kMax;
		if (magnitude < kLimit) {
			const q31_t t = magnitude - kKnee;
			// y = knee + t - t^2 / kLimit
			y = kKnee + t - (multiply_32x32_rshift32(t, t) << 3);
		}
		return negative ? -y : y;
	}
};

} // namespace deluge::dsp::waveguide
