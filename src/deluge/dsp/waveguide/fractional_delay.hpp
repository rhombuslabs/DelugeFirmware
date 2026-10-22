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
#include <array>
#include <cstddef>
#include <cstdint>

namespace deluge::dsp::waveguide {

/// Circular delay line read at fractional delays. Delay 1 is the most recently written sample.
///
/// The first kGuard samples are mirrored past the end of the buffer, so the neighbouring taps of an interpolated read
/// are contiguous and a read needs one wrap check rather than one per tap. Hot loops can keep the write position in a
/// register by using the static read/write functions on data() (see KarplusString::render).
template <size_t kSize>
class FractionalDelay {
public:
	static_assert(kSize >= 4 && kSize <= 65535, "delays are Q16.16 and Lagrange reads need 4 taps");

	/// Lagrange reads need one newer and two older neighbours (Q16.16).
	static constexpr uint32_t kMinLagrangeDelay = 2u << 16;
	static constexpr uint32_t kMaxLagrangeDelay = (kSize - 3) << 16;
	/// Samples mirrored past the end: a Lagrange read reaches 3 beyond its oldest tap
	static constexpr uint32_t kGuard = 3;
	/// Lagrange weights: 1/3 and 1/6 in Q31
	static constexpr q31_t kThird = 715827883;
	static constexpr q31_t kSixth = 357913941;

	/// Starts lazily cleared, so constructing a line doesn't write the whole buffer
	FractionalDelay() { clearLazily(); }

	void clear() {
		buffer_.fill(0);
		write_ = 0;
		written_ = kSize;
	}

	/// Like clear(), but without touching the whole buffer up front: ensureZeroedFor() then zeroes, just in time,
	/// only the never-written samples that the next block's reads can reach. Reads give the same results as after
	/// clear(). This avoids clearing kilobytes per delay line when many notes start in the same audio block.
	void clearLazily() {
		write_ = 0;
		written_ = 0;
		zeroedFrom_ = kSize;
		for (uint32_t i = 0; i < kGuard; ++i) {
			buffer_[kSize + i] = 0;
		}
	}

	/// Call before each block of `blockLength` writes whose reads reach back at most `oldestDelay` samples (whole
	/// samples, including interpolation taps: a Lagrange read at delay d reaches floor(d) + 2, a linear one
	/// floor(d) + 1). Only does work until the line has been written all the way round once.
	void ensureZeroedFor(uint32_t oldestDelay, uint32_t blockLength) {
		if (written_ >= kSize) {
			return;
		}
		// Reads early in the block reach furthest back. Wrapped indices >= write_ have not been written since
		// clearLazily(); oldestDelay <= kSize keeps `lowest` at or above write_, so no written sample is zeroed.
		const uint32_t lowest = kSize + write_ - std::min<uint32_t>(oldestDelay, kSize);
		if (lowest < zeroedFrom_) {
			std::fill(buffer_.begin() + lowest, buffer_.begin() + zeroedFrom_, 0);
			zeroedFrom_ = lowest;
		}
		written_ += blockLength;
	}

	void write(q31_t x) { write(buffer_.data(), write_, x); }

	/// The sample written `delay` writes ago, 1 <= delay <= kSize.
	[[nodiscard]] q31_t at(uint32_t delay) const {
		int32_t index = static_cast<int32_t>(write_) - static_cast<int32_t>(delay);
		if (index < 0) {
			index += kSize;
		}
		return buffer_[index];
	}

	/// 4-point, 3rd-order Lagrange interpolation (Farrow form). delayQ16 in [kMinLagrangeDelay, kMaxLagrangeDelay].
	[[nodiscard]] q31_t readLagrange(uint32_t delayQ16) const { return readLagrange(buffer_.data(), write_, delayQ16); }

	/// Linear interpolation. delayQ16 in [1 << 16, (kSize - 1) << 16].
	[[nodiscard]] q31_t readLinear(uint32_t delayQ16) const { return readLinear(buffer_.data(), write_, delayQ16); }

	[[nodiscard]] q31_t* data() { return buffer_.data(); }
	[[nodiscard]] uint32_t writePosition() const { return write_; }
	void setWritePosition(uint32_t position) { write_ = position; }

	static void write(q31_t* buffer, uint32_t& position, q31_t x) {
		buffer[position] = x;
		if (position < kGuard) {
			buffer[kSize + position] = x;
		}
		position = (position + 1 == kSize) ? 0 : position + 1;
	}

	static q31_t readLagrange(const q31_t* buffer, uint32_t position, uint32_t delayQ16) {
		const uint32_t i = delayQ16 >> 16;
		const q31_t f = static_cast<q31_t>((delayQ16 & 0xFFFF) << 15);
		// Oldest tap first: delays i + 2, i + 1, i, i - 1. f runs from x0 (delay i) towards x1 (delay i + 1)
		const q31_t* taps = buffer + wrap(static_cast<int32_t>(position) - static_cast<int32_t>(i + 2));
		const q31_t x2 = taps[0];
		const q31_t x1 = taps[1];
		const q31_t x0 = taps[2];
		const q31_t xm1 = taps[3];
		const q31_t c1 = x1 - q31_mult(xm1, kThird) - (x0 >> 1) - q31_mult(x2, kSixth);
		const q31_t c2 = ((xm1 >> 1) + (x1 >> 1)) - x0;
		const q31_t c3 = q31_mult(x2 - xm1, kSixth) + ((x0 >> 1) - (x1 >> 1));
		return q31_mult(q31_mult(q31_mult(c3, f) + c2, f) + c1, f) + x0;
	}

	static q31_t readLinear(const q31_t* buffer, uint32_t position, uint32_t delayQ16) {
		const uint32_t i = delayQ16 >> 16;
		const q31_t f = static_cast<q31_t>((delayQ16 & 0xFFFF) << 15);
		const q31_t* taps = buffer + wrap(static_cast<int32_t>(position) - static_cast<int32_t>(i + 1));
		const q31_t a = taps[1]; // delay i
		return a + q31_mult(taps[0] - a, f);
	}

private:
	static uint32_t wrap(int32_t index) {
		return static_cast<uint32_t>(index < 0 ? index + static_cast<int32_t>(kSize) : index);
	}

	std::array<q31_t, kSize + kGuard> buffer_; ///< Deliberately not zero-initialised: see clearLazily()
	uint32_t write_ = 0;
	uint32_t written_ = kSize;    ///< Samples written since clearLazily(), until it reaches kSize
	uint32_t zeroedFrom_ = kSize; ///< Indices from here to kSize have been zeroed since clearLazily()
};

} // namespace deluge::dsp::waveguide
