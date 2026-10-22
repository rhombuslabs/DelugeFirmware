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

#include "dsp/waveguide/filters.hpp"
#include "dsp/waveguide/fractional_delay.hpp"
#include "util/fixedpoint.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace deluge::dsp::waveguide {

struct StringParams {
	uint32_t phaseIncrement; ///< Pitch, 2^32 == sample rate (the Voice's phase increment)
	float sustain;           ///< 0..1, decay time (FEEDBACK)
	float brightness;        ///< 0..1, loop damping (WAVETBL)
	float pluckPosition;     ///< 0..1, bridge .. mid-string (PW)
};

/// Conditions the raw exciter (OSC1 + noise after ENV2) before it enters strings: -12 dB headroom, a DC blocker, and a
/// velocity-dependent lowpass (softer is darker). A voice runs one of these and feeds the result to all of its unison
/// strings, which would otherwise each repeat identical work.
class StringExciter {
public:
	static constexpr int32_t kShift = 2;       ///< -12 dB input headroom
	static constexpr q31_t kSettled = 1 << 13; ///< Below this (about -108 dBFS) a silent exciter counts as settled

	void reset();
	/// Note-on velocity, 0..1
	void setVelocity(float velocity);
	/// Conditions `in` into `out` (same length). Returns false, and leaves `out` untouched, when the block is silent:
	/// all-zero input with the filters settled. Settling snaps their rounding residue to exactly zero.
	[[nodiscard]] bool process(std::span<const q31_t> in, std::span<q31_t> out);

private:
	OnePoleLowpass tone_;
	DcBlocker dc_;
};

/// Karplus-Strong string excited by an external signal, which is added into the loop every sample.
class KarplusString {
public:
	static constexpr float kSampleRate = 44100.f;
	static constexpr float kMaxPeriod = 2205.f; ///< 20 Hz floor
	static constexpr float kMinPeriod = 4.41f;  ///< 10 kHz ceiling
	static constexpr size_t kLoopSize = 2212;   ///< kMaxPeriod + interpolation guard
	static constexpr size_t kCombSize = 1108;   ///< Half of kMaxPeriod + guard
	static constexpr size_t kMaxBlock = 128;    ///< render() works in sub-blocks of this size (stack scratch)

	/// User-provided, so `new (memory) KarplusString()` doesn't zero-initialise the ~13 KB of delay lines first
	KarplusString();
	void reset();
	/// Note-on velocity, 0..1: softer is darker.
	void setVelocity(float velocity);
	/// Call once per block before render(). Returns false when the pitch is above the ceiling; don't render then.
	[[nodiscard]] bool setParams(const StringParams& params);
	/// Renders with a raw exciter, conditioned by this string's own StringExciter.
	void render(std::span<const q31_t> excitation, std::span<q31_t> out);
	/// Renders `out.size()` samples with an exciter already conditioned by a StringExciter. An empty `conditioned`
	/// means a silent exciter (StringExciter::process returned false). Same as pluck() then renderPlucked().
	void renderConditioned(std::span<const q31_t> conditioned, std::span<q31_t> out);

	/// The two halves of renderConditioned(), so one string's pluck-position comb can serve a voice's other unison
	/// strings: they are detuned by cents, and the comb's notches don't need to follow that.
	/// pluck() runs the comb on a conditioned exciter (empty when silent) into `out`, at this string's pluck position
	/// and period (call after setParams). Returns false, leaving `out` untouched, once the comb has nothing left to
	/// output: then pass renderPlucked() an empty span.
	[[nodiscard]] bool pluck(std::span<const q31_t> conditioned, std::span<q31_t> out);
	/// Renders `out.size()` samples (at most kMaxBlock) of the string, excited by pluck()'s output (empty when silent)
	void renderPlucked(std::span<const q31_t> plucked, std::span<q31_t> out);

	static constexpr size_t kGroupSize = 4; ///< Strings per renderPluckedGroup() call: one per NEON lane
	/// renderPlucked() for up to kGroupSize strings at once, all excited by the same `plucked` (empty when silent),
	/// with identical results. `out` is interleaved: string k's sample i goes to out[i * kGroupSize + k]; out.size() /
	/// kGroupSize samples are rendered (at most kMaxBlock). The strings' delay lines must be at the same write
	/// position, as a voice's unison strings are, having started together; returns false, rendering nothing, if not.
	[[nodiscard]] static bool renderPluckedGroup(std::span<KarplusString* const> strings,
	                                             std::span<const q31_t> plucked, std::span<q31_t> out);

	/// Decay time of the fundamental, in seconds.
	static float targetT60(float sustain, float f0);
	[[nodiscard]] float period() const { return period_; }

private:
	// The loop filters keep 32 fractional bits of state. With q31 truncation, a pole near 1 turns the rounding error
	// of every sample into a friction-like loss that grows as the string gets quieter (measured: T60 15% short at
	// 110 Hz, and the decay accelerating as the level fell).

	/// Loop lowpass y += a (x - y), a in [0, 1].
	class LoopLowpass {
	public:
		void reset() { y_ = 0; }
		void setCoefficient(float a) { a_ = (a >= 1.f) ? INT32_MAX : static_cast<int32_t>(a * 2147483648.f); }
		q31_t process(q31_t x) {
			// 32-bit coefficient, so this is a single smull on ARM
			y_ += multiply32x32(x - output(), a_) << 1;
			return output();
		}

	private:
		friend class KarplusString; // renderPluckedGroup() runs this filter on NEON lanes
		[[nodiscard]] q31_t output() const { return static_cast<q31_t>((y_ + (int64_t{1} << 31)) >> 32); }
		int32_t a_ = INT32_MAX; ///< a * 2^31
		int64_t y_ = 0;         ///< y * 2^32
	};

	/// One-pole high-pass in the loop. It stops DC building up even though the damping loss is compensated with a gain
	/// that can exceed 1. Its corner tracks the loop lowpass (see setParams).
	/// y = k (x - x1) + r y1, r = 1 - h, k = (1 + r) / 2 = 1 - h / 2.
	class LoopHighpass {
	public:
		void reset() {
			x1_ = 0;
			y1_ = 0;
		}
		/// h = 1 - pole, in [0, 0.5)
		void setLeak(float h) { h_ = static_cast<int32_t>(h * 4294967296.f); }
		q31_t process(q31_t x) {
			// 32-bit operands (|d| < 2^30, |y| < 2^30, h < 2^31), so each product is a single smull on ARM
			const q31_t d = x - x1_;
			const q31_t y = static_cast<q31_t>(y1_ >> 32);
			y1_ += (static_cast<int64_t>(d) << 32) - (multiply32x32(d, h_) >> 1) - multiply32x32(y, h_);
			x1_ = x;
			return static_cast<q31_t>((y1_ + (int64_t{1} << 31)) >> 32);
		}

	private:
		friend class KarplusString; // renderPluckedGroup() runs this filter on NEON lanes
		int32_t h_ = 0;             ///< h * 2^32, h < 0.5
		q31_t x1_ = 0;
		int64_t y1_ = 0; ///< y * 2^32
	};

	FractionalDelay<kLoopSize> loop_;
	FractionalDelay<kCombSize> comb_;
	LoopLowpass damping_;
	LoopHighpass loopHp_;
	StringExciter exciter_; ///< Used by render() only; a Voice conditions once for all its strings
	/// Consecutive silent samples written to the comb; once it holds only zeros the comb pass is skipped
	uint32_t combQuietSamples_ = 0;
	int32_t loopGain_ = 0; ///< Per-pass gain * 2^30, so compensation for the damping loss can exceed 1
	uint32_t delayQ16_ = 0;
	uint32_t targetDelayQ16_ = 0;
	uint32_t combDelayQ16_ = 1u << 16;
	float period_ = kMaxPeriod;
	bool primed_ = false;
	/// The last accepted params; setParams skips the recompute when they are unchanged
	StringParams lastParams_{};
	bool haveParams_ = false;
};

} // namespace deluge::dsp::waveguide
