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
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>

namespace deluge::dsp::waveguide {

namespace {
constexpr float kTwoPi = 2.f * std::numbers::pi_v<float>;
constexpr float kLogMiddleC = 5.5669313f;      ///< ln(261.63 Hz)
constexpr float kLogT60Range = 6.3969297f;     ///< ln(600): sustain 0..1 spans 50 ms .. 30 s
constexpr float kLogKeyScale = 0.28768207f;    ///< ln(4/3): T60 x 0.75 per octave above middle C
constexpr float kLogSampleRate = 10.694215f;   ///< ln(44100)
constexpr float kLogLowestCutoff = 0.4054651f; ///< ln(1.5): the darkest loop lowpass is at 1.5 f0
constexpr float kLog1000 = 6.9077553f;         ///< 60 dB, in nepers
/// Floor on the loop high-pass corner, so DC and the subsonic mode the high-pass creates die within a few seconds
constexpr float kMinHighpassHz = 1.f;
/// Ceiling on the loop high-pass corner, as a fraction of f0
constexpr float kMaxHighpassRatio = 1.f / 3.f;
/// Ceiling on the loop gain of every mode other than the fundamental
constexpr float kMaxPartialGain = 0.99999f;
/// The gain is stored as gain * 2^30 in an int32, so it must stay below 2
constexpr float kMaxLoopGain = 1.999f;

float t60FromLog(float sustain, float logF0) {
	const float octavesAboveMiddleC = std::max(0.f, logF0 - kLogMiddleC) * std::numbers::log2e_v<float>;
	return 0.05f * std::exp(std::clamp(sustain, 0.f, 1.f) * kLogT60Range - octavesAboveMiddleC * kLogKeyScale);
}

/// |lowpass * high-pass|^2 on the unit circle, as a function of t = sin^2(w / 2). It has a single peak, at
/// t = a h / (4 sqrt(p r)).
struct DampingMagnitude {
	float a, p, h, r, k;
	[[nodiscard]] float operator()(float t) const {
		return a * a * k * k * 4.f * t / ((a * a + 4.f * p * t) * (h * h + 4.f * r * t));
	}
};

/// The 4-point Lagrange interpolator at fraction `f` past its integer delay. Its magnitude is <= 1 and falls with
/// frequency for every f in [0, 1).
struct Lagrange {
	float m1, m0, p1, p2; ///< Weights of delays i - 1, i, i + 1, i + 2
	explicit Lagrange(float f)
	    : m1(-f * (f - 1.f) * (f - 2.f) / 6.f), m0((f + 1.f) * (f - 1.f) * (f - 2.f) / 2.f),
	      p1(-(f + 1.f) * f * (f - 2.f) / 2.f), p2((f + 1.f) * f * (f - 1.f) / 6.f) {}
	/// Response relative to z^-i, at z^-1 = e (c1 - j s1), z^-2 = e^2 (c2 - j s2)
	[[nodiscard]] std::complex<float> response(float e, float c1, float s1, float c2, float s2) const {
		return {m0 + m1 * c1 / e + p1 * e * c1 + p2 * e * e * c2, m1 * s1 / e - p1 * e * s1 - p2 * e * e * s2};
	}
};
} // namespace

void StringExciter::reset() {
	tone_.reset();
	dc_.reset();
}

void StringExciter::setVelocity(float velocity) {
	const float cutoff = 500.f * std::pow(32.f, std::clamp(velocity, 0.f, 1.f)); // 500 Hz .. 16 kHz
	tone_.setCoefficient(OnePoleLowpass::coefficientForCutoff(cutoff, KarplusString::kSampleRate));
}

bool StringExciter::process(std::span<const q31_t> in, std::span<q31_t> out) {
	const size_t n = std::min(in.size(), out.size());
	const q31_t* __restrict input = in.data();
	if (dc_.isQuiet(kSettled) && tone_.isQuiet(kSettled)
	    && std::all_of(input, input + n, [](q31_t x) { return x == 0; })) {
		reset();
		return false;
	}
	// Local copies keep the state in registers (see KarplusString::renderConditioned)
	OnePoleLowpass tone = tone_;
	DcBlocker dc = dc_;
	q31_t* __restrict output = out.data();
	for (size_t i = 0; i < n; ++i) {
		output[i] = tone.process(dc.process(input[i] >> kShift));
	}
	tone_ = tone;
	dc_ = dc;
	return true;
}

KarplusString::KarplusString() {
	reset();
}

void KarplusString::reset() {
	// Lazily: a chord can start dozens of strings in one audio block, and clearing ~13 KB each at once caused CPU
	// spikes at note-on
	loop_.clearLazily();
	comb_.clearLazily();
	damping_.reset();
	loopHp_.reset();
	exciter_.reset();
	combQuietSamples_ = 0;
	primed_ = false;
}

void KarplusString::setVelocity(float velocity) {
	exciter_.setVelocity(velocity);
}

float KarplusString::targetT60(float sustain, float f0) {
	return t60FromLog(sustain, std::log(f0));
}

bool KarplusString::setParams(const StringParams& params) {
	if (haveParams_ && params.phaseIncrement == lastParams_.phaseIncrement && params.sustain == lastParams_.sustain
	    && params.brightness == lastParams_.brightness && params.pluckPosition == lastParams_.pluckPosition) {
		// A held note with no modulation: nothing to recompute
		if (!primed_) {
			delayQ16_ = targetDelayQ16_;
			primed_ = true;
		}
		return true;
	}
	if (params.phaseIncrement == 0) {
		return false;
	}
	float period = 4294967296.f / static_cast<float>(params.phaseIncrement);
	if (period < kMinPeriod) {
		return false;
	}
	period = std::min(period, kMaxPeriod);
	period_ = period;
	const float f0 = kSampleRate / period;
	const float logF0 = std::log(f0);
	const float w = kTwoPi / period;

	// Brightness: the loop lowpass tracks pitch, from 1.5 harmonics (dark) up to the sample rate (open)
	const float logLowest = logF0 + kLogLowestCutoff;
	const float cutoff = std::exp(logLowest + std::clamp(params.brightness, 0.f, 1.f) * (kLogSampleRate - logLowest));
	const float a = std::min(1.f, -std::expm1(-kTwoPi * cutoff / kSampleRate));
	const float p = 1.f - a;
	damping_.setCoefficient(a);

	const float halfSin = std::sin(0.5f * w);
	const float halfCos = std::cos(0.5f * w);
	const float s2 = halfSin * halfSin; // sin^2(w / 2): well conditioned at low pitch, where 1 - cos(w) is not
	const float sinW = 2.f * halfSin * halfCos;
	const float cosW = 1.f - 2.f * s2;

	// Loop high-pass corner, as a fraction c of f0. Its power loss is about c^2 at f0 and c^2 / 4 at 2 f0, while the
	// lowpass loses about q^2 at f0 and 4 q^2 at 2 f0. With c below 2q the harmonics still decay faster than f0 once
	// the loss at f0 is compensated, and c = q / 2 leaves margin. The subsonic mode the high-pass creates (below
	// f0 sqrt(c / 2 pi)) loses about 2 pi c of its power per pass, which must beat the compensated loss at f0: the
	// lowpass's q^2 plus up to `interpolatorLoss` (the Lagrange interpolator at a half-sample fraction, which matters
	// at high pitch). The 1 Hz floor makes DC die quickly on bright low strings, where both are tiny.
	const float q = w * std::sqrt(p) / a;
	const float halfSampleLagrange = halfCos * (1.5f - 0.5f * halfCos * halfCos);
	const float interpolatorLoss = 1.f - halfSampleLagrange * halfSampleLagrange;
	const float c = std::clamp(std::max(0.5f * q, (q * q + interpolatorLoss) / std::numbers::pi_v<float>),
	                           kMinHighpassHz / f0, kMaxHighpassRatio);
	const float h = -std::expm1(-c * w);
	const float r = 1.f - h;
	const float k = 0.5f * (1.f + r);
	loopHp_.setLeak(h);

	// Loop response at the decaying fundamental z = e^(sigma + jw), sigma from the target T60. (On the unit circle the
	// gain needed is off by sigma times the filters' group delay, which matters at low pitch and short T60.)
	const float sigma = -kLog1000 / (kSampleRate * t60FromLog(params.sustain, logF0));
	const float u = -sigma;                                   // <= 0.015
	const float eps = u * (1.f + 0.5f * u * (1.f + u / 3.f)); // |z^-1| - 1 = e^-sigma - 1
	const float e = 1.f + eps;
	// Lowpass a / (1 - p z^-1) and high-pass k (1 - z^-1) / (1 - r z^-1), arranged so nothing cancels when p, r, e ~ 1
	const float lpRe = (a - eps * p) + 2.f * p * e * s2;
	const float lpIm = p * e * sinW;
	const float hpNumRe = -eps + 2.f * e * s2;
	const float hpNumIm = e * sinW;
	const float hpDenRe = (h - eps * r) + 2.f * r * e * s2;
	const float hpDenIm = r * e * sinW;
	const float denRe = lpRe * hpDenRe - lpIm * hpDenIm;
	const float denIm = lpRe * hpDenIm + lpIm * hpDenRe;
	const float phase = std::atan2(hpNumIm * denRe - hpNumRe * denIm, hpNumRe * denRe + hpNumIm * denIm);
	const float dampingSquared =
	    a * a * k * k * (hpNumRe * hpNumRe + hpNumIm * hpNumIm) / (denRe * denRe + denIm * denIm);

	// Tuning: the damping's phase lag (or the high-pass's lead) comes off (or goes onto) the line length
	constexpr float kMinDelay = 2.f;
	constexpr float kMaxDelay = static_cast<float>(kLoopSize - 4);
	float delay = std::clamp(period + phase / w, kMinDelay, kMaxDelay);
	// The interpolator's phase delay at f0 strays from its fraction at high pitch. Uncorrected, that moves the mode
	// off f0, and the interpolator's steep magnitude there then misses the target T60 badly (8 kHz: 40 s for 7 s).
	const float cos2W = 1.f - 2.f * sinW * sinW;
	const float sin2W = 2.f * sinW * cosW;
	const float firstFraction = delay - std::floor(delay);
	const float interpolatorDelay = -std::arg(Lagrange{firstFraction}.response(e, cosW, sinW, cos2W, sin2W)) / w;
	delay = std::clamp(delay - (interpolatorDelay - firstFraction), kMinDelay, kMaxDelay);
	targetDelayQ16_ = static_cast<uint32_t>(delay * 65536.f);
	if (!primed_) {
		delayQ16_ = targetDelayQ16_;
		primed_ = true;
	}
	const uint32_t whole = targetDelayQ16_ >> 16;
	const Lagrange lagrange{static_cast<float>(targetDelayQ16_ & 0xFFFF) / 65536.f};

	// Sustain: the per-pass gain that decays the fundamental 60 dB in targetT60, compensating the damping's and the
	// interpolator's loss at f0
	const float loopSquared = dampingSquared * std::norm(lagrange.response(e, cosW, sinW, cos2W, sin2W));
	float gain = std::exp(sigma * static_cast<float>(whole)) / std::sqrt(loopSquared);

	// That can exceed 1, so cap it to keep every other mode of the loop below unity gain:
	// - harmonics 2, 3, ...: at most the damping at 2 f0 (or at its peak, if that is higher), times the interpolator
	//   at 2 f0.
	// - the subsonic mode from the high-pass: it lies below f0 sqrt(c / 2 pi), padded 20% here for the line-length
	//   and lowpass shifts, where the damping is still rising (or else at most its peak).
	const DampingMagnitude damping{a, p, h, r, k};
	const float tPeak = (p > 0.f) ? std::min(1.f, a * h / (4.f * std::sqrt(p * r))) : 1.f;
	const float cos4W = 1.f - 2.f * sin2W * sin2W;
	const float sin4W = 2.f * sin2W * cos2W;
	const float harmonics =
	    damping(std::max(sinW * sinW, tPeak)) * std::norm(lagrange.response(1.f, cos2W, sin2W, cos4W, sin4W));
	const float subsonicHalfAngle = 0.6f * w * std::sqrt(c / kTwoPi);
	const float subsonic = damping(std::min(subsonicHalfAngle * subsonicHalfAngle, tPeak));
	gain = std::min({gain, kMaxPartialGain / std::sqrt(std::max(harmonics, subsonic)), kMaxLoopGain});
	loopGain_ = static_cast<int32_t>(gain * 1073741824.f);

	const float beta = 0.05f + 0.45f * std::clamp(params.pluckPosition, 0.f, 1.f);
	combDelayQ16_ = static_cast<uint32_t>(std::clamp(beta * period, 1.f, static_cast<float>(kCombSize - 2)) * 65536.f);

	lastParams_ = params;
	haveParams_ = true;
	return true;
}

void KarplusString::render(std::span<const q31_t> excitation, std::span<q31_t> out) {
	const size_t n = std::min(excitation.size(), out.size());
	q31_t conditioned[kMaxBlock];
	for (size_t start = 0; start < n; start += kMaxBlock) {
		const size_t length = std::min(kMaxBlock, n - start);
		const bool excited = exciter_.process(excitation.subspan(start, length), {conditioned, length});
		renderConditioned(excited ? std::span<const q31_t>{conditioned, length} : std::span<const q31_t>{},
		                  out.subspan(start, length));
	}
}

void KarplusString::renderConditioned(std::span<const q31_t> conditioned, std::span<q31_t> out) {
	q31_t plucked[kMaxBlock];
	for (size_t start = 0; start < out.size(); start += kMaxBlock) {
		const size_t length = std::min(kMaxBlock, out.size() - start);
		const bool active =
		    pluck(conditioned.empty() ? conditioned : conditioned.subspan(start, length), {plucked, length});
		renderPlucked(active ? std::span<const q31_t>{plucked, length} : std::span<const q31_t>{},
		              out.subspan(start, length));
	}
}

bool KarplusString::pluck(std::span<const q31_t> conditioned, std::span<q31_t> out) {
	const size_t n = out.size();
	const bool silent = conditioned.empty();
	// Once a full comb length of silence has gone in, the comb holds only zeros and its output is exactly zero
	constexpr uint32_t kCombFlushed = kCombSize + 3;
	if (n == 0 || (silent && combQuietSamples_ >= kCombFlushed)) {
		return false;
	}
	// Zero any never-written samples this block's reads can reach (see FractionalDelay::clearLazily)
	comb_.ensureZeroedFor((combDelayQ16_ >> 16) + 1, static_cast<uint32_t>(n));

	// Local copies keep the state in registers (see renderPlucked)
	using CombLine = FractionalDelay<kCombSize>;
	q31_t* __restrict combBuffer = comb_.data();
	uint32_t combPosition = comb_.writePosition();
	const uint32_t combDelay = combDelayQ16_;
	const q31_t* __restrict in = conditioned.data();
	q31_t* __restrict output = out.data();
	for (size_t i = 0; i < n; ++i) {
		const q31_t toned = silent ? 0 : in[i];
		// Pluck position: cancelling the exciter against itself delayed by beta * period
		// notches the harmonics that have a node at the pluck point
		output[i] = toned - CombLine::readLinear(combBuffer, combPosition, combDelay);
		CombLine::write(combBuffer, combPosition, toned);
	}
	combQuietSamples_ = silent ? std::min<uint32_t>(combQuietSamples_ + n, kCombFlushed) : 0;
	comb_.setWritePosition(combPosition);
	return true;
}

void KarplusString::renderPlucked(std::span<const q31_t> plucked, std::span<q31_t> out) {
	const size_t n = std::min(out.size(), kMaxBlock);
	if (n == 0) {
		return;
	}
	const bool excited = !plucked.empty();

	// Glide the line length across the block so pitch changes don't step
	const int32_t delayStep =
	    (static_cast<int32_t>(targetDelayQ16_) - static_cast<int32_t>(delayQ16_)) / static_cast<int32_t>(n);
	int32_t delay = static_cast<int32_t>(delayQ16_);

	// Zero any never-written samples this block's reads can reach (see FractionalDelay::clearLazily)
	loop_.ensureZeroedFor((std::max(delayQ16_, targetDelayQ16_) >> 16) + 2, static_cast<uint32_t>(n));

	// Work on local copies so the state stays in registers: `out` is int32_t too, so the compiler must otherwise
	// assume every output store could change the members, and reload and re-store them all each sample
	using LoopLine = FractionalDelay<kLoopSize>;
	q31_t* __restrict loopBuffer = loop_.data();
	uint32_t loopPosition = loop_.writePosition();
	LoopLowpass damping = damping_;
	LoopHighpass loopHp = loopHp_;
	const int32_t loopGain = loopGain_;
	const q31_t* __restrict in = plucked.data();
	q31_t* __restrict output = out.data();

	for (size_t i = 0; i < n; ++i) {
		delay += delayStep;
		const q31_t exciter = excited ? in[i] : 0;
		const q31_t damped = loopHp.process(
		    damping.process(LoopLine::readLagrange(loopBuffer, loopPosition, static_cast<uint32_t>(delay))));
		// Rounded toward zero: rounding to nearest would let a gain just under 1 hold small signals forever, and
		// rounding down would lose more on one sign than the other
		// (branch-free: negative products get 2^30 - 1 added before the floor shift). No clamp is needed: loop
		// samples are within kMax, the Lagrange taps' weights sum to at most 1.25 in magnitude, the lowpass can't
		// grow the signal and the highpass at most doubles it, so |damped| < 2^30; with loopGain < 2^31 the result
		// is below 2^31
		const int64_t product = multiply32x32(damped, loopGain);
		const q31_t returned = static_cast<q31_t>((product + ((product >> 63) & ((int64_t{1} << 30) - 1))) >> 30);
		// The saturator's rounding can overshoot kMax by a few LSB
		const q31_t sample = std::clamp(SoftSaturator::process(add_saturate(returned, exciter)), -SoftSaturator::kMax,
		                                SoftSaturator::kMax);
		LoopLine::write(loopBuffer, loopPosition, sample);
		output[i] = sample;
	}

	loop_.setWritePosition(loopPosition);
	damping_ = damping;
	loopHp_ = loopHp;
	delayQ16_ = targetDelayQ16_;
}

} // namespace deluge::dsp::waveguide
