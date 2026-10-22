#include "dsp/waveguide/filters.hpp"
#include "dsp/waveguide/karplus_string.hpp"

#include "cppspec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <memory>
#include <numbers>
#include <random>
#include <type_traits>
#include <vector>

namespace wg = deluge::dsp::waveguide;

namespace {

constexpr double kFs = 44100.0;
constexpr double kPi = std::numbers::pi;
constexpr q31_t kImpulse = 1 << 28;

uint32_t phaseIncrementFor(double hz) {
	return static_cast<uint32_t>(hz / kFs * 4294967296.0);
}

std::unique_ptr<wg::KarplusString> makeString(double hz, float sustain, float brightness, float pluck = 0.f) {
	auto string = std::make_unique<wg::KarplusString>();
	string->reset();
	string->setVelocity(1.f);
	(void)string->setParams({phaseIncrementFor(hz), sustain, brightness, pluck});
	return string;
}

/// Renders in 128-sample blocks. `excitation` is zero-padded; `paramsPerBlock`, if set, is called before each block.
template <typename OnBlock = std::nullptr_t>
std::vector<double> render(wg::KarplusString& string, double seconds, const std::vector<q31_t>& excitation,
                           OnBlock onBlock = nullptr) {
	const size_t total = static_cast<size_t>(seconds * kFs);
	std::vector<double> out;
	out.reserve(total);
	std::array<q31_t, 128> in{};
	std::array<q31_t, 128> block{};
	for (size_t pos = 0; pos < total; pos += 128) {
		const size_t n = std::min<size_t>(128, total - pos);
		if constexpr (!std::is_same_v<OnBlock, std::nullptr_t>) {
			onBlock(pos / kFs);
		}
		for (size_t i = 0; i < n; ++i) {
			in[i] = (pos + i < excitation.size()) ? excitation[pos + i] : 0;
		}
		string.render({in.data(), n}, {block.data(), n});
		for (size_t i = 0; i < n; ++i) {
			out.push_back(block[i]);
		}
	}
	return out;
}

/// Hann-windowed DFT at `hz`, phase referenced to absolute sample index.
std::complex<double> dft(const std::vector<double>& x, size_t start, size_t length, double hz) {
	std::complex<double> sum{};
	const double w = 2 * kPi * hz / kFs;
	for (size_t i = 0; i < length; ++i) {
		const double hann = 0.5 - 0.5 * std::cos(2 * kPi * i / (length - 1));
		sum += x[start + i] * hann * std::polar(1.0, -w * static_cast<double>(start + i));
	}
	return sum;
}

double levelDb(const std::vector<double>& x, size_t start, size_t length, double hz) {
	return 20 * std::log10(std::abs(dft(x, start, length, hz)) + 1e-9);
}

/// Refines a frequency guess from the phase advance of the DFT between two windows, coarse to fine.
double measureFrequency(const std::vector<double>& x, double guess, double fromSeconds) {
	const size_t window = static_cast<size_t>(std::max(8.0 * kFs / guess, 2048.0));
	const size_t start = static_cast<size_t>(fromSeconds * kFs);
	double f = guess;
	for (double gapSeconds : {0.01, 0.05, 0.25}) {
		const size_t gap = static_cast<size_t>(gapSeconds * kFs);
		const double advance = std::arg(dft(x, start + gap, window, f) / dft(x, start, window, f));
		f += advance * kFs / (2 * kPi * gap);
	}
	return f;
}

/// T60 of the component at f0, from a linear fit of its level (dB) against time.
double measureT60(const std::vector<double>& x, double f0, double fromSeconds, double toSeconds) {
	const size_t window = static_cast<size_t>(std::max(4.0 * kFs / f0, 1024.0));
	std::vector<double> t;
	std::vector<double> db;
	for (size_t start = static_cast<size_t>(fromSeconds * kFs); start + window < toSeconds * kFs; start += window / 2) {
		t.push_back((start + window / 2) / kFs);
		db.push_back(levelDb(x, start, window, f0));
	}
	const double n = static_cast<double>(t.size());
	double st = 0, sd = 0, stt = 0, std_ = 0;
	for (size_t i = 0; i < t.size(); ++i) {
		st += t[i];
		sd += db[i];
		stt += t[i] * t[i];
		std_ += t[i] * db[i];
	}
	const double slope = (n * std_ - st * sd) / (n * stt - st * st);
	return -60.0 / slope;
}

double maxSecondDifference(const std::vector<double>& x) {
	double worst = 0;
	for (size_t i = 2; i < x.size(); ++i) {
		worst = std::max(worst, std::abs(x[i] - 2 * x[i - 1] + x[i - 2]));
	}
	return worst;
}

/// Energy about the mean between two times, so a constant offset doesn't count.
double acEnergy(const std::vector<double>& x, double fromSeconds, double toSeconds) {
	const size_t from = static_cast<size_t>(fromSeconds * kFs);
	const size_t to = std::min(x.size(), static_cast<size_t>(toSeconds * kFs));
	double mean = 0;
	for (size_t i = from; i < to; ++i) {
		mean += x[i];
	}
	mean /= static_cast<double>(to - from);
	double energy = 0;
	for (size_t i = from; i < to; ++i) {
		energy += (x[i] - mean) * (x[i] - mean);
	}
	return energy;
}

std::vector<q31_t> noise(size_t length, q31_t level, uint32_t seed) {
	std::mt19937 rng{seed};
	std::uniform_int_distribution<int32_t> dist{-level, level};
	std::vector<q31_t> out(length);
	for (auto& s : out) {
		s = dist(rng);
	}
	return out;
}

} // namespace

// clang-format off
describe karplus_string("KarplusString", $ {
	it("is silent after reset with no excitation", _ {
		auto string = makeString(220, 1.f, 0.5f);
		const auto out = render(*string, 0.1, {});
		expect(*std::max_element(out.begin(), out.end())).to_equal(0.0);
	});

	it("tunes within 3 cents from 29 Hz to 4 kHz at several brightnesses", _ {
		for (double hz : {29.14, 55.0, 110.0, 220.0, 440.0, 880.0, 1760.0, 3520.0, 4000.0}) {
			for (float brightness : {0.2f, 0.5f, 0.9f}) {
				auto string = makeString(hz, 1.f, brightness);
				const auto out = render(*string, 1.0, {kImpulse});
				const double cents = 1200 * std::log2(measureFrequency(out, hz, 0.2) / hz);
				expect(std::abs(cents)).to_be_less_than(3.0);
			}
		}
	});

	it("decays the fundamental at the target T60 (within 10%)", _ {
		for (double hz : {29.14, 110.0, 440.0, 2000.0}) {
			const float sustain = 0.6f;
			const double target = wg::KarplusString::targetT60(sustain, static_cast<float>(hz));
			auto string = makeString(hz, sustain, 0.5f);
			const auto out = render(*string, 0.7 * target, {kImpulse});
			const double measured = measureT60(out, hz, 0.1 * target, 0.6 * target);
			expect(std::abs(measured / target - 1.0)).to_be_less_than(0.1);
		}
	});

	it("shortens T60 by 0.75 per octave above middle C", _ {
		const float low = wg::KarplusString::targetT60(0.5f, 261.63f);
		const float high = wg::KarplusString::targetT60(0.5f, 523.26f);
		expect(std::abs(high / low - 0.75f)).to_be_less_than(0.001f);
		expect(std::abs(wg::KarplusString::targetT60(0.f, 100.f) - 0.05f)).to_be_less_than(1e-5f);
		expect(std::abs(wg::KarplusString::targetT60(1.f, 100.f) - 30.f)).to_be_less_than(1e-3f);
	});

	it("stays bounded and centred under 60 s of full-scale excitation at max sustain", _ {
		auto string = makeString(110, 1.f, 1.f);
		const auto excitation = noise(static_cast<size_t>(60 * kFs), 1 << 30, 7);
		const auto out = render(*string, 60.0, excitation);
		double peak = 0;
		for (double s : out) {
			peak = std::max(peak, std::abs(s));
		}
		expect(peak <= wg::SoftSaturator::kMax).to_be_true();
		double mean = 0;
		for (size_t i = out.size() - 44100; i < out.size(); ++i) {
			mean += out[i];
		}
		mean /= 44100;
		expect(std::abs(mean)).to_be_less_than(0.01 * wg::SoftSaturator::kMax);
	});

	it("decays a bright fundamental at its target T60, and no faster than the 2nd harmonic", _ {
		const double hz = 440;
		const float sustain = 0.8f;
		const double target = wg::KarplusString::targetT60(sustain, static_cast<float>(hz));
		auto string = makeString(hz, sustain, 1.f);
		const auto out = render(*string, 0.7 * target, {kImpulse});
		const double fundamental = measureT60(out, hz, 0.1 * target, 0.6 * target);
		const double second = measureT60(out, 2 * hz, 0.1 * target, 0.6 * target);
		expect(std::abs(fundamental / target - 1.0)).to_be_less_than(0.15);
		expect(fundamental >= second).to_be_true();
	});

	it("loses energy after an impulse at max sustain, from 20 Hz to 10 kHz at any brightness", _ {
		for (double hz : {20.0, 110.0, 1000.0, 10000.0}) {
			for (float brightness : {0.f, 0.5f, 1.f}) {
				auto string = makeString(hz, 1.f, brightness);
				const auto out = render(*string, 4.0, {kImpulse});
				expect(acEnergy(out, 3.5, 4.0)).to_be_less_than(acEnergy(out, 0.5, 1.0));
			}
		}
	});

	it("loses energy at max sustain when the pitch sweeps while ringing", _ {
		for (float brightness : {0.f, 1.f}) {
			auto string = makeString(20, 1.f, brightness);
			const auto out = render(*string, 4.0, {kImpulse}, [&](double t) {
				// 20 Hz -> 10 kHz -> 20 Hz, exponential, over 2 s, then held
				const double position = t < 1.0 ? t : std::max(0.0, 2.0 - t);
				const double hz = 20.0 * std::pow(10000.0 / 20.0, position);
				(void)string->setParams({phaseIncrementFor(hz), 1.f, brightness, 0.f});
			});
			expect(acEnergy(out, 3.5, 4.0)).to_be_less_than(acEnergy(out, 0.0, 0.5));
		}
	});

	it("does not latch onto a DC offset in the excitation", _ {
		auto string = makeString(110, 1.f, 1.f);
		std::vector<q31_t> excitation(static_cast<size_t>(5 * kFs), 1 << 28);
		const auto out = render(*string, 5.0, excitation);
		double mean = 0;
		for (size_t i = out.size() - 44100; i < out.size(); ++i) {
			mean += out[i];
		}
		mean /= 44100;
		expect(std::abs(mean)).to_be_less_than(0.01 * wg::SoftSaturator::kMax);
	});

	it("refuses pitches above 10 kHz and clamps pitches below 20 Hz", _ {
		wg::KarplusString string;
		string.reset();
		expect(string.setParams({phaseIncrementFor(15000), 0.5f, 0.5f, 0.f})).to_be_false();
		expect(string.setParams({phaseIncrementFor(5), 0.5f, 0.5f, 0.f})).to_be_true();
		expect(string.period()).to_equal(wg::KarplusString::kMaxPeriod);
		expect(string.setParams({0, 0.5f, 0.5f, 0.f})).to_be_false();
	});

	it("glides without discontinuities when the pitch sweeps", _ {
		auto still = makeString(1000, 1.f, 0.7f);
		const double bound = maxSecondDifference(render(*still, 2.0, {kImpulse}));

		auto swept = makeString(29.14, 1.f, 0.7f);
		const auto out = render(*swept, 2.0, {kImpulse}, [&](double t) {
			// 29 Hz -> 1 kHz -> 29 Hz, exponential, over 2 s
			const double position = t < 1.0 ? t : 2.0 - t;
			const double hz = 29.14 * std::pow(1000.0 / 29.14, position);
			(void)swept->setParams({phaseIncrementFor(hz), 1.f, 0.7f, 0.f});
		});
		expect(maxSecondDifference(out) <= 2.0 * bound).to_be_true();
	});

	it("adds energy to a ringing string when re-excited without reset", _ {
		auto string = makeString(220, 1.f, 0.5f);
		const auto first = render(*string, 0.1, {kImpulse});
		double before = 0;
		for (size_t i = first.size() - 441; i < first.size(); ++i) {
			before += first[i] * first[i];
		}
		const auto second = render(*string, 0.01, {kImpulse});
		double after = 0;
		for (double s : second) {
			after += s * s;
		}
		expect(after).to_be_greater_than(before);
	});

	it("renders bit-identically for the same excitation", _ {
		const auto excitation = noise(4410, 1 << 28, 3);
		auto a = makeString(330, 0.7f, 0.6f, 0.3f);
		auto b = makeString(330, 0.7f, 0.6f, 0.3f);
		expect(render(*a, 0.5, excitation) == render(*b, 0.5, excitation)).to_be_true();
	});

	it("can share one string's pluck comb: a string fed another's pluck() output matches it", _ {
		// A voice combs once and excites all its unison strings with the result
		const auto excitation = noise(4410, 1 << 28, 5);
		auto owner = makeString(330, 0.7f, 0.6f, 0.3f);
		auto follower = makeString(330, 0.7f, 0.6f, 0.3f);
		wg::StringExciter exciter;
		exciter.reset();
		exciter.setVelocity(1.f);
		std::array<q31_t, 128> in{};
		std::array<q31_t, 128> conditioned{};
		std::array<q31_t, 128> plucked{};
		std::array<q31_t, 128> a{};
		std::array<q31_t, 128> b{};
		bool same = true;
		size_t activeBlocks = 0;
		for (size_t pos = 0; pos < 22050; pos += 128) {
			for (size_t i = 0; i < 128; ++i) {
				in[i] = (pos + i < excitation.size()) ? excitation[pos + i] : 0;
			}
			const bool excited = exciter.process(in, conditioned);
			const bool active =
			    owner->pluck(excited ? std::span<const q31_t>{conditioned} : std::span<const q31_t>{}, plucked);
			activeBlocks += active;
			const std::span<const q31_t> shared = active ? std::span<const q31_t>{plucked} : std::span<const q31_t>{};
			owner->renderPlucked(shared, a);
			follower->renderPlucked(shared, b);
			same = same && a == b;
		}
		expect(same).to_be_true();
		// The comb goes quiet once the exciter has been silent for a comb length
		expect(activeBlocks).to_be_less_than(22050 / 128);
	});

	it("renders a group of strings on NEON lanes bit-identically to one at a time", _ {
		// Pitches from the 20 Hz floor to near the 10 kHz ceiling (where the Lagrange taps reach the newest samples),
		// gliding in the second half; odd block lengths; excited, then silent. The second set starts a lane right at
		// the ceiling, where the delay is shortest (about 4.4 samples, still long enough to interleave two samples)
		const double pitchSets[2][4] = {{25.0, 110.0, 1234.5, 9000.0}, {9950.0, 3000.0, 440.0, 61.0}};
		bool same = true;
		for (size_t combination = 0; combination < 2 * wg::KarplusString::kGroupSize; ++combination) {
			const size_t count = 1 + combination % wg::KarplusString::kGroupSize;
			const double* pitches = pitchSets[combination / wg::KarplusString::kGroupSize];
			std::vector<std::unique_ptr<wg::KarplusString>> grouped;
			std::vector<std::unique_ptr<wg::KarplusString>> single;
			for (size_t k = 0; k < count; ++k) {
				grouped.push_back(makeString(pitches[k], 0.8f, 0.3f + 0.2f * k, 0.5f));
				single.push_back(makeString(pitches[k], 0.8f, 0.3f + 0.2f * k, 0.5f));
			}
			std::vector<wg::KarplusString*> group;
			for (auto& string : grouped) {
				group.push_back(string.get());
			}
			const auto excitation = noise(20000, 1 << 29, 11);
			std::array<q31_t, 128 * wg::KarplusString::kGroupSize> interleaved{};
			std::array<q31_t, 128> one{};
			size_t pos = 0;
			for (size_t block = 0; block < 900; ++block) {
				const size_t n = 1 + (block * 37) % 128;
				const bool excited = pos < excitation.size();
				const std::span<const q31_t> plucked =
				    excited ? std::span<const q31_t>{excitation.data() + pos, std::min(n, excitation.size() - pos)}
				            : std::span<const q31_t>{};
				const size_t length = excited ? plucked.size() : n;
				for (size_t k = 0; k < count; ++k) {
					const double glide = block > 450 ? 1.0 + 0.002 * static_cast<double>(block - 450) : 1.0;
					const wg::StringParams params{phaseIncrementFor(pitches[k] / glide), 0.8f, 0.3f + 0.2f * k, 0.5f};
					(void)grouped[k]->setParams(params);
					(void)single[k]->setParams(params);
				}
				const bool rendered = wg::KarplusString::renderPluckedGroup(
				    group, plucked, {interleaved.data(), length * wg::KarplusString::kGroupSize});
				same = same && rendered;
				for (size_t k = 0; k < count; ++k) {
					single[k]->renderPlucked(plucked, {one.data(), length});
					for (size_t i = 0; i < length; ++i) {
						same = same && one[i] == interleaved[i * wg::KarplusString::kGroupSize + k];
					}
				}
				pos += length;
			}
		}
		expect(same).to_be_true();
	});

	it("refuses to group strings whose delay lines are at different positions", _ {
		auto a = makeString(220, 0.8f, 0.5f);
		auto b = makeString(220, 0.8f, 0.5f);
		std::array<q31_t, 16> one{};
		a->renderPlucked({}, one); // a is now 16 samples ahead
		std::array<q31_t, 16 * wg::KarplusString::kGroupSize> interleaved{};
		wg::KarplusString* group[] = {a.get(), b.get()};
		expect(wg::KarplusString::renderPluckedGroup(group, {}, interleaved)).to_be_false();
	});

	it("mid-string pluck (beta 0.5) suppresses even harmonics relative to beta 0.05", _ {
		auto ratio = [](float pluck) {
			auto string = makeString(220, 1.f, 1.f, pluck);
			const auto out = render(*string, 0.2, {kImpulse});
			const size_t window = 4096;
			return levelDb(out, 2205, window, 440) - levelDb(out, 2205, window, 220);
		};
		// beta = 0.05 + 0.45 * pluck, so pluck 0 is beta 0.05 and pluck 1 is beta 0.5
		expect(ratio(1.f)).to_be_less_than(ratio(0.f) - 20.0);
	});

	it("soft velocity darkens the excitation", _ {
		auto brightness = [](float velocity) {
			auto string = makeString(220, 1.f, 1.f);
			string->setVelocity(velocity);
			const auto out = render(*string, 0.2, {kImpulse});
			return levelDb(out, 2205, 4096, 220 * 12) - levelDb(out, 2205, 4096, 220);
		};
		expect(brightness(0.f)).to_be_less_than(brightness(1.f) - 6.0);
	});
});
// clang-format on

CPPSPEC_SPEC(karplus_string)
