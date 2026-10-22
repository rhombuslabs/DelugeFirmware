#include "dsp/waveguide/filters.hpp"
#include "dsp/waveguide/fractional_delay.hpp"

#include "cppspec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <random>

namespace wg = deluge::dsp::waveguide;

namespace {

constexpr q31_t kLevel = 1 << 28;

/// Reference 4-point Lagrange interpolation, in doubles, for the same tap layout as readLagrange.
double lagrange(double xm1, double x0, double x1, double x2, double f) {
	return xm1 * (-f * (f - 1) * (f - 2) / 6) + x0 * ((f + 1) * (f - 1) * (f - 2) / 2)
	       + x1 * (-(f + 1) * f * (f - 2) / 2) + x2 * ((f + 1) * f * (f - 1) / 6);
}

} // namespace

// clang-format off
describe waveguide("Waveguide primitives", $ {
	context("FractionalDelay", _ {
		it("returns the sample written `delay` writes ago at integer delays", _ {
			wg::FractionalDelay<64> line;
			line.clear();
			line.write(kLevel);
			for (int i = 0; i < 9; ++i) {
				line.write(0);
			}
			// The impulse was written 10 writes ago
			expect(line.readLagrange(10 << 16)).to_equal(kLevel);
			expect(line.readLagrange(9 << 16)).to_equal(0);
			expect(line.readLinear(10 << 16)).to_equal(kLevel);
		});

		it("matches double-precision Lagrange interpolation at fractional delays", _ {
			wg::FractionalDelay<64> line;
			line.clear();
			std::mt19937 rng{1};
			std::uniform_int_distribution<int32_t> dist{-kLevel, kLevel};
			std::array<int32_t, 64> written{};
			for (int i = 0; i < 64; ++i) {
				written[i] = dist(rng);
				line.write(written[i]);
			}
			auto at = [&](int delay) { return static_cast<double>(written[64 - delay]); };
			double worst = 0;
			for (uint32_t frac = 0; frac < 65536; frac += 4099) {
				const uint32_t delayQ16 = (20u << 16) | frac;
				const double f = frac / 65536.0;
				const double expected = lagrange(at(19), at(20), at(21), at(22), f);
				worst = std::max(worst, std::abs(line.readLagrange(delayQ16) - expected));
			}
			expect(worst).to_be_less_than(64.0);
		});

		it("interpolates linearly halfway between two samples", _ {
			wg::FractionalDelay<16> line;
			line.clear();
			line.write(1000);
			line.write(3000);
			// 3000 is at delay 1, 1000 at delay 2
			expect(std::abs(line.readLinear((1u << 16) | 0x8000) - 2000)).to_be_less_than(4);
		});

		it("reads the same after clearLazily() over stale memory as after clear()", _ {
			// Two lines over identical stale contents: one cleared fully, one lazily. Driving both the same way
			// (growing and shrinking delays, blocks of 7 writes) must give identical reads throughout.
			auto lazy = std::make_unique<wg::FractionalDelay<64>>();
			auto full = std::make_unique<wg::FractionalDelay<64>>();
			std::mt19937 rng{5};
			std::uniform_int_distribution<int32_t> dist{-kLevel, kLevel};
			for (int i = 0; i < 64 + 3; ++i) {
				lazy->data()[i] = dist(rng);
			}
			lazy->clearLazily();
			full->clear();
			bool same = true;
			uint32_t delayQ16 = 3u << 16;
			for (int block = 0; block < 30; ++block) {
				// Sweep the delay up to the maximum and back while the line is still filling
				delayQ16 = std::clamp<uint32_t>(delayQ16 + ((block < 15) ? (4u << 16) : -(3u << 16)) + 0x5A5A,
				                               wg::FractionalDelay<64>::kMinLagrangeDelay,
				                               wg::FractionalDelay<64>::kMaxLagrangeDelay);
				lazy->ensureZeroedFor((delayQ16 >> 16) + 2, 7);
				for (int k = 0; k < 7; ++k) {
					same = same && lazy->readLagrange(delayQ16) == full->readLagrange(delayQ16)
					       && lazy->readLinear(delayQ16) == full->readLinear(delayQ16);
					const q31_t x = dist(rng);
					lazy->write(x);
					full->write(x);
				}
			}
			expect(same).to_be_true();
		});
	});

	context("OnePoleLowpass", _ {
		it("settles on a constant input", _ {
			wg::OnePoleLowpass lp;
			lp.setCoefficient(wg::OnePoleLowpass::coefficientForCutoff(1000.f, 44100.f));
			q31_t y = 0;
			for (int i = 0; i < 4410; ++i) {
				y = lp.process(kLevel);
			}
			expect(std::abs(y - kLevel)).to_be_less_than(kLevel / 1000);
		});

		it("passes the input straight through with coefficient ONE_Q31", _ {
			wg::OnePoleLowpass lp;
			lp.setCoefficient(ONE_Q31);
			expect(std::abs(lp.process(kLevel) - kLevel)).to_be_less_than(4);
		});
	});

	context("DcBlocker", _ {
		it("removes a DC step within a second", _ {
			wg::DcBlocker dc;
			q31_t y = 0;
			for (int i = 0; i < 44100; ++i) {
				y = dc.process(kLevel);
			}
			expect(std::abs(y)).to_be_less_than(kLevel / 100);
		});

		it("passes 440 Hz with at least 99% amplitude", _ {
			wg::DcBlocker dc;
			double peak = 0;
			for (int i = 0; i < 44100; ++i) {
				const double x = kLevel * std::sin(2 * std::numbers::pi * 440 * i / 44100.0);
				const q31_t y = dc.process(static_cast<q31_t>(x));
				if (i > 22050) {
					peak = std::max(peak, std::abs(static_cast<double>(y)));
				}
			}
			expect(peak).to_be_greater_than(0.99 * kLevel);
		});
	});

	context("SoftSaturator", _ {
		it("is the identity below the knee", _ {
			expect(wg::SoftSaturator::process(wg::SoftSaturator::kKnee)).to_equal(wg::SoftSaturator::kKnee);
			expect(wg::SoftSaturator::process(-12345)).to_equal(-12345);
		});

		it("is monotonic, symmetric and bounded by kMax", _ {
			q31_t previous = 0;
			for (int64_t x = 0; x <= INT32_MAX; x += 1 << 20) {
				const q31_t y = wg::SoftSaturator::process(static_cast<q31_t>(x));
				expect(y >= previous).to_be_true();
				expect(y <= wg::SoftSaturator::kMax).to_be_true();
				expect(wg::SoftSaturator::process(static_cast<q31_t>(-x))).to_equal(-y);
				previous = y;
			}
			expect(wg::SoftSaturator::process(INT32_MIN)).to_equal(-wg::SoftSaturator::kMax);
		});
	});
});
// clang-format on

CPPSPEC_SPEC(waveguide)
