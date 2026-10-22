#include "dsp/waveguide/string_param_mapping.hpp"

#include "cppspec.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace wg = deluge::dsp::waveguide;

// clang-format off
describe string_param_mapping("STRING param mapping", $ {
	it("maps pulse width final values 0..2^30 to 0..1", _ {
		expect(wg::pulseWidthToUnit(0)).to_equal(0.f);
		expect(wg::pulseWidthToUnit(1 << 30)).to_equal(1.f);
		expect(wg::pulseWidthToUnit(-5)).to_equal(0.f);
		expect(std::abs(wg::pulseWidthToUnit(1 << 29) - 0.5f)).to_be_less_than(1e-6f);
	});

	it("maps wave index final values -2^30..2^30 to 0..1", _ {
		expect(wg::waveIndexToUnit(-(1 << 30))).to_equal(0.f);
		expect(wg::waveIndexToUnit(0)).to_equal(0.5f);
		expect(wg::waveIndexToUnit(1 << 30)).to_equal(1.f);
	});

	it("maps feedback final values 0..kFeedbackFinalMax to 0..1, clamping cable overshoot", _ {
		expect(wg::feedbackToUnit(0)).to_equal(0.f);
		expect(wg::feedbackToUnit(wg::kFeedbackFinalMax)).to_equal(1.f);
		expect(wg::feedbackToUnit(3 * wg::kFeedbackFinalMax)).to_equal(1.f);
	});

	it("maps ENV2 output to a 0..ONE_Q31 gain", _ {
		expect(wg::envelopeToGain(std::numeric_limits<int32_t>::min())).to_equal(0);
		expect(wg::envelopeToGain(std::numeric_limits<int32_t>::max())).to_equal(ONE_Q31);
	});

	it("ramps a gain linearly across a buffer", _ {
		std::array<q31_t, 4> buffer{1 << 28, 1 << 28, 1 << 28, 1 << 28};
		wg::applyGainRamp(buffer, 0, ONE_Q31);
		expect(buffer[0]).to_be_less_than(buffer[1]);
		expect(buffer[2]).to_be_less_than(buffer[3]);
		expect(std::abs(buffer[3] - (1 << 28))).to_be_less_than(64);
	});
});
// clang-format on

CPPSPEC_SPEC(string_param_mapping)
