#include "dsp/waveguide/filters.hpp"
#include "dsp/waveguide/string_mix.hpp"

#include "cppspec.hpp"

#include <random>
#include <vector>

namespace wg = deluge::dsp::waveguide;

namespace {

/// Random values, a quarter of them at or near the int32 extremes so every saturation is reached
int32_t randomExtreme(std::mt19937& random) {
	const uint32_t r = random();
	switch (r % 8) {
	case 0:
		return INT32_MAX - static_cast<int32_t>(r % 1000);
	case 1:
		return INT32_MIN + static_cast<int32_t>(r % 1000);
	default:
		return static_cast<int32_t>(random());
	}
}

/// String output: within the saturator's ceiling, often right at it
int32_t randomString(std::mt19937& random) {
	constexpr int32_t kMax = wg::SoftSaturator::kMax;
	const uint32_t r = random();
	switch (r % 4) {
	case 0:
		return (r & 8) ? kMax : -kMax;
	default:
		return static_cast<int32_t>(random() % (2u * kMax + 1)) - kMax;
	}
}

} // namespace

// clang-format off
describe string_mix("mixStrings", ${
	it("matches mixStringsReference bit for bit", _ {
		std::mt19937 random(1234);
		bool same = true;
		for (int trial = 0; trial < 4000; ++trial) {
			const size_t stride = (trial % 2 == 0) ? 1 : 4;
			const size_t count = (stride == 1) ? 1 : 1 + (trial / 2) % 4;
			const bool stereo = (trial / 8) % 2 == 1;
			const size_t numSamples = random() % 129;
			std::vector<q31_t> in(numSamples * stride + 4);
			for (auto& x : in) {
				x = randomString(random);
			}
			int32_t panLeft[4];
			int32_t panRight[4];
			for (size_t k = 0; k < 4; ++k) {
				panLeft[k] = static_cast<int32_t>(random() % ((1u << 30) + 1));
				panRight[k] = static_cast<int32_t>(random() % ((1u << 30) + 1));
			}
			// An envelope level, 0 .. INT32_MAX, ramping anywhere within that range over the block (as a voice's
			// amplitude does; the reference's running sum would be signed overflow if it left it)
			const int64_t start = random() % (int64_t{INT32_MAX} + 1);
			const int64_t end = random() % (int64_t{INT32_MAX} + 1);
			const int64_t increment = numSamples ? (end - start) / static_cast<int64_t>(numSamples) : 0;
			const wg::StringMix mix{
			    in.data(),
			    stride,
			    count,
			    numSamples,
			    static_cast<int32_t>(start),
			    static_cast<int32_t>(increment),
			    stereo ? panLeft : nullptr,
			    stereo ? panRight : nullptr,
			};
			std::vector<q31_t> expected((stereo ? 2 : 1) * numSamples);
			for (auto& x : expected) {
				x = randomExtreme(random);
			}
			std::vector<q31_t> actual = expected;
			wg::mixStringsReference(mix, expected.data());
			wg::mixStrings(mix, actual.data());
			same = same && expected == actual;
		}
		expect(same).to_be_true();
	});
});
// clang-format on

CPPSPEC_SPEC(string_mix)
