#include "CppUTest/TestHarness.h"
#include "gui/menu_item/osc/osc_type_options.h"

#include <vector>

using namespace deluge::gui::menu_item::osc;

namespace {
std::vector<OscType> list(const OscTypeOptionContext& context) {
	const OscTypeOptionList options = availableOscTypes(context);
	return {options.types.begin(), options.types.begin() + options.count};
}

const std::vector<OscType> kWaves = {OscType::SINE, OscType::TRIANGLE,     OscType::SQUARE,   OscType::ANALOG_SQUARE,
                                     OscType::SAW,  OscType::ANALOG_SAW_2, OscType::WAVETABLE};

std::vector<OscType> plus(std::vector<OscType> base, std::initializer_list<OscType> extra) {
	base.insert(base.end(), extra);
	return base;
}
} // namespace

TEST_GROUP(OscTypeOptions){};

TEST(OscTypeOptions, osc1MatchesTheOldListWithDx7AndOneInput) {
	const auto expected = plus(kWaves, {OscType::SAMPLE, OscType::DX7, OscType::INPUT_L});
	CHECK(list({0, false, SynthMode::SUBTRACTIVE, false, true, OscType::SAW}) == expected);
}

TEST(OscTypeOptions, osc1InAKitHasNoDx7) {
	const auto expected = plus(kWaves, {OscType::SAMPLE, OscType::INPUT_L});
	CHECK(list({0, true, SynthMode::SUBTRACTIVE, false, true, OscType::SAW}) == expected);
}

TEST(OscTypeOptions, inputsPluggedInOfferAllThreeInputs) {
	const auto expected =
	    plus(kWaves, {OscType::SAMPLE, OscType::DX7, OscType::INPUT_L, OscType::INPUT_R, OscType::INPUT_STEREO});
	CHECK(list({0, false, SynthMode::SUBTRACTIVE, true, false, OscType::SAW}) == expected);
}

TEST(OscTypeOptions, ringmodOffersOnlyWaves) {
	CHECK(list({1, false, SynthMode::RINGMOD, true, true, OscType::SAW}) == kWaves);
}

TEST(OscTypeOptions, osc2OffersStringWhenTheFeatureIsOn) {
	const auto expected = plus(kWaves, {OscType::SAMPLE, OscType::INPUT_L, OscType::STRING});
	CHECK(list({1, false, SynthMode::SUBTRACTIVE, false, true, OscType::SAW}) == expected);
}

TEST(OscTypeOptions, osc2HidesStringWhenTheFeatureIsOff) {
	const auto expected = plus(kWaves, {OscType::SAMPLE, OscType::INPUT_L});
	CHECK(list({1, false, SynthMode::SUBTRACTIVE, false, false, OscType::SAW}) == expected);
}

TEST(OscTypeOptions, osc2KeepsStringListedIfAlreadySelectedWithFeatureOff) {
	const auto expected = plus(kWaves, {OscType::SAMPLE, OscType::INPUT_L, OscType::STRING});
	CHECK(list({1, false, SynthMode::SUBTRACTIVE, false, false, OscType::STRING}) == expected);
}

TEST(OscTypeOptions, osc1NeverOffersString) {
	for (OscType t : list({0, false, SynthMode::SUBTRACTIVE, true, true, OscType::SAW})) {
		CHECK(t != OscType::STRING);
	}
}

TEST(OscTypeOptions, unpluggedStereoInputMapsToTheInputEntry) {
	const OscTypeOptionList options =
	    availableOscTypes({0, false, SynthMode::SUBTRACTIVE, false, true, OscType::INPUT_STEREO});
	CHECK_EQUAL(9u, indexOfOscType(options, OscType::INPUT_STEREO));
	CHECK_EQUAL(4u, indexOfOscType(options, OscType::SAW));
}
