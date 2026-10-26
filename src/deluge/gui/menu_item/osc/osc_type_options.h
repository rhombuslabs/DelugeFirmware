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

#include "definitions_cxx.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace deluge::gui::menu_item::osc {

struct OscTypeOptionContext {
	uint8_t sourceId;
	bool editingKit;
	SynthMode synthMode;
	bool inputsPluggedIn;
	bool stringFeatureOn;
	OscType currentType;
};

struct OscTypeOptionList {
	std::array<OscType, kNumOscTypes> types{};
	size_t count = 0;
	constexpr void push(OscType type) { types[count++] = type; }
};

/// The OSC TYPE menu's options, in display order.
constexpr OscTypeOptionList availableOscTypes(const OscTypeOptionContext& context) {
	OscTypeOptionList options;
	for (OscType type : {OscType::SINE, OscType::TRIANGLE, OscType::SQUARE, OscType::ANALOG_SQUARE, OscType::SAW,
	                     OscType::ANALOG_SAW_2, OscType::WAVETABLE}) {
		options.push(type);
	}
	if (context.synthMode == SynthMode::RINGMOD) {
		return options;
	}
	options.push(OscType::SAMPLE);
	if (!context.editingKit && context.sourceId == 0) {
		options.push(OscType::DX7);
	}
	options.push(OscType::INPUT_L); // Shown as "Input" when nothing is plugged in
	if (context.inputsPluggedIn) {
		options.push(OscType::INPUT_R);
		options.push(OscType::INPUT_STEREO);
	}
	if (context.sourceId == 1 && context.synthMode == SynthMode::SUBTRACTIVE
	    && (context.stringFeatureOn || context.currentType == OscType::STRING)) {
		options.push(OscType::STRING);
	}
	return options;
}

/// Index of `type` in `options`. Inputs not offered fall back to the single input entry; anything else to 0.
constexpr size_t indexOfOscType(const OscTypeOptionList& options, OscType type) {
	for (size_t i = 0; i < options.count; ++i) {
		if (options.types[i] == type) {
			return i;
		}
	}
	if (type == OscType::INPUT_R || type == OscType::INPUT_STEREO) {
		return indexOfOscType(options, OscType::INPUT_L);
	}
	return 0;
}

} // namespace deluge::gui::menu_item::osc
