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
#include "gui/ui/sound_editor.h"
#include "processing/sound/sound.h"

namespace deluge::gui::menu_item::osc {

/// Whether `sourceId` of `sound` is a STRING oscillator (OSC2, Subtractive).
inline bool isStringOsc(const Sound& sound, uint8_t sourceId) {
	return sourceId == 1 && sound.getSynthMode() == SynthMode::SUBTRACTIVE
	       && sound.sources[1].oscType == OscType::STRING;
}

/// Same, for the sound currently open in the sound editor.
inline bool isEditingStringOsc(uint8_t sourceId) {
	return soundEditor.currentSound != nullptr && isStringOsc(*soundEditor.currentSound, sourceId);
}

} // namespace deluge::gui::menu_item::osc
