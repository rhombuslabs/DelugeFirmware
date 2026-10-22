/*
 * Copyright (c) 2014-2023 Synthstrom Audible Limited
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
#include "gui/menu_item/formatted_title.h"
#include "gui/menu_item/osc/osc_type_options.h"
#include "gui/menu_item/selection.h"
#include "gui/menu_item/submenu.h"
#include "gui/ui/sound_editor.h"
#include "model/settings/runtime_feature_settings.h"
#include "model/song/song.h"
#include "processing/engines/audio_engine.h"
#include "processing/sound/sound.h"
#include "processing/source.h"
#include "util/comparison.h"

#include <hid/display/oled.h>

#include <algorithm>

extern gui::menu_item::Submenu dxMenu;

namespace deluge::gui::menu_item::osc {
class Type final : public Selection, public FormattedTitle {
public:
	Type(l10n::String name, l10n::String title_format_str, uint8_t source_id)
	    : Selection(name), FormattedTitle(title_format_str, source_id + 1), sourceId_{source_id} {};
	void beginSession(MenuItem* navigatedBackwardFrom) override { Selection::beginSession(navigatedBackwardFrom); }

	[[nodiscard]] OscTypeOptionList options() const {
		Sound* sound = soundEditor.currentSound;
		return availableOscTypes({
		    .sourceId = sourceId_,
		    .editingKit = soundEditor.editingKit(),
		    .synthMode = sound->getSynthMode(),
		    .inputsPluggedIn = AudioEngine::micPluggedIn || AudioEngine::lineInPluggedIn,
		    .stringFeatureOn = runtimeFeatureSettings.isOn(RuntimeFeatureSettingType::PhysicalModellingOscillators),
		    .currentType = sound->sources[sourceId_].oscType,
		});
	}

	void readCurrentValue() override {
		setValue(static_cast<int32_t>(indexOfOscType(options(), soundEditor.currentSound->sources[sourceId_].oscType)));
	}

	void writeCurrentValue() override {
		OscType oldValue = soundEditor.currentSound->sources[sourceId_].oscType;
		const OscTypeOptionList available = options();
		const OscType newValue = available.types[std::min<size_t>(getValue(), available.count - 1)];

		const auto needs_unassignment = {
		    OscType::INPUT_L,
		    OscType::INPUT_R,
		    OscType::INPUT_STEREO,
		    OscType::SAMPLE,
		    OscType::DX7,
		    OscType::STRING,

		    // Haven't actually really determined if this needs to be here - maybe not?
		    OscType::WAVETABLE,
		};

		if (util::one_of(oldValue, needs_unassignment) || util::one_of(newValue, needs_unassignment)) {
			soundEditor.currentSound->killAllVoices();
		}

		soundEditor.currentSound->sources[sourceId_].setOscType(newValue);

		if (oldValue == OscType::SQUARE || newValue == OscType::SQUARE || oldValue == OscType::STRING
		    || newValue == OscType::STRING) {
			soundEditor.currentSound->setupPatchingForAllParamManagers(currentSong);
		}
	}

	[[nodiscard]] std::string_view getTitle() const override { return FormattedTitle::title(); }

	deluge::vector<std::string_view> getOptions(OptType optType) override {
		(void)optType;
		using enum l10n::String;
		const bool inputsPluggedIn = AudioEngine::micPluggedIn || AudioEngine::lineInPluggedIn;
		const OscTypeOptionList available = options();
		deluge::vector<std::string_view> names;
		for (size_t i = 0; i < available.count; ++i) {
			names.emplace_back(l10n::getView([&] {
				switch (available.types[i]) {
				case OscType::SINE:
					return STRING_FOR_SINE;
				case OscType::TRIANGLE:
					return STRING_FOR_TRIANGLE;
				case OscType::SQUARE:
					return STRING_FOR_SQUARE;
				case OscType::ANALOG_SQUARE:
					return STRING_FOR_ANALOG_SQUARE;
				case OscType::SAW:
					return STRING_FOR_SAW;
				case OscType::ANALOG_SAW_2:
					return STRING_FOR_ANALOG_SAW;
				case OscType::WAVETABLE:
					return STRING_FOR_WAVETABLE;
				case OscType::SAMPLE:
					return STRING_FOR_SAMPLE;
				case OscType::DX7:
					return STRING_FOR_DX7;
				case OscType::INPUT_L:
					return inputsPluggedIn ? STRING_FOR_INPUT_LEFT : STRING_FOR_INPUT;
				case OscType::INPUT_R:
					return STRING_FOR_INPUT_RIGHT;
				case OscType::INPUT_STEREO:
					return STRING_FOR_INPUT_STEREO;
				case OscType::STRING:
					return STRING_FOR_STRING_OSC;
				}
				return STRING_FOR_SINE;
			}()));
		}
		return names;
	}

	bool isRelevant(ModControllableAudio* modControllable, int32_t whichThing) const override {
		Sound* sound = static_cast<Sound*>(modControllable);
		return (sound->getSynthMode() != SynthMode::FM);
	}

	MenuItem* selectButtonPress() override {
		if (soundEditor.currentSound->sources[sourceId_].oscType != OscType::DX7) {
			return nullptr;
		}
		return &dxMenu;
	}

	[[nodiscard]] bool showColumnLabel() const override { return false; }

	void renderInHorizontalMenu(const SlotPosition& slot) override {
		oled_canvas::Canvas& image = OLED::main;

		const OscType osc_type = soundEditor.currentSound->sources[sourceId_].oscType;
		if (osc_type == OscType::DX7 || osc_type == OscType::STRING) {
			// STRING's option index moves with input plug state, so don't index by getValue() for it
			const auto option = osc_type == OscType::STRING ? l10n::getView(l10n::String::STRING_FOR_STRING_OSC).data()
			                                                : getOptions(OptType::FULL)[getValue()].data();
			return image.drawStringCentered(option, slot.start_x, slot.start_y + kHorizontalMenuSlotYOffset + 5,
			                                kTextTitleSpacingX, kTextTitleSizeY, slot.width);
		}

		const Icon& icon = [&] {
			switch (osc_type) {
			case OscType::SINE:
				return OLED::sineIcon;
			case OscType::TRIANGLE:
				return OLED::triangleIcon;
			case OscType::SQUARE:
			case OscType::ANALOG_SQUARE:
				return OLED::squareIcon;
			case OscType::SAW:
			case OscType::ANALOG_SAW_2:
				return OLED::sawIcon;
			case OscType::SAMPLE:
				return OLED::sampleIcon;
			case OscType::INPUT_STEREO:
			case OscType::INPUT_L:
			case OscType::INPUT_R:
				return AudioEngine::lineInPluggedIn ? OLED::inputIcon : OLED::micIcon;
			case OscType::WAVETABLE:
				return OLED::wavetableIcon;
			default:
				return OLED::sineIcon;
			}
		}();

		image.drawIconCentered(icon, slot.start_x, slot.width, slot.start_y + kHorizontalMenuSlotYOffset + 2);

		if (osc_type == OscType::ANALOG_SQUARE || osc_type == OscType::ANALOG_SAW_2) {
			const int32_t x = slot.start_x + 4;
			constexpr int32_t y = OLED_MAIN_HEIGHT_PIXELS - kTextSpacingY - 8;
			image.clearAreaExact(x - 1, y - 1, x + kTextSpacingX + 1, y + kTextSpacingY + 1);
			image.drawChar('A', x, y, kTextSpacingX, kTextSpacingY);
		}
	}

	bool wrapAround() override {
		return parent != nullptr && parent->renderingStyle() == Submenu::RenderingStyle::HORIZONTAL;
	}

private:
	uint8_t sourceId_;
};

} // namespace deluge::gui::menu_item::osc
