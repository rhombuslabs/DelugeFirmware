// Listening harness for KarplusString. Writes 16-bit mono WAVs; asserts nothing (not a *_spec file).
#include "dsp/waveguide/karplus_string.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace wg = deluge::dsp::waveguide;

namespace {

constexpr double kFs = 44100.0;

uint32_t phaseIncrementFor(double hz) {
	return static_cast<uint32_t>(hz / kFs * 4294967296.0);
}

void writeWav(const std::string& path, const std::vector<q31_t>& samples) {
	std::ofstream file(path, std::ios::binary);
	auto u32 = [&](uint32_t v) { file.write(reinterpret_cast<const char*>(&v), 4); };
	auto u16 = [&](uint16_t v) { file.write(reinterpret_cast<const char*>(&v), 2); };
	const uint32_t bytes = static_cast<uint32_t>(samples.size() * 2);
	file.write("RIFF", 4);
	u32(36 + bytes);
	file.write("WAVEfmt ", 8);
	u32(16);
	u16(1);
	u16(1);
	u32(44100);
	u32(44100 * 2);
	u16(2);
	u16(16);
	file.write("data", 4);
	u32(bytes);
	for (q31_t s : samples) {
		// The string ceiling (SoftSaturator::kMax = 0.75 * 2^29) maps to 24576, about -2.5 dBFS; mirrors readWav's <<
		// 14
		const int32_t scaled = std::clamp(s >> 14, -32767, 32767);
		u16(static_cast<uint16_t>(static_cast<int16_t>(scaled)));
	}
	std::printf("wrote %s\n", path.c_str());
}

/// Reads a 16-bit PCM WAV (mono or stereo, folded to mono) into q31 at oscillator level (~2^29 peak).
std::vector<q31_t> readWav(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
	std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if (data.size() < 44) {
		std::fprintf(stderr, "error: cannot read WAV (missing or shorter than 44 bytes): %s\n", path.c_str());
		return {};
	}
	const uint16_t channels = *reinterpret_cast<const uint16_t*>(&data[22]);
	std::vector<q31_t> out;
	for (size_t pos = 44; pos + 2 * channels <= data.size(); pos += 2 * channels) {
		int32_t sum = 0;
		for (uint16_t c = 0; c < channels; ++c) {
			sum += *reinterpret_cast<const int16_t*>(&data[pos + 2 * c]);
		}
		out.push_back((sum / channels) << 14);
	}
	return out;
}

template <typename OnBlock>
std::vector<q31_t> run(wg::KarplusString& string, double seconds, const std::vector<q31_t>& excitation,
                       OnBlock onBlock) {
	const size_t total = static_cast<size_t>(seconds * kFs);
	std::vector<q31_t> out;
	std::array<q31_t, 128> in{};
	std::array<q31_t, 128> block{};
	for (size_t pos = 0; pos < total; pos += 128) {
		const size_t n = std::min<size_t>(128, total - pos);
		onBlock(pos / kFs);
		for (size_t i = 0; i < n; ++i) {
			in[i] = pos + i < excitation.size() ? excitation[pos + i] : 0;
		}
		string.render({in.data(), n}, {block.data(), n});
		out.insert(out.end(), block.begin(), block.begin() + n);
	}
	return out;
}

std::vector<q31_t> noiseBurst(double hz, uint32_t seed) {
	std::mt19937 rng{seed};
	std::uniform_int_distribution<int32_t> dist{-(1 << 29), 1 << 29};
	std::vector<q31_t> out(static_cast<size_t>(kFs / hz));
	for (auto& s : out) {
		s = dist(rng);
	}
	return out;
}

/// The frames of a 16-bit mono wavetable WAV (any chunk layout), `frameSize` samples each, as -1..1.
std::vector<std::vector<float>> readWavetable(const std::string& path, size_t frameSize) {
	std::ifstream file(path, std::ios::binary);
	std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	std::vector<std::vector<float>> frames;
	for (size_t pos = 12; pos + 8 <= data.size();) {
		const uint32_t size = *reinterpret_cast<const uint32_t*>(&data[pos + 4]);
		if (std::string(&data[pos], 4) == "data") {
			const auto* samples = reinterpret_cast<const int16_t*>(&data[pos + 8]);
			const size_t count = std::min<size_t>(size, data.size() - pos - 8) / 2;
			for (size_t start = 0; start + frameSize <= count; start += frameSize) {
				frames.emplace_back(frameSize);
				for (size_t i = 0; i < frameSize; ++i) {
					frames.back()[i] = samples[start + i] / 32768.f;
				}
			}
			break;
		}
		pos += 8 + size + (size & 1);
	}
	return frames;
}

/// One cycle of `frame` at `hz` (linear interpolation), with full scale at `level`.
std::vector<q31_t> oneCycle(const std::vector<float>& frame, double hz, double level) {
	std::vector<q31_t> out(static_cast<size_t>(kFs / hz));
	for (size_t i = 0; i < out.size(); ++i) {
		const double position = static_cast<double>(i) / out.size() * frame.size();
		const size_t a = static_cast<size_t>(position);
		const double f = position - a;
		const double v = frame[a] * (1 - f) + frame[(a + 1) % frame.size()] * f;
		out[i] = static_cast<q31_t>(v * level);
	}
	return out;
}

/// Scales `samples` so their RMS over the first 0.3 s is about -20 dBFS in the WAV, so pluck exciters of different
/// strengths can be compared by tone rather than loudness.
std::vector<q31_t> levelMatched(std::vector<q31_t> samples) {
	const size_t n = std::min(samples.size(), static_cast<size_t>(0.3 * kFs));
	double sum = 0;
	for (size_t i = 0; i < n; ++i) {
		sum += static_cast<double>(samples[i]) * samples[i];
	}
	const double rms = std::sqrt(sum / std::max<size_t>(n, 1));
	// writeWav maps 2^29 to about 32767 (>> 14); -20 dBFS is a tenth of that
	const double gain = rms > 0 ? 0.1 * 536870912.0 / rms : 1.0;
	for (auto& s : samples) {
		s = static_cast<q31_t>(std::clamp(s * gain, -2147483647.0, 2147483647.0));
	}
	return samples;
}

} // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::printf("usage: StringPreview <out_dir> [exciter.wav]\n"
		            "       StringPreview <out_dir> --pluck-table <wavetable.wav>\n");
		return 1;
	}
	const std::string dir = argv[1];

	if (argc >= 4 && std::string(argv[2]) == "--pluck-table") {
		// Plucks with one cycle of a few frames of an exciter wavetable, next to a noise-burst pluck, level-matched.
		// Params as the PLUCK presets: sustain 0.6, brightness 0.48, pluck position 0.17
		const auto frames = readWavetable(argv[3], 2048);
		if (frames.empty()) {
			std::fprintf(stderr, "error: no wavetable frames in %s\n", argv[3]);
			return 1;
		}
		auto pluck = [&](const std::vector<q31_t>& exciter, double hz, const std::string& name) {
			wg::KarplusString string;
			string.reset();
			string.setVelocity(0.8f);
			const auto out = run(string, 2.5, exciter,
			                     [&](double) { (void)string.setParams({phaseIncrementFor(hz), 0.6f, 0.48f, 0.17f}); });
			writeWav(dir + "/" + name + "_" + std::to_string(static_cast<int>(hz)) + ".wav", levelMatched(out));
		};
		for (double hz : {110.0, 220.0, 440.0}) {
			pluck(noiseBurst(hz, 1), hz, "noise");
			for (size_t index : {size_t{0}, frames.size() / 2, frames.size() - 1}) {
				const std::string name = index == 0                   ? "frame000_finger"
				                         : index == frames.size() - 1 ? "frame255_pick"
				                                                      : "frame128_middle";
				pluck(oneCycle(frames[index], hz, 536870912.0), hz, name);
			}
		}
		return 0;
	}

	for (double hz : {29.14, 55.0, 110.0, 220.0, 440.0, 880.0, 1760.0}) {
		wg::KarplusString string;
		string.reset();
		string.setVelocity(0.8f);
		const auto out = run(string, 3.0, noiseBurst(hz, 1),
		                     [&](double) { (void)string.setParams({phaseIncrementFor(hz), 0.6f, 0.5f, 0.15f}); });
		writeWav(dir + "/pluck_" + std::to_string(static_cast<int>(hz)) + ".wav", out);
	}

	{
		// Bowed: a sustained, quiet saw exciter at the string's pitch
		wg::KarplusString string;
		string.reset();
		std::vector<q31_t> saw(static_cast<size_t>(3 * kFs));
		for (size_t i = 0; i < saw.size(); ++i) {
			saw[i] = static_cast<q31_t>((std::fmod(i * 110.0 / kFs, 1.0) * 2 - 1) * (1 << 27));
		}
		const auto out =
		    run(string, 4.0, saw, [&](double) { (void)string.setParams({phaseIncrementFor(110), 0.8f, 0.6f, 0.2f}); });
		writeWav(dir + "/bowed_110.wav", out);
	}

	{
		wg::KarplusString string;
		string.reset();
		const auto out = run(string, 2.0, noiseBurst(55, 2), [&](double t) {
			const double position = t < 1.0 ? t : 2.0 - t;
			(void)string.setParams({phaseIncrementFor(55 * std::pow(8.0, position)), 1.f, 0.7f, 0.1f});
		});
		writeWav(dir + "/bend.wav", out);
	}

	if (argc >= 3) {
		const auto exciter = readWav(argv[2]);
		if (exciter.empty()) {
			return 1;
		}
		wg::KarplusString string;
		string.reset();
		const auto out = run(string, exciter.size() / kFs + 3.0, exciter,
		                     [&](double) { (void)string.setParams({phaseIncrementFor(110), 0.9f, 0.6f, 0.2f}); });
		writeWav(dir + "/resonator.wav", out);
	}
	return 0;
}
