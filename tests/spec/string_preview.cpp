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

} // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::printf("usage: StringPreview <out_dir> [exciter.wav]\n");
		return 1;
	}
	const std::string dir = argv[1];

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
