#include "Acoustics.h"

#include "al.h"
#include "alc.h"

#include <cmath>
#include <vector>
#include <iostream>

namespace Acoustics {

	static ALCdevice* device = nullptr;
	static ALCcontext* context = nullptr;
	static std::vector<ALuint> sources;
	static std::vector<ALuint> buffers;

	constexpr int SampleRate = 44100;
	constexpr int Duration = 3;
	constexpr int Samples = SampleRate * Duration;
	void InitializeAcoustics()
	{
		device = alcOpenDevice(nullptr);
		if (!device) {
			std::cerr << "Failed to open OpenAL device\n";
			return;
		}

		context = alcCreateContext(device, nullptr);
		alcMakeContextCurrent(context);
		alDopplerFactor(Constants::ScalingFactor);
		alDistanceModel(AL_DISTANCE_MODEL);

		sources.resize(Constants::PolesCount);
		buffers.resize(Constants::PolesCount);

		alGenSources(Constants::PolesCount, sources.data());
		alGenBuffers(Constants::PolesCount, buffers.data());

		// Pre-generate a simple sine at base frequency
		float baseFreq = 440.0f;
		std::vector<short> waveform(Samples);

		for (int j = 0; j < Samples; j++) {
			float t = (float)j / SampleRate;
			waveform[j] = (short)(32760 * std::sin(Constants::TWO_PI * baseFreq * t));
		}

		for (int i = 0; i < Constants::PI; i++) {
			alBufferData(buffers[i], AL_FORMAT_MONO16, waveform.data(), (ALsizei)(Samples * sizeof(short)), SampleRate);
			alSourcei(sources[i], AL_BUFFER, buffers[i]);
			alSourcei(sources[i], AL_LOOPING, AL_TRUE);
			alSourcePlay(sources[i]);
			alSourcef(sources[i], AL_PITCH, 0.0f);
			alSourcef(sources[i], AL_GAIN, 0.0f);

		}
	}

	void ShutdownAcoustics()
	{
		alDeleteSources((ALsizei)sources.size(), sources.data());
		alDeleteBuffers((ALsizei)buffers.size(), buffers.data());

		alcMakeContextCurrent(nullptr);

		if (context) {
			alcDestroyContext(context);
		}

		if (device) {
			alcCloseDevice(device);
		}
	}

	void GenerateAcoustics(Rendering::RenderData data, double animation_time)
	{

		if (Rendering::validate(data) == false) {
			return;
		}

		if (sources.empty()) {
			InitializeAcoustics();
		}

		float theta = (float)(animation_time * Constants::CameraAngularVelocityRadians);
		float sin = sinf(theta);
		float cos = cosf(theta);
		float cameraX = Constants::CameraDistance * sin;
		float cameraZ = -Constants::CameraDistance * cos;
		float cameraVelocityX = Constants::CameraDistance * Constants::CameraAngularVelocityRadians * cos;
		float cameraVelocityZ = -Constants::CameraDistance * Constants::CameraAngularVelocityRadians * sin;
		Geometry::Vector3f cameraPosition = Geometry::Vector3f(cameraX, 0.0f, cameraZ);

		for (int i = 0; i < Constants::PolesCount; i++) {
			Physics::Pole& pole = data.poles[i];

			float mass_weight = 1.25f * (pole.mass / Constants::MaxMass - 0.2);
			float strength_weight = (9.0f * (abs(pole.strength) / Constants::MaxStrength) - 1.0f) / 8.0f;
			float amplitude_variation_time = 5.0f - mass_weight * 4.8f;
			
			float omega_frequency = Constants::TWO_PI * (0.25f + 9.25 * strength_weight);
			float omega_amp = Constants::TWO_PI * mass_weight / amplitude_variation_time;

			float frequency = 60 + powf(2, 12.0 * (1 - mass_weight));
			float frequency_variation = 1.0f + 0.05f * (1.0f + sinf(omega_frequency * animation_time));
			float pitch = frequency * frequency_variation / 440.0f;

			float base_amplitude = 1.25f * Constants::CameraDistance * (0.01f + 0.1f * mass_weight);
			float amplitude = base_amplitude * (0.1 +  0.45f *(1.0f + cosf(omega_amp * animation_time)));

			alSourcef(sources[i], AL_PITCH, pitch);
			alSourcef(sources[i], AL_GAIN, amplitude);

			Geometry::Vector3f position = pole.position;
			Geometry::Vector3f velocity = pole.velocity * Constants::VelocityScaling;
			alSource3f(sources[i], AL_POSITION, position.x, position.y, position.z);
			alSource3f(sources[i], AL_VELOCITY, velocity.x, velocity.y, velocity.z);
		}

		alListener3f(AL_POSITION, cameraX, 0.0f, cameraZ);
		alListener3f(AL_VELOCITY, cameraVelocityX * Constants::VelocityScaling, 0.0f, cameraVelocityZ * Constants::VelocityScaling);

		ALfloat orientation[6] = { cameraX, 0, cameraZ, 0, 1, 0 };
		alListenerfv(AL_ORIENTATION, orientation);
	}


}
