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

	void InitializeAcoustics()
	{
		device = alcOpenDevice(nullptr);
		if (!device) {
			std::cerr << "Failed to open OpenAL device\n";
			return;
		}

		context = alcCreateContext(device, nullptr);
		alcMakeContextCurrent(context);
		alDopplerFactor(Constants::DopplerScaling);
		alDistanceModel(AL_DISTANCE_MODEL);

		sources.resize(Constants::PolesCount);
		buffers.resize(Constants::PolesCount);

		alGenSources(Constants::PolesCount, sources.data());
		alGenBuffers(Constants::PolesCount, buffers.data());

		std::vector<short> waveform(Constants::Samples);

		for (int j = 0; j < Constants::Samples; j++) {
			float t = (float)j / Constants::SampleRate;
			waveform[j] = (short)(Constants::PeakAmplitude * sinf(Constants::TWO_PI * Constants::BaseFrequency * t));
		}

		for (int i = 0; i < Constants::PI; i++) {
			alBufferData(buffers[i], AL_FORMAT_MONO16, waveform.data(), (ALsizei)(Constants::Samples * sizeof(short)), Constants::SampleRate);
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

		float radius = Constants::CameraDistance;
		float azimuth = (float)animation_time * Constants::CameraAngularVelocityRadians;
		float elevation = Constants::CameraElevationRadians;
		float cameraX = radius * cosf(elevation) * sinf(azimuth);
		float cameraY = radius * sinf(elevation);
		float cameraZ = radius * cosf(elevation) * cosf(azimuth);
		float cameraVelocityX = cameraX * Constants::CameraAngularVelocityRadians;
		float cameraVelocityZ = -cameraZ * Constants::CameraAngularVelocityRadians;
		float cameraDistance = sqrtf(cameraX * cameraX + cameraY * cameraY + cameraZ * cameraZ);

		float min_acceleration = std::numeric_limits<float>::infinity();
		float max_acceleration = 0.0f;
		for (int i = 0; i < Constants::PolesCount; i++) {
			float acceleration = Geometry::mag(data.poles[i].acceleration);
			min_acceleration = std::min(min_acceleration, acceleration);
			max_acceleration = std::max(max_acceleration, acceleration);
		}

		float c = 0.0001f;
		float a = log2(min_acceleration + c);
		float b = log2(max_acceleration + c);
		float den = b - a;

		for (int i = 0; i < Constants::PolesCount; i++) {
			Physics::Pole& pole = data.poles[i];

			float mass_weight = (pole.mass - Constants::MinMass) / (Constants::MaxMass - Constants::MinMass);
			float strength_weight = (abs(pole.strength) - Constants::MinStrength) / (Constants::MaxStrength - Constants::MinStrength);
			float acceleration_weight = (log2(Geometry::mag(pole.acceleration) + c) - a) / den;

			float amplitude_variation_time = Constants::MinAmplitudeVariationTime * mass_weight
				+ Constants::MaxAmplitudeVariationTime * (1 - mass_weight);
			float frequency_variation_time = Constants::MaxFrequencyVariationTime * acceleration_weight +
				Constants::MinFrequencyVariationTime * (1 - acceleration_weight);

			float amplitude_variation = 1.0f + sinf(Constants::TWO_PI * animation_time / amplitude_variation_time);
			float frequency_variation = 1.0f + sinf(Constants::TWO_PI * animation_time / frequency_variation_time);

			float amplitude = Constants::Amplitude * amplitude_variation;

			float frequency = frequency_variation *
				Constants::MinFrequency * pow(Constants::MaxFrequency / Constants::MinFrequency, strength_weight);

			float pitch = frequency / 440.0f;

			alSourcef(sources[i], AL_PITCH, pitch);
			alSourcef(sources[i], AL_GAIN, amplitude);

			Geometry::Vector3f position = pole.position;
			Geometry::Vector3f velocity = pole.velocity * Constants::TimeScaling;
			alSource3f(sources[i], AL_POSITION, position.x, position.y, position.z);
			alSource3f(sources[i], AL_VELOCITY, velocity.x, velocity.y, velocity.z);
		}

		alListener3f(AL_POSITION, cameraX, 0.0f, cameraZ);
		alListener3f(AL_VELOCITY, cameraVelocityX, 0.0f, cameraVelocityZ);
		ALfloat orientation[6] = { -cameraX / cameraDistance, -cameraY / cameraDistance, -cameraZ / cameraDistance, 0, 1, 0 };
		alListenerfv(AL_ORIENTATION, orientation);
	}


}
