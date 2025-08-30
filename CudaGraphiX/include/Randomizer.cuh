#pragma once

#include <random>
#include <cmath>
#include <numbers>

#include "Header.cuh"
#include "Vector3f.cuh"
#include "Pole.cuh"

/// @brief Provides functions for generating random floating-point numbers, 3D vectors, and Physics::Pole objects within specified constraints.
/// @details This namespace encapsulates various randomization utilities, including functions to generate random float values, random 3D vectors with a specified radius, and random Physics::Pole objects with properties constrained by maximum values.
namespace Randomization {

	/// @brief Generates a random floating-point number between -1.0 and 1.0.
	/// @details This function uses a random number generator to produce a float value uniformly distributed in the range [-1.0, 1.0].
	/// @return A random float value in the range [-1.0, 1.0].
	inline static float random() {

		static thread_local std::mt19937 gen(std::random_device{}());
		static thread_local std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		return dist(gen);
	}


	/// @brief Generates a random 3D vector with a specified maximum length (radius).
	/// @param radius The maximum length (magnitude) of the generated vector.
	/// @return A random 3D vector of type Geometry::Vector3f whose length does not exceed the specified radius.
	inline Geometry::Vector3f randomVector(float radius) {

		float theta = random() * Constants::PI;
		float phi = random() * Constants::TWO_PI;

		float x = radius * sinf(phi) * cosf(theta);
		float y = radius * sinf(phi) * sinf(theta);
		float z = radius * cosf(phi);

		return Geometry::Vector3f(x, y, z);
	}

	/// @brief Generates a random Physics::Pole object with properties constrained by the specified maximum values.
	/// @param max_speed The maximum speed value for the generated pole. Default is 0.4f.
	/// @param max_mass The maximum mass value for the generated pole. Default is 8.
	/// @param radius The radius of the generated pole. Default is 16.
	/// @param max_strength The maximum strength value for the generated pole. Default is 16.
	/// @return A Physics::Pole object with randomly assigned properties within the specified limits.
	inline Physics::Pole randomPole(float max_speed = 0.4f, float max_mass = 8.0f, float radius = 16.0f, float max_strength = 16.0f) {

		static Geometry::Vector3f axis = Geometry::Vector3f(0.0f, 0.0f, 1.0f);
		float rnd = random();
		float sign = rnd < 0 ? -1 : 1;

		float strength = (sign / 2.25f) * (random() + 1.25f) * max_strength;

		Geometry::Vector3f position = randomVector(radius);

		float speed = max_speed * (random() + 1.25f) / 2.25f;

		Geometry::Vector3f velocity = Geometry::normalize(Geometry::cross(Geometry::cross(position, axis), axis)) * speed;

		float mass = fmax((random() + 1.5f) * max_mass / 2.5f, 0.25f);
		return Physics::Pole(strength, mass, position, velocity);
	}

	/// @brief Generates an array of randomly initialized Pole objects.
	/// @param count The number of Pole objects to generate.
	/// @param max_speed The maximum speed assigned to each Pole.
	/// @param max_mass The maximum mass assigned to each Pole.
	/// @param radius The radius assigned to each Pole.
	/// @param max_strength The maximum strength assigned to each Pole.
	/// @return A pointer to the first element of an array of randomly initialized Pole objects.
	inline Physics::Pole* randomPoles(int count, float max_speed, float max_mass, float radius, float max_strength) {
		//count -= count % 2;
		Physics::Pole* poles = new Physics::Pole[count];
		float strength = 0;
		
		for (int i = 0; i < count; ++i) {
			float dist = (random() + 1.5f) * 0.4f * radius;
			Physics::Pole pole = randomPole(0.0f, max_mass, dist, max_strength);
			poles[i] = pole;
			strength += pole.strength;
		}

		strength /= Constants::PolesCount;

		for (int i = 0; i < count; i++)
		{
			poles[i].strength -= strength;
			Geometry::Vector3f field_strength = Physics::naive_field_strength(poles, count, poles[i].position);
			float speed = ((random() + 2.0f) * max_speed) / 3.0f;
			poles[i].velocity = speed * Geometry::normalize(field_strength * poles[i].strength);
		}
		
		return poles;
	}


	inline Physics::Pole* randomDipoles(int count, float max_mass, float radius, float max_strength) {
		Physics::Pole* poles = new Physics::Pole[count];
		float max_dipole_separation = radius / 64.0f;
		int length = count / 2;

		for (int i = 0; i < length; i++) {
			Geometry::Vector3f center_position = randomVector((random() + 1.5f) * 0.4f * radius);
			float separation = (random() + 1.5f) * 0.4f * max_dipole_separation;
			Geometry::Vector3f dipole_vector = randomVector(separation);

			float strength = (random() + 1.5f) * 0.4f * max_strength;
			float mass = (random() + 1.5f) * 0.4f * max_mass;

			Physics::Pole pole1 = Physics::Pole(strength, mass, center_position + dipole_vector, Geometry::Vector3f(0.0, 0.0, 0.0));
			Physics::Pole pole2 = Physics::Pole(-strength, mass, center_position - dipole_vector, Geometry::Vector3f(0.0, 0.0, 0.0));

			float speed = sqrtf((abs(strength * strength) / (separation * mass)));

			Geometry::Vector3f v1 = randomVector(1.0f);
			v1 = Geometry::cross(v1, dipole_vector);
			v1 = Geometry::cross(v1, dipole_vector);
			v1 = Geometry::normalize(v1);

			pole1.velocity = speed * v1;
			pole2.velocity = -speed * v1;

			poles[2 * i] = pole1;
			poles[2 * i + 1] = pole2;
		}

		if (count % 2 == 1)	{
			float distance = (random() + 1.5f) * 0.4f * radius;
			poles[count - 1] = randomPole(0.0f, max_mass, distance, max_strength);
		}

		return poles;
	}
}