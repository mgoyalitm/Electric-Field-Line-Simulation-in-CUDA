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

	inline static float random(float max, float min) {
		std::mt19937 gen(std::random_device{}());
		std::uniform_real_distribution<float> dist(min, max);
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

	inline Geometry::Vector3f randomVector(float max_radius, float min_radius) {
		float radius = random(max_radius, min_radius);
		return randomVector(radius);
	}

	/// @brief Generates a random Physics::Pole object with properties constrained by the specified maximum values.
	/// @details This function creates a Physics::Pole object with random strength, mass, position, and velocity, all within the provided limits.
	/// @return A randomly initialized Physics::Pole object.
	inline Physics::Pole randomPole() {

		Geometry::Vector3f velocity = randomVector(Constants::MaxInitialSpeed, Constants::MinInitialSpeed);
		Geometry::Vector3f position = randomVector(Constants::MaxPlacementDistance, Constants::MinPlacementDistance);

		float mass = random(Constants::MaxMass, Constants::MinMass);
		float strength = random(Constants::MaxStrength, Constants::MinStrength);

		if (random() < 0) {
			strength = -strength;
		}

		return Physics::Pole(strength, mass, position, velocity);
	}

	/// @brief Generates an array of randomly initialized Pole objects.
	/// @param count The number of Pole objects to generate.
	/// @return A pointer to the first element of an array of randomly initialized Pole objects.
	inline Physics::Pole* randomPoles() {

		//count -= count % 2;
		Physics::Pole* poles = new Physics::Pole[Constants::PolesCount];
		float strength = 0;

		for (int i = 0; i < Constants::PolesCount; ++i) {
			Physics::Pole pole = randomPole();
			poles[i] = pole;
			strength += pole.strength;
		}

		strength /= Constants::PolesCount;

		for (int i = 0; i < Constants::PolesCount; i++) {
			poles[i].strength -= strength;
		}

		return poles;
	}

	inline Physics::Pole* randomBalancedPoles() {
		Physics::Pole* poles = new Physics::Pole[Constants::PolesCount];

		Geometry::Vector3f origin = Geometry::Vector3f();
		float total_strength = 0;

		for (int i = 0; i < Constants::PolesCount; i++) {
			float distance = random(Constants::MaxPlacementDistance, Constants::MinPlacementDistance);
			Geometry::Vector3f position = Physics::fibonacci_point(origin, origin, i, Constants::PolesCount, distance);
			Geometry::Vector3f velocity = Randomization::randomVector(1.0f);
			float mass = random(Constants::MaxMass, Constants::MinMass);
			float rnd = random();
			float strength = (rnd > 0 ? 1 : -1) * random(Constants::MaxStrength, Constants::MinStrength);
			total_strength += strength;
			poles[i] = Physics::Pole(strength, mass, position, velocity);
		}

		total_strength /= Constants::PolesCount;

		for (int i = 0; i < Constants::PolesCount; i++) {
			poles[i].strength -= total_strength;
		}

		for (int i = 0; i < Constants::PolesCount; i++) {
			Physics::Pole& pole = poles[i];
			Geometry::Vector3f field = Physics::naive_field_strength(poles, Constants::PolesCount, pole.position);
			float energy = abs(pole.strength * Geometry::mag(field));
			float speed = sqrtf(2.0f * energy / pole.mass);

			Geometry::Vector3f direction = Geometry::cross(pole.velocity, field);
			direction = Geometry::cross(direction, field);

			pole.velocity = speed * Geometry::normalize(direction);
		}

		return poles;
	}

	inline Physics::Pole* randomDipoles() {

		Physics::Pole* poles = new Physics::Pole[Constants::PolesCount];
		int length = Constants::PolesCount / 2;

		for (int i = 0; i < length; i++) {
			float distance = random(Constants::MaxPlacementDistance, Constants::MinPlacementDistance);

			Geometry::Vector3f center_position = randomVector(distance);
			float separation = 7.0f * distance / Constants::PolesCount;
			Geometry::Vector3f dipole_vector = randomVector(separation);

			float strength = random(Constants::MaxStrength, Constants::MinStrength);
			float mass = random(Constants::MaxMass, Constants::MinMass);
			Physics::Pole pole1 = Physics::Pole(strength, mass, center_position + dipole_vector, Geometry::Vector3f(0.0, 0.0, 0.0));
			Physics::Pole pole2 = Physics::Pole(-strength, mass, center_position - dipole_vector, Geometry::Vector3f(0.0, 0.0, 0.0));

			float speed = 0.45f * sqrtf((abs(strength * strength) / (separation * mass)));

			Geometry::Vector3f v1 = randomVector(1.0f);
			v1 = Geometry::cross(v1, dipole_vector);
			v1 = Geometry::cross(v1, dipole_vector);
			v1 = Geometry::normalize(v1);

			pole1.velocity = speed * v1;
			pole2.velocity = -speed * v1;

			poles[2 * i] = pole1;
			poles[2 * i + 1] = pole2;
		}

		if (Constants::PolesCount % 2 == 1) {
			Physics::Pole pole = randomPole();
			pole.strength = Constants::MinStrength / 10.0f;
			pole.mass = Constants::MaxMass * 10.0f;
			pole.position = (Constants::MinPlacementDistance / 10.0f) * Geometry::normalize(pole.position);
			poles[Constants::PolesCount - 1] = pole;
		}

		return poles;
	}

	/// @brief Generates a random atomic dipole configuration.
	/// @details This function creates an array of randomly generated atomic dipoles, adjusting the mass and velocity of each pole based on the provided mass ratio. The dipoles are generated using the randomDipoles function, and their properties are modified to reflect the specified mass ratio.
	/// @param mass_ratio The ratio of the mass of the dipole to the mass of the individual poles.
	/// @return A pointer to an array of randomly generated atomic dipoles.
	inline Physics::Pole* randomAtomicDipole(float mass_ratio) {
		Physics::Pole* poles = randomDipoles();

		for (int i = 0; i < Constants::PolesCount; i++) {
			Physics::Pole& pole = poles[i];
			float lambda = sqrt(abs(mass_ratio));
			float kappa = sqrt(abs(mass_ratio));


			if (pole.strength > 0) {
				pole.mass *= lambda;
				pole.velocity /= kappa;
			}
			else
			{
				pole.mass /= lambda;
				pole.velocity *= kappa;
			}
		}

		return poles;
	}

	inline Physics::Pole* randomAtomicDipoleBalanced(float mass_ratio) {

		Physics::Pole* poles = randomDipoles();

		int count = Constants::PolesCount / 2;

		for (int i = 0; i < count; i++) {
			Physics::Pole& pole1 = poles[i];
			Physics::Pole& pole2 = poles[i + 1];
			float reverse = random() > 0 ? 1 : -1;
			pole1.strength *= reverse;
			pole2.strength *= reverse;
		}

		return poles;
	}

	inline Physics::Pole* randomPolesBiased(float mass_weight, float velocity_weight) {
		Physics::Pole* poles = randomPoles();
		mass_weight = abs(mass_weight);

		for (int i = 0; i < Constants::PolesCount; i++) {
			Physics::Pole& pole = poles[i];

			if (pole.strength > 0) {
				pole.mass *= mass_weight;
				pole.velocity *= velocity_weight;
			}
		}

		return poles;
	}
}