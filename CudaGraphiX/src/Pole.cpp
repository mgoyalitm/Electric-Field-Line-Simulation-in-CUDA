#include <algorithm>
#include <execution> 
#include "Pole.cuh"
#include "Randomizer.cuh"
#include "Graphix.hpp"

namespace Physics {

	constexpr float CriticalAcceleration = 40000.0f;
	constexpr float SafeAcceleration = 400.0f;
	float energies_buffer[Constants::PolesCount];
	constexpr float RetardationFactor = 0.1f / Constants::AnimationSteps;
	constexpr float Multiplier = (1 - RetardationFactor);
	constexpr float MultiplierSquare = Multiplier * Multiplier;

	void animate_poles_cpu(Pole* poles) {

		for (int i = 0; i < Constants::AnimationSteps; i++)
		{
			std::for_each(std::execution::par, poles, poles + Constants::PolesCount, [&](Physics::Pole& pole) {
				Geometry::Vector3f field_strength = naive_field_strength(poles, Constants::PolesCount, pole.position);
				Geometry::Vector3f acceleration = field_strength * (pole.strength / pole.mass);
				pole.acceleration = acceleration;
				});

			std::for_each(std::execution::par, poles, poles + Constants::PolesCount, [&](Physics::Pole& pole) {
				pole.velocity += pole.acceleration * Constants::dt;
				pole.position += pole.velocity * Constants::dt;
				});


			if (i == 0)
			{
				float average_acceleration = 0.0f;
				float dominant_acceleration = 1.0f;
				float low_acceleration = 0.0f;

				for (int i = 0; i < Constants::PolesCount; i++) {
					float acceleration = Geometry::mag(poles[i].acceleration);
					average_acceleration += acceleration;

					dominant_acceleration *= pow(acceleration, 1.0f / Constants::PolesCount);
					low_acceleration += Constants::PolesCount / (acceleration + 0.0000000001);
				}

				low_acceleration = 1.0 / low_acceleration;
				printf("Averages: %2f, %2f %2f\n", average_acceleration, dominant_acceleration, low_acceleration);
			}
		}
	}
}