#include <algorithm>
#include <execution> 
#include "Pole.cuh"

namespace Physics {

	void animate_poles_cpu(Pole* poles) {

		for (int i = 0; i < Constants::AnimationSteps; i++)
		{
			Geometry::Vector3f accelerations[Constants::PolesCount];

			std::for_each(std::execution::par, poles, poles + Constants::PolesCount, [&](Physics::Pole& pole) {
				int index = &pole - poles;
				float magnitude = Geometry::mag(pole.position);
				Geometry::Vector3f direction = pole.position / magnitude;
				magnitude /= 10.0f;
				magnitude = sqrtf(magnitude);
				//magnitude *= magnitude;
				Geometry::Vector3f binding_force = 0.001f * (-10.0f * magnitude + (0.0025f / magnitude)) * direction;
				Geometry::Vector3f gravity_force = 0.00f * naive_gravity(poles, Constants::PolesCount, pole.position);
				Geometry::Vector3f electric_force = naive_field_strength(poles, Constants::PolesCount, pole.position) * pole.strength;
				Geometry::Vector3f acceleration = (binding_force + gravity_force + electric_force) / pole.mass;
				accelerations[index] = acceleration;
				});

			std::for_each(std::execution::par, poles, poles + Constants::PolesCount, [&](Physics::Pole& pole) {
				int index = &pole - poles;
				Geometry::Vector3f acceleration = accelerations[index];
				Geometry::Vector3f velocity = pole.velocity;
				pole.velocity += acceleration * Constants::dt;
				pole.position += pole.velocity * Constants::dt;

				if (i == 0) {
					pole.rotation_constant = 0;
				}

				pole.rotation_constant += abs(Geometry::mag(Geometry::cross(Geometry::normalize(velocity), Geometry::normalize(pole.velocity))));

				});
		}
	}
}