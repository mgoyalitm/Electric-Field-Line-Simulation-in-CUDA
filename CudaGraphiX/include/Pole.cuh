#pragma once

#include <iostream>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "Header.cuh"
#include "Vector3f.cuh"

/// @brief Provides physics-related utilities for field line generation, pole representation, field strength calculation, Fibonacci sphere point computation, and pole animation.
/// @details This namespace encapsulates various functions and structures that are essential for simulating and visualizing magnetic fields, including the representation of poles, calculation of field strengths, generation of points on a Fibonacci sphere, and animation of poles.
namespace Physics {


	constexpr float dx2 = 0.001f * Constants::dx * Constants::dx;

	/// @brief Represents a pole with physical properties such as mass, strength, position, and velocity.
	/// @details This structure encapsulates the characteristics of a pole, which can be used in simulations involving magnetic fields. It includes properties for strength, mass, position, and velocity, allowing for dynamic interactions and animations.
	struct Pole
	{
		/// @brief A variable representing pole strength as a floating-point value.
		/// @details This value is used to determine the influence of the pole on the field strength at a given point in space.
		float strength;

		/// @brief Represents a mass as a floating-point value.
		/// @details This value is used to determine the inertia of the pole, affecting how it responds to forces applied to it.
		float mass;

		/// @brief Represents a 3-dimensional vector of floating-point values, used to denote a position vector in 3D space.
		/// @details This vector indicates the location of the pole in a 3D coordinate system.
		Geometry::Vector3f position;

		/// @brief Represents a 3-dimensional vector of floating-point values, used to denote the velocity vector of the pole.
		/// @details This vector indicates the speed and direction of the pole's movement in 3D space.
		Geometry::Vector3f velocity;

		/// @brief Represents a 3-dimensional vector of floating-point values, used to denote the acceleration vector of the pole.
		/// @details This vector indicates the rate of change of the pole's velocity in 3D space.
		Geometry::Vector3f acceleration;

		/// @brief Parameterized constructor initializes the pole with specified strength, mass, position, and velocity.
		/// @details This constructor allows for the creation of a Pole object with specific physical properties.
		/// @param strength The strength of the pole.
		/// @param mass The mass of the pole.
		/// @param position The position of the pole in 3D space.
		/// @param velocity The velocity of the pole in 3D space.
		/// @return A Pole object initialized with the specified parameters.
		__host__ __device__ inline Pole(float strength, float mass, Geometry::Vector3f position, Geometry::Vector3f velocity) : strength(strength), mass(mass), position(position), velocity(velocity), acceleration(Geometry::Vector3f()) {}

		/// @brief Parameterized constructor initializes the pole with specified strength, mass, position, and velocity.
		/// @details This constructor allows for the creation of a Pole object with specific physical properties.
		/// @param strength The strength of the pole.
		/// @param mass The mass of the pole.
		/// @param position The position of the pole in 3D space.
		/// @param velocity The velocity of the pole in 3D space.
		/// @param acceleration The acceleration of the pole in 3D space.
		/// @return A Pole object initialized with the specified parameters.
		__host__ __device__ inline Pole(float strength, float mass, Geometry::Vector3f position, Geometry::Vector3f velocity, Geometry::Vector3f acceleration) : strength(strength), mass(mass), position(position), velocity(velocity), acceleration(acceleration) {}

		/// @brief Default constructor for the Pole class, initializing its members to default values.
		/// @details This constructor creates a Pole object with default values for strength, mass, position, and velocity.
		/// @return Constructs a Pole object with strength and mass set to 0.0f, and position and velocity set to zero vectors.
		__host__ __device__ inline Pole() : strength(0.0f), mass(0.0f), position(Geometry::Vector3f(0.0f, 0.0f, 0.0f)), velocity(Geometry::Vector3f(0.0f, 0.0f, 0.0f)), acceleration(Geometry::Vector3f()) {}

		__host__ __device__ inline Pole operator-() const noexcept {
			return Pole(strength, mass, -position, -velocity, -acceleration);
		}
	};

	/// @brief Calculates the naive field strength vector at a given point due to a pole in 3D space.
	/// @details This function computes the field strength vector at a specified point based on the position and strength of a pole.
	/// @param pole The source pole, containing its position and strength.
	/// @param point The 3D point at which to compute the field strength.
	/// @return A Geometry::Vector3f representing the field strength vector at the specified point. Returns a zero vector if the point coincides with the pole's position.
	__host__ __device__ inline  Geometry::Vector3f naive_field_strength(const Pole& pole, const Geometry::Vector3f& point) noexcept {

		Geometry::Vector3f direction = point - pole.position;
		float sqr_dist = Geometry::sqr_mag(direction);

		if (sqr_dist == 0.0f) {
			return Geometry::Vector3f(0.0f, 0.0f, 0.0f);
		}

		float k = sqr_dist + dx2;

		float multiplier = pole.strength / (k * sqrtf(sqr_dist));
		return Geometry::Vector3f(direction.x * multiplier, direction.y * multiplier, direction.z * multiplier);
	}

	/// @brief Calculates the total field strength at a given point due to multiple poles using a naive summation approach.
	/// @details This function computes the cumulative field strength at a specified point by summing the contributions from each pole in an array.
	/// @param poles Pointer to an array of Pole objects representing the sources of the field.
	/// @param pole_count The number of poles in the array.
	/// @param point The 3D point at which to compute the field strength.
	/// @return A Geometry::Vector3f representing the total field strength at the specified point.
	__host__ __device__ inline Geometry::Vector3f naive_field_strength(Pole* poles, int pole_count, const Geometry::Vector3f& point) noexcept {

		Geometry::Vector3f field_strength(0.0f, 0.0f, 0.0f);

		for (int i = 0; i < pole_count; ++i) {
			field_strength += naive_field_strength(poles[i], point);
		}

		return field_strength;
	}

	/// @brief Calculates the gravitational force vector exerted by a pole at a given point using a naive inverse-square law approach.
	/// @param pole The source of gravity, containing its position and mass.
	/// @param point The point in space where the gravitational force is calculated.
	/// @return A Vector3f representing the gravitational force vector at the specified point due to the pole. Returns a zero vector if the point coincides with the pole's position.
	__host__ __device__ inline Geometry::Vector3f naive_gravity(const Pole& pole, const Geometry::Vector3f& point) noexcept {
		Geometry::Vector3f direction = pole.position - point;
		float sqr_dist = Geometry::sqr_mag(direction);

		if (sqr_dist == 0.0f) {
			return Geometry::Vector3f(0.0f, 0.0f, 0.0f);
		}

		float multiplier = pole.mass * sqrtf(sqr_dist);
		return Geometry::Vector3f(direction.x * multiplier, direction.y * multiplier, direction.z * multiplier);
	}

	/// @brief Calculates the total gravitational vector at a given point due to multiple poles using a naive summation approach.
	/// @param poles Pointer to an array of Pole objects representing the sources of gravity.
	/// @param pole_count The number of poles in the array.
	/// @param point The 3D point at which to compute the gravitational vector.
	/// @return The resulting 3D gravitational vector at the specified point.
	__host__ __device__ inline Geometry::Vector3f naive_gravity(Pole* poles, int pole_count, const Geometry::Vector3f& point) noexcept {
		Geometry::Vector3f gravity(0.0f, 0.0f, 0.0f);

		for (int i = 0; i < pole_count; ++i) {
			gravity += naive_gravity(poles[i], point);
		}

		return gravity;
	}

	/// @brief Calculates a point on a Fibonacci sphere based on the given index and radius.
	/// @param point The base point from which the Fibonacci point is calculated.
	/// @param direction The direction vector used for orientation.
	/// @param index The index of the point on the Fibonacci sphere.
	/// @param radius The radius of the sphere.
	/// @param total The total number of points on the sphere.
	/// @return A Vector3f representing the computed Fibonacci point on the sphere.
	__host__ __device__ inline Geometry::Vector3f fibonacci_point(Geometry::Vector3f point, Geometry::Vector3f direction, int index, int total, float radius) {
		const float phi = 2.39996322973f;
		float y = 1.0f - (2.0f * index + 1.0f) / (float)total;
		float r = sqrtf(1.0f - y * y);
		float theta = index * phi;
		float x = cosf(theta) * r;
		float z = sinf(theta) * r;

		if (point.x == 0 && point.y == 0 && point.z == 0) {
			return Geometry::Vector3f(x * radius, y * radius, z * radius);
		}
		else if (direction.x == 0 && direction.y == 0 && direction.z == 0) {
			return Geometry::Vector3f(x * radius + point.x, y * radius + point.y, z * radius + point.z);
		}

		Geometry::Vector3f y_vec = Geometry::normalize(direction);
		Geometry::Vector3f cross = Geometry::cross(point, y_vec);
		float mag = Geometry::mag(cross);

		if (mag == 0) {
			return Geometry::Vector3f(x * radius + point.x, y * radius + point.y, z * radius + point.z);
		}

		Geometry::Vector3f x_vec = cross / mag;
		Geometry::Vector3f z_vec = Geometry::normalize(Geometry::cross(x_vec, y_vec));

		Geometry::Vector3f vector = x_vec * (x * radius) + y_vec * (y * radius) + z_vec * (z * radius) + point;
		return vector;
	}

	/// @brief Calculates a point on a Fibonacci sphere based on the given index and radius.
	/// @param point The base point from which the Fibonacci point is calculated.
	/// @param index The index of the point on the Fibonacci sphere.
	/// @param radius The radius of the sphere.
	/// @param total The total number of points on the sphere.
	/// @return A Vector3f representing the computed Fibonacci point on the sphere.
	__host__ __device__ inline Geometry::Vector3f fibonacci_point(Geometry::Vector3f point, int index, int total, float radius) {
		const float phi = 2.39996322973f;
		float y = 1.0f - (2.0f * index + 1.0f) / (float)total;
		float r = sqrtf(1.0f - y * y);
		float theta = index * phi;
		float x = cosf(theta) * r;
		float z = sinf(theta) * r;
		return Geometry::Vector3f(x * radius + point.x, y * radius + point.y, z * radius + point.z);
	}

	/// @brief Calculates a point on a Fibonacci sphere based on the given index and radius.
	/// @param point The base point from which the Fibonacci point is calculated.
	/// @param index The index of the point on the Fibonacci sphere.
	/// @param radius The radius of the sphere.
	/// @param total The total number of points on the sphere.
	/// @return A Vector3f representing the computed Fibonacci point on the sphere.
	__host__ __device__ inline Geometry::Vector3f fibonacci_point(int index, int total, float radius) {
		return fibonacci_point(Geometry::Vector3f(), index, total, radius);
	}
	/// @brief Animates the poles on the CPU.
	/// @details This function updates the positions and velocities of the poles based on their field strengths and interactions.
	/// @param poles Pointer to an array of Pole objects.
	/// @param pole_count The number of poles in the array.
	void animate_poles_cpu(Pole* poles);
}