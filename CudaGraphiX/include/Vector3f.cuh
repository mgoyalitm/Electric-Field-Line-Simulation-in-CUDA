#pragma once

#include <iostream>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

/// @brief Provides a 3D vector structure with floating-point components and basic arithmetic operations, along with utility functions for vector math.
/// @details This namespace encapsulates a Vector3f structure that represents a 3D vector with x, y, and z components, along with various operator overloads for vector arithmetic. Additionally, it includes utility functions for common vector operations such as dot product, cross product, magnitude calculation, normalization, and distance computation.
namespace Geometry {

	/// @brief Represents a 3D vector with floating-point x, y, and z components, and provides basic arithmetic operations.
	struct Vector3f
	{
		/// @brief Declares three floating-point variables.
		/// @details This structure represents a 3D vector with x, y, and z components.
		/// @param x The x component of the vector.
		/// @param y The y component of the vector.
		/// @param z The z component of the vector.
		float x, y, z;

		/// @brief Default constructor for the Vector3f class, initializing all components to zero.
		/// @details This constructor initializes the x, y, and z components of the vector to 0.0f.
		/// @return A Vector3f object with x, y, and z components set to 0.0f.
		__host__ __device__ inline Vector3f() : x(0.0f), y(0.0f), z(0.0f) {}

		/// @brief Default constructor initializes the vector to (0, 0, 0).
		/// @details This constructor sets all components of the vector to zero.
		/// @param x The x component of the vector (default is 0).
		/// @param y The y component of the vector (default is 0).
		/// @param z The z component of the vector (default is 0).
		/// @return A Vector3f object initialized to (0, 0, 0).
		__host__ __device__ inline Vector3f(float x, float y, float z) : x(x), y(y), z(z) {}

		/// @brief Adds two Vector3f objects component-wise.
		/// @details This operator performs component-wise addition of the current vector and the given vector.
		/// @param vector The Vector3f object to add to the current vector.
		/// @return A new Vector3f object representing the sum of the current vector and the given vector.
		__host__ __device__ inline Vector3f operator+(const Vector3f& vector) const noexcept {
			return Vector3f(x + vector.x, y + vector.y, z + vector.z);
		}

		/// @brief Adds the components of another Vector3f to this vector and assigns the result to this vector.
		/// @details This operator performs component-wise addition of the given vector to this vector.
		/// @param vector The Vector3f whose components will be added to this vector.
		/// @return A reference to this Vector3f after addition.
		__host__ __device__	inline Vector3f& operator+=(const Vector3f& vector) noexcept {
			x += vector.x;
			y += vector.y;
			z += vector.z;
			return *this;
		}

		/// @brief Subtracts the components of another Vector3f from this vector.
		/// @details This operator performs component-wise subtraction of the given vector from this vector.
		/// @param vector The Vector3f object to subtract from the current vector.
		/// @return A new Vector3f object representing the difference between this vector and the given vector.
		__host__ __device__ inline Vector3f operator-(const Vector3f& vector) const  noexcept {
			return Vector3f(x - vector.x, y - vector.y, z - vector.z);
		}

		/// @brief Subtracts the components of another Vector3f from this vector and assigns the result to this vector.
		/// @details This operator performs component-wise subtraction of the given vector from this vector.
		/// @param vector The Vector3f whose components will be subtracted from this vector.
		/// @return A reference to this Vector3f after subtraction.
		__host__ __device__	inline Vector3f& operator-=(const Vector3f& vector) noexcept {
			x -= vector.x;
			y -= vector.y;
			z -= vector.z;
			return *this;
		}

		__host__ __device__ inline Vector3f operator-() const noexcept {
			return Vector3f(-x, -y, -z);
		}

		/// @brief Multiplies the components of this vector by a scalar value.
		/// @details This operator scales the vector by multiplying each component by the given scalar.
		/// @param scalar The scalar value to multiply with.
		/// @return A new Vector3f object representing the scaled vector.
		__host__ __device__	inline Vector3f operator*(float scalar) const noexcept {
			return Vector3f(x * scalar, y * scalar, z * scalar);
		}

		/// @brief Multiplies the components of this vector by a scalar value and assigns the result to this vector.
		/// @details This operator scales the vector by multiplying each component by the given scalar and updates this vector.
		/// @param scalar The scalar value to multiply with.
		/// @return A reference to this Vector3f after scaling.
		__host__ __device__	inline Vector3f& operator*=(float scalar) noexcept {
			x *= scalar;
			y *= scalar;
			z *= scalar;
			return *this;
		}

		/// @brief Divides the components of this vector by a scalar value.
		/// @details This operator scales the vector by dividing each component by the given scalar.
		/// @param scalar The scalar value to divide by.
		/// @return A new Vector3f object representing the scaled vector.
		__host__ __device__	inline Vector3f operator/(float scalar) const noexcept {
			return Vector3f(x / scalar, y / scalar, z / scalar);
		}

		/// @brief Divides the components of this vector by a scalar value and assigns the result to this vector.
		/// @details This operator scales the vector by dividing each component by the given scalar and updates this vector.
		/// @param scalar The scalar value to divide by.
		/// @return A reference to this Vector3f after scaling.
		__host__ __device__	inline Vector3f& operator/=(float scalar) noexcept {
			x /= scalar;
			y /= scalar;
			z /= scalar;
			return *this;
		}
	};

	/// @brief Multiplies a 3D vector by a scalar value.
	/// @details This function scales each component of the vector by the given scalar value.
	/// @param scaler The scalar value to multiply each component of the vector by.
	/// @param vector The 3D vector to be scaled.
	/// @return A new Vector3f representing the scaled vector.
	__host__ __device__ inline Vector3f operator*(float scaler, Vector3f vector) noexcept {
		return Vector3f(vector.x * scaler, vector.y * scaler, vector.z * scaler);
	}

	/// @brief Computes the dot product of two 3D vectors.
	/// @details This function calculates the dot product of two vectors, which is a scalar value representing the cosine of the angle between them multiplied by their magnitudes.
	/// @param a The first 3D vector operand.
	/// @param b The second 3D vector operand.
	/// @return The dot product of the two vectors as a float.
	__host__ __device__	inline float dot(const Vector3f& a, const Vector3f& b) noexcept {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	/// @brief Computes the cross product of two 3D vectors.
	/// @details This function calculates the cross product of two vectors, which results in a vector that is perpendicular to both input vectors.
	/// @param a The first 3D vector operand.
	/// @param b The second 3D vector operand.
	/// @return A Vector3f representing the cross product of vectors a and b.
	__host__ __device__ inline Vector3f cross(const Vector3f& a, const Vector3f& b) noexcept {
		return Vector3f(
			a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x
		);
	}

	/// @brief Calculates the magnitude (length) of a 3D vector.
	/// @details This function computes the magnitude of a vector using the formula sqrt(x^2 + y^2 + z^2), which is the Euclidean norm of the vector.
	/// @param vector The 3D vector whose magnitude is to be computed.
	/// @return The magnitude (Euclidean norm) of the input vector as a float.
	__host__ __device__	inline float mag(const Vector3f& vector) noexcept {
		return sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
	}

	/// @brief Computes the squared magnitude of a 3D vector using a custom formula.
	/// @details This function calculates the squared magnitude of a vector using the formula (x * x + y * y + z * z), which is equivalent to the dot product of the vector with itself.
	/// @param vector A constant reference to the Vector3f object whose squared magnitude is to be calculated.
	/// @return The squared magnitude of the vector, calculated as (vector.x * vector.x * vector.y + vector.z * vector.z).
	__host__ __device__	inline float sqr_mag(const Vector3f& vector) noexcept {
		return vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;
	}

	/// @brief Returns the normalized (unit length) version of a 3D vector.
	/// @details This function computes the normalized vector by dividing each component of the input vector by its magnitude.
	/// @param vector The 3D vector to normalize.
	/// @return A Vector3f representing the normalized version of the input vector. If the input vector has zero magnitude, a zero vector is returned.
	__host__ __device__ inline Vector3f normalize(const Vector3f& vector) noexcept {
		float magnitude = mag(vector);
		if (magnitude == 0.0f) return Vector3f(0.0f, 0.0f, 0.0f);
		return vector / magnitude;
	}

	/// @brief Calculates the Euclidean distance between two 3D vectors.
	/// @details This function computes the distance between two points in 3D space represented by Vector3f objects.
	/// @param a The first 3D vector.
	/// @param b The second 3D vector.
	/// @return The Euclidean distance between vectors a and b as a float.
	__host__ __device__ inline float dist(const Vector3f& a, const Vector3f& b) noexcept {

		float dx = a.x - b.x;
		float dy = a.y - b.y;
		float dz = a.z - b.z;

		return sqrtf(dx * dx + dy * dy + dz * dz);
	}

	/// @brief Calculates the Euclidean distance between two 3D vectors.
	/// @details This function computes the distance between two points in 3D space represented by Vector3f objects.
	/// @param a The first 3D vector.
	/// @param b The second 3D vector.
	/// @return The Euclidean distance between vectors a and b as a float.
	__host__ __device__ inline float sqr_dist(const Vector3f& a, const Vector3f& b) noexcept {

		float dx = a.x - b.x;
		float dy = a.y - b.y;
		float dz = a.z - b.z;

		return dx * dx + dy * dy + dz * dz;
	}

	/// @brief Computes the circumcenter of a triangle defined by three points in 3D space.
	/// @param point1 The first vertex of the triangle.
	/// @param point2 The second vertex of the triangle.
	/// @param point3 The third vertex of the triangle.
	/// @return A Vector3f representing the circumcenter of the triangle. If the points are collinear, returns a default Vector3f.
	__host__ __device__ inline Vector3f circumcenter(const Vector3f& point1, const Vector3f& point2, const Vector3f& point3) noexcept {
		Vector3f v21 = point2 - point1;
		Vector3f v31 = point3 - point1;

		Vector3f cross_product = cross(v21, v31);
		float denominator = 2.0f * sqr_mag(cross_product);

		if (denominator == 0.0f) {
			return  Vector3f();
		}

		Vector3f cross21 = cross(v21, cross_product);
		Vector3f cross31 = cross(v31, cross_product);
		float mag21 = sqr_mag(v21);
		float mag31 = sqr_mag(v31);
		Vector3f circumcenter = point1 + (mag21 * cross31 - mag31 * cross21) / denominator;

		return circumcenter;
	}

	/// @brief Checks if a 3D vector is the null (zero) vector.
	/// @param vector The 3D vector to check.
	/// @return True if all components of the vector are zero; otherwise, false.
	__host__ __device__ inline bool IsNullVector(const Vector3f& vector) noexcept {
		return vector.x == 0.0f && vector.y == 0.0f && vector.z == 0.0f;
	}
}