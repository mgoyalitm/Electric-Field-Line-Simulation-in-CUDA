#pragma once

#include "Header.cuh"
#include "Pole.cuh"
#include "Vector3f.cuh"

/// @brief Checks whether the given RenderData object is valid for rendering operations.
/// @details This function verifies that the RenderData object contains valid pointers and data, ensuring that it can be used safely in rendering operations.
namespace Rendering {

	/// @brief Represents data required for rendering, including poles and field lines.
	/// @details This structure holds pointers to poles and a vector of field lines, which are used for rendering the magnetic field.
	struct RenderData {

		/// @brief Pointer to a Physics::Pole object or an array of such objects.
		/// @details This pointer points to the poles that influence the magnetic field, allowing for efficient access during rendering.
		Physics::Pole* poles;

		/// @brief A pointer to a pointer to a Vector3f object, typically used to represent an array of field lines in 3D geometry.
		/// @details This pointer points to an array of field lines, where each field line is represented as a series of 3D vectors (Vector3f).
		Geometry::Vector3f** field_lines;

		/// @brief Pointer to an array containing the lengths of field lines.
		/// @details This pointer points to an array of integers, where each integer represents the length of a corresponding field line in the field_lines array.
		int* field_line_lengths;

		/// @brief Renders data for a set of physics poles and their associated field lines.
		/// @details This constructor initializes the RenderData object with the provided poles and field lines, allowing for rendering operations to be performed efficiently.
		/// @param poles Pointer to an array of Physics::Pole objects representing the poles to render.
		/// @param field_lines Pointer to an array of pointers, each pointing to an array of Geometry::Vector3f objects representing the field lines for each pole.
		/// @param field_line_lengths Pointer to an array of integers specifying the length of each field line array.
		inline RenderData(Physics::Pole* poles, Geometry::Vector3f** field_lines, int* field_line_lengths)
			: poles(poles), field_lines(field_lines), field_line_lengths(field_line_lengths) {
		}

		/// @brief Initializes a RenderData object with all pointer members set to nullptr.
		/// @details This default constructor sets all pointer members to nullptr, indicating that the RenderData object is not yet initialized with valid data.
		inline RenderData() 
			: poles(nullptr), field_lines(nullptr), field_line_lengths(nullptr) {
		}
	};

	/// @brief Destroys the object and releases any associated resources.
	/// @details This destructor cleans up the resources used by the RenderData object, including deleting the poles and field lines to prevent memory leaks.
	inline void destroy(RenderData data) {

		if (data.poles != nullptr) {
			delete[] data.poles;
		}

		for (int i = 0; i < Constants::FieldLinesTotal; i++) {
			if (data.field_lines != nullptr) {
				delete[] data.field_lines[i];
			}
		}

		if (data.field_line_lengths != nullptr) {
			delete[] data.field_line_lengths;
		}
	}

	/// @brief Checks whether the given RenderData object is valid.
	/// @details This function checks if the RenderData object contains valid pointers and data, ensuring that it can be used safely in rendering operations.
	/// @param data The RenderData object to validate.
	/// @return true if the RenderData object is valid; otherwise, false.
	inline bool validate(RenderData data) {
		return data.poles != nullptr && data.field_lines != nullptr && data.field_line_lengths != nullptr;
	}
}