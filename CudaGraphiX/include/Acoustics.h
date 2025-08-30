#pragma once

#include "RenderData.hpp"
#include "Header.cuh"

namespace Acoustics {

	/// @brief Initializes the acoustics system.
	/// @summary This function sets up the necessary components for the acoustics system to function properly. It may involve allocating resources, configuring settings, or preparing data structures required for acoustics processing.
	/// @return True if the acoustics system was successfully initialized; otherwise, false.
	void InitializeAcoustics();

	/// @brief Shuts down the acoustics system and releases associated resources.
	/// @summary This function cleans up and frees any resources allocated during the initialization or operation of the acoustics system. It ensures that the system is properly shut down to prevent memory leaks or other issues.
	void ShutdownAcoustics();

	/// @brief Generates acoustic data based on the provided rendering information.
	/// @summary This function processes the rendering data to create acoustics, which can be used for sound simulation or audio effects in a 3D environment.
	/// @param data The rendering data used to generate acoustics.
	/// @param animation_time The current time in the animation sequence, which may influence the acoustics generation.
	void GenerateAcoustics(Rendering::RenderData data, double animation_time);
}