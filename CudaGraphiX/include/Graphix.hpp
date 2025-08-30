#pragma once

#include "RenderData.hpp"

/// @brief Provides functions for initializing an OpenGL window, running a simulation, and rendering a scene.
/// @details This namespace contains functions that handle the setup and execution of graphical simulations using OpenGL.
namespace Rendering {

	/// @brief Runs a simulation and returns whether it was successful.
	/// @details This function executes the main simulation loop and handles any necessary updates.
	/// @return True if the simulation completed successfully; otherwise, false.
	bool RenderSimulation();

	/// @brief Renders the current scene.
	/// @details This function handles the rendering of the scene to the OpenGL window.
	/// @param data The RenderData object containing the necessary information for rendering, including poles and field lines.
	void RenderScene(Rendering::RenderData data);

	/// @brief Retrieves the current animation time.
	/// @details This function returns the current time in the animation sequence, which can be used for timing and synchronization purposes.
	/// @return The current animation time as a double value.
	double GetAnimationTime();
}