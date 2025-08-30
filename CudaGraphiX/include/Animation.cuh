#pragma once

#include "Header.cuh"
#include "Vector3f.cuh"
#include "Pole.cuh"
#include "Randomizer.cuh"

/// @brief Provides constants and functions for managing animation sequences and interacting with physics poles within the Animation namespace.
namespace Animation {
	
	/// @brief Starts the animation sequence.
	/// @details This function initiates the animation process, allowing for the rendering of physics poles and their interactions over time.
	void StartAnimation();

	/// @brief Stops the currently running animation.
	/// @details This function halts the animation process, cleaning up any resources used during the animation and resetting the state.
	void StopAnimation();

	/// @brief Returns the current number of frames processed or rendered.
	/// @details This function retrieves the count of frames that have been processed or rendered in the current animation sequence.
	/// @return The number of frames as an integer.
	int FramesCount();

	/// @brief Retrieves CUDA frames and returns a pointer to a Pole object containing them.
	/// @details This function pulls the CUDA frames from the animation system and returns a pointer to the corresponding Pole object.
	/// @return A pointer to a Pole object that holds the pulled CUDA frames.
	Physics::Pole* PullCudaFrames();
}