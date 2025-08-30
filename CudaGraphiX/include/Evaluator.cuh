	#pragma once

#include "Header.cuh"
#include "Vector3f.cuh"
#include "Pole.cuh"
#include "Randomizer.cuh"
#include "Animation.cuh"
#include "RenderData.hpp"

/// @brief Defines the Evaluation namespace, which can be used to group related functions, classes, or variables for field line evaluation purposes.
/// @details This namespace encapsulates functionality related to evaluating magnetic fields, rendering data, and managing field lines.
namespace Evaluation {
	
	/// @brief Initiates the evaluation process for field lines.
	/// @details This function starts the evaluation of field lines, which may involve calculations or rendering operations based on the current state of the poles and field lines.
	void StartFieldLineEvaluations();

	/// @brief Stops the evaluation of field lines.
	/// @details This function halts the ongoing evaluation of field lines, ensuring that no further calculations or rendering operations are performed.
	void StopFieldLineEvaluations();

	/// @brief Returns whether the number of evaluations is available.
	/// @details This function checks if the count of evaluations can be retrieved, which may be useful for monitoring or debugging purposes.
	/// @return True if the count of evaluations can be retrieved; otherwise, false.
	bool EvaluationsCount();

	/// @brief Pulls the render data for the current evaluation.
	/// @details This function retrieves the current render data, which includes the poles and field lines that have been evaluated.
	/// @return The current render data for the evaluation.
	Rendering::RenderData PullRenderData();
}
