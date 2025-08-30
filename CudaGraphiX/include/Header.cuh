#pragma once

// IntelliSense-friendly __syncthreads definition
#ifdef __INTELLISENSE__
inline void __syncthreads() {}
#endif

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); } } while(0)

/// @brief Defines a set of constants used for animation and simulation parameters, including pole counts, field line properties, frame management, CUDA array sizes, and mathematical values.
/// @details This namespace encapsulates various constants that are essential for configuring the behavior of animations and simulations, particularly in the context of magnetic field visualizations and CUDA computations.
namespace Constants {


	/// @brief Defines the number of field lines per pole as a constant.
	/// @details This constant determines how many field lines are generated for each pole in the animation, which can affect the visual complexity and performance of the simulation.
	constexpr int FieldLinesPerPole = 12;

	/// @brief Defines a constant representing the number of poles.
	/// @details This constant is used to specify the number of poles in the animation sequence, which can be adjusted based on the requirements of the simulation.
	constexpr int PolesCount = 1024 / FieldLinesPerPole;

	/// @brief Defines the number of field lines per frame as a constant.
	/// @details This constant specifies how many field lines are generated for each frame in the animation sequence, which can help control the detail and performance of the rendering process.
	constexpr int FieldLinesTotal = PolesCount * FieldLinesPerPole;

	/// @brief Defines the number of frames to be processed in parallel.
	/// @details This constant specifies how many frames can be processed simultaneously, which can help optimize performance in multi-threaded or parallel processing environments.
	constexpr int ParallelFrames = 120;

	/// @brief Defines the maximum number of frames allowed.
	/// @details This constant limits the number of frames that can be stored in the animation sequence, ensuring efficient memory usage and performance.
	constexpr int MaxFrames = ParallelFrames * 4;

	/// @brief Defines a constant expression for the length of a CUDA pole array.
	/// @details This constant is used to determine the size of the array that holds pole data in CUDA, which is essential for managing memory and performance in GPU computations.
	constexpr int CudaPoleArrayLength = ParallelFrames * FieldLinesTotal;

	/// @brief Specifies the maximum length of a field line.
	/// @details This constant defines the maximum number of points that can be stored in a single field line, which is crucial for managing memory and performance in simulations involving magnetic fields.
	constexpr int MaxFieldLineLength = 256;

	/// @brief Defines a constant representing the length of CUDA field line data.
	/// @details This constant is used to calculate the total size of the field line data array in CUDA, which is essential for efficient memory management and performance in GPU computations.
	constexpr int CudaFieldLineDataLength = CudaPoleArrayLength * MaxFieldLineLength;

	/// @brief Specifies the delay duration for an animation in seconds.
	/// @details This constant defines how long the animation will pause or wait before proceeding to the next step, allowing for controlled timing in the animation sequence.
	constexpr int AnimationDelaySeconds = 10;

	/// @brief Defines the number of steps in an animation.
	/// @details This constant indicates how many discrete steps the animation will take to transition from one state to another, allowing for smoother animations.
	constexpr int AnimationSteps = 16;

	/// @brief Defines a constant representing a time step value.
	/// @details This constant is used to define the duration of each animation step, which can be adjusted to control the speed of the animation.
	constexpr float dt = 0.00000005f;

	/// @brief Defines a constant representing a small dx value for floating-point calculations.
	/// @details This constant is used to ensure numerical stability in calculations, particularly when dealing with small differences in floating-point values.
	constexpr float dx = 0.025f;

	/// @brief Defines a constant representing a small delta value for floating-point calculations.
	/// @details This constant is used to define a small increment for calculations that require a finer resolution, such as in simulations or numerical methods.
	constexpr float delta = dx * 4;

	/// @brief Defines the mathematical constant pi as a float value.
	/// @details This constant is used in various geometric calculations, such as those involving circles and spheres.
	constexpr float PI = 3.14159265358979323846f;

	/// @brief Defines a constant representing twice the value of PI.
	/// @details This constant is useful in calculations involving full rotations or circular paths.
	constexpr float TWO_PI = 2.0f * PI;

	/// @brief Defines a constant representing half of pi as a float value.
	/// @details This constant is often used in trigonometric calculations, such as those involving angles in radians.
	constexpr float HALF_PI = PI / 2.0f;

	/// @brief Represents a small angle in radians, equivalent to one degree.
	/// @details This constant is used in calculations where small angles are involved, such as in physics simulations or graphics rendering.
	constexpr float small_angle = PI / 30.0f;

	/// @brief Represents the camera's angular velocity in degrees per unit time.
	/// @details This constant defines how quickly the camera rotates around a focal point, affecting the speed of camera movement in animations or visualizations.
	constexpr float CameraAngularVelocityDegree = 2.5f;
	
	/// @brief Represents the camera's angular velocity in radians per unit time.
	/// @details This constant defines how quickly the camera rotates around a focal point in radians, affecting the speed of camera movement in animations or visualizations.
	constexpr float CameraAngularVelocityRadians = PI * CameraAngularVelocityDegree / 180.0f;

	/// @brief Defines a constant representing the camera distance.
	/// @details This constant specifies the distance of the camera from the origin or focal point, which can influence the perspective and field of view in visualizations.
	constexpr float CameraDistance = 25.0f;

	/// @brief Defines the maximum initial speed as a constant floating-point value.
	/// @details This constant is used to limit the initial speed of objects in the simulation, ensuring that they do not exceed a specified threshold.
	constexpr float MaxInitialSpeed = 0.0f;

	/// @brief Represents the maximum allowed placement distance.
	/// @details This constant defines the furthest distance from a reference point where objects can be placed, ensuring they remain within a manageable range for the simulation.
	constexpr float MaxPlacementDistance = 0.0001f;

	/// @brief Represents the maximum allowed mass as a constant floating-point value.
	/// @details This constant is used to limit the mass of objects in the simulation, ensuring they do not exceed a specified threshold that could affect performance or realism.
	constexpr float MaxMass = 10.0f;

	/// @brief Defines the maximum strength value as a constant.
	/// @details This constant is used to limit the strength of forces or fields in the simulation, ensuring they remain within a manageable range for realistic behavior.
	constexpr float MaxStrength = 10.0f;

	/// @brief Represents the speed of sound in air at standard conditions, in meters per second.
	/// @details This constant is used in simulations involving acoustics or sound propagation, providing a reference value for calculations related to sound speed.
	constexpr float SoundSpeed = 343.3f;

	/// @brief Defines a constant scaling factor.
	/// @details This constant is used to scale various parameters in the simulation or animation, allowing for adjustments in size or intensity.
	constexpr float ScalingFactor = 0.1f * SoundSpeed / CameraDistance;

	/// @brief Defines a constant representing frames per second.
	constexpr int FPS = 50;

	/// @brief Defines the duration of a single frame in milliseconds based on the frames per second (FPS).
	constexpr int FrameDuration = 1000 / FPS;

	constexpr float VelocityScaling = dt * AnimationSteps * FPS;
}