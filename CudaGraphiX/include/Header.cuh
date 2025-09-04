#pragma once

// IntelliSense-friendly __syncthreads definition
#ifdef __INTELLISENSE__
inline void __syncthreads() {}
#endif

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); } } while(0)

/// @brief Defines a set of constants used for animation and simulation parameters, including pole counts, field line properties, frame management, CUDA array sizes, and mathematical values.
/// @details This namespace encapsulates various constants that are essential for configuring the behavior of animations and simulations, particularly in the context of magnetic field visualizations and CUDA computations.
namespace Constants {

#pragma region Gobal
	/// @brief Defines the mathematical constant pi as a float value.
	/// @details This constant is used in various geometric calculations, such as those involving circles and spheres.
	constexpr float PI = 3.14159265358979323846f;

	/// @brief Defines a constant representing twice the value of PI.
	/// @details This constant is useful in calculations involving full rotations or circular paths.
	constexpr float TWO_PI = 2.0f * PI;

	/// @brief Defines a constant representing half of pi as a float value.
	/// @details This constant is often used in trigonometric calculations, such as those involving angles in radians.
	constexpr float HALF_PI = PI / 2.0f;

	/// @brief Represents the speed of sound in air at standard conditions, in meters per second.
	/// @details This constant is used in simulations involving acoustics or sound propagation, providing a reference value for calculations related to sound speed.
	constexpr float SoundSpeed = 343.3f;
#pragma endregion

#pragma region Camera and Rendering
	/// @brief Represents the camera's angular velocity in degrees per unit time.
	/// @details This constant defines how quickly the camera rotates around a focal point, affecting the speed of camera movement in animations or visualizations.
	constexpr float CameraAngularVelocityDegree = 2.5f;

	/// @brief Represents the camera's angular velocity in radians per unit time.
	/// @details This constant defines how quickly the camera rotates around a focal point in radians, affecting the speed of camera movement in animations or visualizations.
	constexpr float CameraAngularVelocityRadians = PI * CameraAngularVelocityDegree / 180.0f;

	/// @brief Defines the camera elevation angle in degrees.
	/// @details This constant specifies the angle at which the camera is elevated above the horizontal plane, influencing the perspective and viewpoint in visualizations.
	constexpr float CameraElevationDegrees = 15.0f;

	/// @brief Calculates the camera elevation angle in radians from the elevation angle in degrees.
	/// @details This constant is derived from the CameraElevationDegrees constant and is used in calculations that require the elevation angle to be expressed in radians, such as in camera positioning and orientation.
	constexpr float CameraElevationRadians = PI * CameraElevationDegrees / 180.0f;

	/// @brief Defines a constant representing the camera distance.
	/// @details This constant specifies the distance of the camera from the origin or focal point, which can influence the perspective and field of view in visualizations.
	constexpr float CameraDistance = 35.0f;

	/// @brief Defines the minimum draw distance as a constant expression.
	/// @details This constant specifies the closest distance at which objects will be rendered, helping to avoid rendering artifacts that can occur when objects are too close to the camera.
	constexpr float MinDrawDistance = 0.01f * CameraDistance;

	/// @brief Defines the maximum draw distance as a constant expression.
	/// @details This constant specifies the furthest distance at which objects will be rendered, helping to optimize performance by culling distant objects that are not visible.
	constexpr float MaxDrawDistance = 100.0f * CameraDistance;

	/// @brief Represents the field of view in degrees.
	/// @details This constant defines the vertical field of view angle for the camera, affecting how much of the scene is visible in the rendered output.
	constexpr float FieldOfViewDegrees = 45.0f;

	/// @brief Calculates the field of view in radians from the field of view in degrees.
	/// @details This constant is derived from the FieldOfViewDegrees constant and is used in calculations that require the field of view to be expressed in radians, such as in perspective projection matrices.
	constexpr float FieldOfViewRadians = PI * FieldOfViewDegrees / 180.0f;

	/// @brief Defines the thickness of a field line as a constant value.
	/// @details This constant is used to specify the visual thickness of field lines in the rendering process, affecting their appearance in the final output.
	constexpr float FieldLineThickness = 1.0f;

	/// @brief Defines the radius of a pole sphere as a constant floating-point value.
	/// @details This constant is used to specify the size of the spheres representing poles in the rendering process, affecting their visibility and prominence in the final output.
	constexpr float PoleSphereRadius = 0.3f;

	/// @brief Represents the minimum alpha value for a field line.
	/// @details This constant defines the minimum transparency level for field lines in the rendering process, ensuring they remain visible while allowing for some degree of transparency.
	constexpr float MinAlpha = 0.18f;

	/// @brief Represents the maximum alpha value for a field line.
	/// @details This constant defines the maximum opacity level for field lines in the rendering process, ensuring they are fully visible without being overly opaque.
	constexpr float MaxAlpha = 0.54f;

	/// @brief Represents the difference between MaxAlpha and MinAlpha as a constant expression.
	/// @details This constant is used to calculate the range of alpha values for field lines, allowing for dynamic adjustments in transparency during rendering.
	constexpr float AlphaChange = MaxAlpha - MinAlpha;

	/// @brief Represents the red component value for a line color.
	/// @details This constant defines the intensity of the red color channel for lines in the rendering process, contributing to the overall color appearance of the lines.
	constexpr float LineColorRed = 1.5f;

	/// @brief Represents the green component value for a line color.
	/// @details This constant defines the intensity of the green color channel for lines in the rendering process, contributing to the overall color appearance of the lines.
	constexpr float LineColorGreen = 0.725f;

	/// @brief Represents the blue component value for a line color.
	/// @details This constant defines the intensity of the blue color channel for lines in the rendering process, contributing to the overall color appearance of the lines.
	constexpr float LineColorBlue = 0.0f;

	/// @brief Represents the red color value for a pole as a constant floating-point value.
	/// @details This constant defines the intensity of the red color channel for poles in the rendering process, contributing to the overall color appearance of the poles.
	constexpr float PoleColorRed = 1.0f;

	/// @brief Represents the color value for a green pole.
	/// @details This constant defines the intensity of the green color channel for poles in the rendering process, contributing to the overall color appearance of the poles.
	constexpr float PoleColorGreen = 1.0f;

	/// @brief Represents the blue color value for a pole.
	/// @details This constant defines the intensity of the blue color channel for poles in the rendering process, contributing to the overall color appearance of the poles.
	constexpr float PoleColorBlue = 0.5f;

	/// @brief Defines the radius of a pole as a constant expression.
	/// @details This constant is used to specify the size of the poles in the rendering process, affecting their visibility and prominence in the final output.
	constexpr float PoleRadius = 0.002f * CameraDistance;
#pragma endregion

#pragma region Acoustics
	/// @brief Defines a constant expression for Doppler scaling based on sound speed and camera distance.
	/// @details This constant is used to adjust the Doppler effect in simulations involving sound, taking into account the speed of sound and the distance of the camera from the sound source.
	constexpr float DopplerScaling = 0.001f * SoundSpeed / CameraDistance;

	/// @brief Represents the base frequency value.
	/// @details This constant defines a reference frequency used in sound simulations, often corresponding to the standard pitch of musical notes (A4 = 440 Hz).
	constexpr float BaseFrequency = 440.0f;

	/// @brief Represents the minimum frequency value.
	/// @details This constant defines the lowest frequency that can be used in sound simulations, ensuring that audio remains within a perceptible range.
	constexpr float MinFrequency = 65.0f;

	/// @brief Defines the maximum frequency value.
	/// @details This constant sets the upper limit for frequency in sound simulations, ensuring that audio remains within a perceptible range and does not exceed typical human hearing capabilities.
	constexpr float MaxFrequency = 2000.0f;

	/// @brief Represents the maximum amplitude value as a constant floating-point number.
	/// @details This constant defines the peak amplitude for audio signals, ensuring that sound levels remain within a safe and manageable range to prevent distortion or damage to audio equipment.
	constexpr float MaxAmplitude = 32760.0f;
#pragma endregion

#pragma region Frame Management
	/// @brief Defines the number of frames to be processed in parallel.
	/// @details This constant specifies how many frames can be processed simultaneously, which can help optimize performance in multi-threaded or parallel processing environments.
	constexpr int ParallelFrames = 240;

	/// @brief Defines the maximum number of frames allowed.
	/// @details This constant limits the number of frames that can be stored in the animation sequence, ensuring efficient memory usage and performance.
	constexpr int MaxFrames = ParallelFrames * 4;

	/// @brief Defines a constant representing frames per second.
	/// @details This constant is used to specify the frame rate of the animation, which can affect the smoothness and performance of the rendering process.
	constexpr int FPS = 100;

	/// @brief Defines the duration of a single frame in milliseconds based on the frames per second (FPS).
	/// @details This constant is calculated by dividing 1000 milliseconds by the FPS value, providing a time duration for each frame in the animation or simulation.
	constexpr int FrameTimeMilliseconds = 1000.0f / FPS;
#pragma endregion

#pragma region Field-line Evaluation And CUDA Kernel

	/// @brief Defines a constant representing the number of poles.
	/// @details This constant is used to specify the number of poles in the animation sequence, which can be adjusted based on the requirements of the simulation.
	constexpr int PolesCount = 128;

	/// @brief Defines the number of field lines per pole as a constant.
	/// @details This constant determines how many field lines are generated for each pole in the animation, which can affect the visual complexity and performance of the simulation.
	constexpr int FieldLinesPerPole = 1024 / PolesCount;

	/// @brief Defines the number of field lines per frame as a constant.
	/// @details This constant specifies how many field lines are generated for each frame in the animation sequence, which can help control the detail and performance of the rendering process.
	constexpr int FieldLinesTotal = PolesCount * FieldLinesPerPole;

	/// @brief Defines a constant expression for the length of a CUDA pole array.
	/// @details This constant is used to determine the size of the array that holds pole data in CUDA, which is essential for managing memory and performance in GPU computations.
	constexpr int CudaPoleArrayLength = ParallelFrames * FieldLinesTotal;

	/// @brief Specifies the maximum length of a field line.
	/// @details This constant defines the maximum number of points that can be stored in a single field line, which is crucial for managing memory and performance in simulations involving magnetic fields.
	constexpr int MaxFieldLineLength = 256;

	/// @brief Defines a constant representing the length of CUDA field line data.
	/// @details This constant is used to calculate the total size of the field line data array in CUDA, which is essential for efficient memory management and performance in GPU computations.
	constexpr int CudaFieldLineDataLength = CudaPoleArrayLength * MaxFieldLineLength;

	/// @brief Defines the number of evaluation steps as a constant.
	/// @details This constant specifies how many steps are taken during the evaluation phase of the simulation or animation, which can affect the accuracy and performance of the calculations.
	constexpr int EvaluationSteps = 2;

	/// @brief Defines a constant representing a small dx value for floating-point calculations.
	/// @details This constant is used to ensure numerical stability in calculations, particularly when dealing with small differences in floating-point values.
	constexpr float dx = CameraDistance / 960.0f;

	/// @brief Defines the minimum and maximum dx values for floating-point calculations.
	/// @details These constants are used to constrain the range of dx values, ensuring that calculations remain stable and accurate within specified limits.
	constexpr float dx_min = 0.05f * dx;

	/// @brief Defines the maximum dx value for floating-point calculations.
	/// @details This constant sets an upper limit on the dx value, preventing excessively large increments that could lead to instability in numerical computations.
	constexpr float dx_max = 20.0f * dx;

	/// @brief Defines a constant representing a small delta value for floating-point calculations.
	/// @details This constant is used to define a small increment for calculations that require a finer resolution, such as in simulations or numerical methods.
	constexpr float delta = dx * EvaluationSteps;

	/// @brief Defines the minimum and maximum delta values for floating-point calculations.
	/// @details These constants are used to constrain the range of delta values, ensuring that calculations remain stable and accurate within specified limits.
	constexpr float delta_min = dx_min * EvaluationSteps;

	/// @brief Defines the maximum delta value for floating-point calculations.
	/// @details This constant sets an upper limit on the delta value, preventing excessively large increments that could lead to instability in numerical computations.
	constexpr float delta_max = dx_max * EvaluationSteps;

	/// @brief Defines a constant representing the bending step in degrees.
	/// @details This constant is used in calculations involving angles, particularly in scenarios where bending or rotation is involved.
	constexpr float BendingStep = 60.0f;

	/// @brief Defines a small angle in radians based on the bending step.
	/// @details This constant is calculated by dividing PI by the bending step, providing a small angle value for use in geometric or trigonometric calculations.
	constexpr float BendingAngle = PI / BendingStep;

	/// @brief Defines the maximum number of iterations for line evaluation.
	/// @details This constant is calculated based on the number of evaluation steps and the bending step, providing a limit on the number of iterations that can be performed during line evaluation to ensure performance and prevent infinite loops.
	constexpr int MaxLineEvaluationIterations = EvaluationSteps * BendingStep * 8.0f;
#pragma endregion

#pragma region Animation
	/// @brief Specifies the delay duration for an animation in seconds.
	/// @details This constant defines how long the animation will pause or wait before proceeding to the next step, allowing for controlled timing in the animation sequence.
	constexpr int AnimationDelaySeconds = 0;

	/// @brief Defines the number of steps in an animation.
	/// @details This constant indicates how many discrete steps the animation will take to transition from one state to another, allowing for smoother animations.
	constexpr int AnimationSteps = 4;

	/// @brief Defines a constant scaling factor for time-related calculations.
	/// @details This constant is used to scale time values in the simulation or animation, allowing for adjustments in speed or duration.
	constexpr float TimeScaling = 0.001f;

	/// @brief Defines a constant representing a time step value.
	/// @details This constant is used to define the duration of each animation step, which can be adjusted to control the speed of the animation.
	constexpr float dt = TimeScaling / (AnimationSteps * FPS);
#pragma endregion

#pragma region Placement and randomitaion
	/// @brief Defines the maximum initial speed as a constant floating-point value.
	/// @details This constant is used to limit the initial speed of objects in the simulation, ensuring that they do not exceed a specified threshold.
	constexpr float MaxInitialSpeed = 1.0f;

	/// @brief Defines the minimum initial speed as a constant floating-point value.
	/// @details This constant is used to ensure that objects in the simulation have a minimum initial speed, preventing them from being stationary or moving too slowly.
	constexpr float MinInitialSpeed = MaxInitialSpeed / 10.0f;

	/// @brief Represents the maximum allowed placement distance.
	/// @details This constant defines the furthest distance from a reference point where objects can be placed, ensuring they remain within a manageable range for the simulation.
	constexpr float MaxPlacementDistance = 0.001f;

	/// @brief Represents the minimum allowed placement distance.
	/// @details This constant is used to ensure that objects in the simulation are not placed too close to a reference point, preventing overlap or unrealistic positioning.
	constexpr float MinPlacementDistance = MaxPlacementDistance / 10.0f;

	/// @brief Represents the maximum allowed mass as a constant floating-point value.
	/// @details This constant is used to limit the mass of objects in the simulation, ensuring they do not exceed a specified threshold that could affect performance or realism.
	constexpr float MaxMass = 10.0f;

	/// @brief Represents the minimum allowed mass value.
	/// @details This constant is used to ensure that objects in the simulation have a minimum mass, preventing
	constexpr float MinMass = MaxMass / 10.0f;

	/// @brief Defines the maximum strength value as a constant.
	/// @details This constant is used to limit the strength of forces or fields in the simulation, ensuring they remain within a manageable range for realistic behavior.
	constexpr float MaxStrength = 10.0f;

	/// @brief Defines the minimum strength value as a constant.
	/// @details This constant is used to ensure that forces or fields in the simulation have a minimum strength, preventing them from being too weak to have a noticeable effect.
	constexpr float MinStrength = MaxStrength / 10.0f;
#pragma endregion
}