#include <condition_variable>
#include <atomic>
#include <queue>
#include <mutex>
#include <algorithm>
#include <execution> 
#include <cuda_runtime.h>
#include <thread>
#include <chrono>

#include "Evaluator.cuh"

namespace Evaluation {

	std::queue<Rendering::RenderData> scenes;
	std::mutex mutex;
	std::atomic<bool> isEvaluating = false;
	std::atomic<bool> evaluationCompleted = true;
	std::atomic<int> scenesCount;
	std::thread thread;

	void EvaluationThread();
	inline void PushRenderData(Rendering::RenderData data);
	__global__ void FieldLineEvaluationKernel(Physics::Pole* poles_array, Geometry::Vector3f* field_lines, int* field_line_lengths);

	void StartFieldLineEvaluations() {
		std::unique_lock<std::mutex> lock(mutex);

		if (evaluationCompleted == false || isEvaluating) {
			isEvaluating = true;
			return;
		}

		isEvaluating = true;
		evaluationCompleted = false;
		thread = std::thread(EvaluationThread);
	}

	void StopFieldLineEvaluations() {

		isEvaluating = false;

		if (thread.joinable()) {
			thread.join();
		}

		std::unique_lock<std::mutex> lock(mutex);

		while (scenes.empty() == false) {
			Rendering::RenderData data = scenes.front();
			scenes.pop();
			Rendering::destroy(data);
		}

		scenesCount = static_cast<int>(scenes.size());
	}

	bool EvaluationsCount() {
		return scenesCount;
	}

	Rendering::RenderData PullRenderData() {
		std::unique_lock<std::mutex> lock(mutex);

		if (scenes.empty()) {
			return Rendering::RenderData(nullptr, nullptr, nullptr);
		}

		Rendering::RenderData data = scenes.front();
		scenes.pop();
		scenesCount = static_cast<int>(scenes.size());
		return data;
	}

	inline void PushRenderData(Rendering::RenderData data) {
		std::unique_lock<std::mutex> lock(mutex);

		if (scenesCount > Constants::MaxFrames + Constants::ParallelFrames || Rendering::validate(data) == false) {
			Rendering::destroy(data);
			return;
		}

		scenes.push(data);
		scenesCount = static_cast<int>(scenes.size());
	}

	void EvaluationThread() {
		std::this_thread::sleep_for(std::chrono::seconds(Constants::AnimationDelaySeconds));

		static std::mutex evaluation_mutex;
		std::unique_lock<std::mutex> lock(evaluation_mutex);

		Geometry::Vector3f* cuda_field_line_data;
		int* cuda_field_line_lengths;

		CUDA_CHECK(cudaMalloc(&cuda_field_line_data, Constants::CudaFieldLineDataLength * sizeof(Geometry::Vector3f)));
		CUDA_CHECK(cudaMalloc(&cuda_field_line_lengths, Constants::CudaPoleArrayLength * sizeof(int)));

		while (isEvaluating) {
			
			while (isEvaluating && (Animation::FramesCount() < Constants::ParallelFrames || scenesCount > Constants::MaxFrames - Constants::ParallelFrames)) {
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
			}

			if (isEvaluating == false) {
				break;
			}

			Physics::Pole* cuda_poles = Animation::PullCudaFrames();

			dim3 block_dim(Constants::ParallelFrames, 1, 1);
			dim3 thread_dim(Constants::PolesCount, Constants::FieldLinesPerPole, 1);

			FieldLineEvaluationKernel<<<block_dim, thread_dim>>>(cuda_poles, cuda_field_line_data, cuda_field_line_lengths);
			CUDA_CHECK(cudaDeviceSynchronize());

			for (int i = 0; i < Constants::ParallelFrames; i++) {
				Physics::Pole* poles = new Physics::Pole[Constants::PolesCount];
				Geometry::Vector3f** field_lines = new Geometry::Vector3f * [Constants::FieldLinesTotal];
				int* field_line_lengths = new int[Constants::FieldLinesTotal];

				CUDA_CHECK(cudaMemcpy(poles, cuda_poles + i * Constants::PolesCount, Constants::PolesCount * sizeof(Physics::Pole), cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaMemcpy(field_line_lengths, cuda_field_line_lengths + i * Constants::FieldLinesTotal, Constants::FieldLinesTotal * sizeof(int), cudaMemcpyDeviceToHost));

				for (int j = 0; j < Constants::FieldLinesTotal; j++) {
					int length = field_line_lengths[j];
					int start = (i * Constants::FieldLinesTotal + j) * Constants::MaxFieldLineLength;
					Geometry::Vector3f* field_line = new Geometry::Vector3f[length];
					CUDA_CHECK(cudaMemcpy(field_line, cuda_field_line_data + start, length * sizeof(Geometry::Vector3f), cudaMemcpyDeviceToHost));
					field_lines[j] = field_line;
				}

				Rendering::RenderData data(poles, field_lines, field_line_lengths);
				PushRenderData(data);
			}

			CUDA_CHECK(cudaFree(cuda_poles));
		}

		CUDA_CHECK(cudaFree(cuda_field_line_data));
		CUDA_CHECK(cudaFree(cuda_field_line_lengths));

		evaluationCompleted = true;
	}

	__global__ void FieldLineEvaluationKernel(Physics::Pole* poles_array, Geometry::Vector3f* field_lines, int* field_line_lengths) {

		Physics::Pole pole = poles_array[Constants::PolesCount * blockIdx.x + threadIdx.x];
		__shared__ Physics::Pole poles[Constants::PolesCount];

		if (threadIdx.y == 0) {
		}
			poles[threadIdx.x] = pole;

		__syncthreads();

		int field_index = blockIdx.x * Constants::FieldLinesTotal + threadIdx.x * Constants::FieldLinesPerPole + threadIdx.y;
		int offset_index = Constants::MaxFieldLineLength * field_index;
		int index = 2;
		float multiplier = pole.strength > 0 ? 1.0f : -1.0f;
		float dx = Constants::dx;
		float delta = Constants::delta;
		field_lines[offset_index] = pole.position;

		Geometry::Vector3f direction = multiplier * Physics::naive_field_strength(poles, Constants::PolesCount, pole.position);
		Geometry::Vector3f position = Physics::fibonacci_point(pole.position, direction, threadIdx.y, Constants::FieldLinesPerPole, Constants::delta);
		field_lines[offset_index + 1] = position;
		bool processing = true;

		while (processing && index < Constants::MaxFieldLineLength) {
			float size = 0.0f;

			while (size < delta) {
				Geometry::Vector3f field_strength = Physics::naive_field_strength(poles, Constants::PolesCount, position);
				position += field_strength * (multiplier * dx / Geometry::mag(field_strength));
				size += dx;
			}

			Geometry::Vector3f v1 = Geometry::normalize(field_lines[offset_index + index - 2] - field_lines[offset_index + index - 1]);
			Geometry::Vector3f v2 = Geometry::normalize(position - field_lines[offset_index + index - 1]);
			float angle = acosf(fminf(fmaxf(Geometry::dot(v1, v2), -1.0f), 1.0f));

			if (angle >= Constants::small_angle) {

				field_lines[offset_index + index++] = position;
			}

			for (int i = 0; i < Constants::PolesCount; i++) {
				if (i != threadIdx.x) {
					float dist = Geometry::dist(position, poles[i].position);

					if (dist < Constants::delta) {
						field_lines[offset_index + index++] = poles[i].position;
						processing = false;
					}
				}
			}
		}

		field_line_lengths[field_index] = index;
	}
}