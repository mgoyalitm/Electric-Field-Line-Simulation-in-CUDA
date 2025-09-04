#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable> 
#include <algorithm>
#include <execution> 
#include <cuda_runtime.h>
#include <thread>
#include <chrono>

#include "Animation.cuh"

namespace Animation {

	std::queue<Physics::Pole*> frames;
	std::mutex mutex;
	std::atomic<bool> isAnimating = false;
	std::atomic<bool> animatingCompleted = true;
	std::atomic<int> frameCount = 0;
	std::thread thread;

	inline void PushFrame(Physics::Pole* frame);
	void AnimationThread();

	void StartAnimation() {

		std::unique_lock<std::mutex> lock(mutex);

		if (animatingCompleted == false || isAnimating) {
			isAnimating == true;
			return;
		}

		isAnimating = true;
		animatingCompleted = false;
		thread = std::thread(AnimationThread);
	}

	void StopAnimation() {
		std::unique_lock<std::mutex> lock(mutex);

		isAnimating = false;
		std::this_thread::sleep_for(std::chrono::milliseconds(100));

		if (thread.joinable()) {
			thread.join();
		}

		while (frames.empty() == false) {
			Physics::Pole* frame = frames.front();
			frames.pop();
			delete[] frame;
		}

		frameCount = frames.size();
	}

	int FramesCount() {
		return frameCount;
	}

	Physics::Pole* PullCudaFrames() {
		std::unique_lock<std::mutex> lock(mutex);

		if (frameCount < Constants::ParallelFrames) {
			return nullptr;
		}

		Physics::Pole* cuda_poles;
		CUDA_CHECK(cudaMalloc(&cuda_poles, Constants::PolesCount * Constants::ParallelFrames * sizeof(Physics::Pole)));

		for (int i = 0; i < Constants::ParallelFrames; i++) {
			int offset = i * Constants::PolesCount;

			Physics::Pole* frame = frames.front();
			CUDA_CHECK(cudaMemcpy(cuda_poles + offset, frame, Constants::PolesCount * sizeof(Physics::Pole), cudaMemcpyHostToDevice));
			frames.pop();
			frameCount = frames.size();
			delete[] frame;
		}

		return cuda_poles;
	}

	inline void PushFrame(Physics::Pole* frame) {
		std::unique_lock<std::mutex> lock(mutex);
		frames.push(frame);
		frameCount = frames.size();
	}

	void AnimationThread() {

		Physics::Pole* poles = Randomization::randomDipoles();

		static std::mutex animation_mutex;
		std::unique_lock<std::mutex> lock(animation_mutex);

		while (isAnimating)
		{
			while (isAnimating && frameCount >= Constants::MaxFrames) {
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}

			if (isAnimating == false) {
				break;
			}

			Physics::animate_poles_cpu(poles);

			Physics:: Pole* frame = new Physics::Pole[Constants::PolesCount];
			std::copy(poles, poles + Constants::PolesCount, frame);
			PushFrame(frame);
		}

		delete[] poles;
		animatingCompleted = true;
	}
}