#include<iostream>

#include <GLFW/glfw3.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>

#include "Animation.cuh"
#include "Evaluator.cuh"
#include "Graphix.hpp"
#include "RenderData.hpp"

std::thread frame_thread;
std::mutex frame_mutex;
std::atomic<bool> isSimulationRunning = true;

void SimulationThread();

int main(void)
{
	Animation::StartAnimation();
	Evaluation::StartFieldLineEvaluations();

	frame_thread = std::thread(SimulationThread);
	Rendering::RenderSimulation();

	isSimulationRunning = false;
	if (frame_thread.joinable()) {
		frame_thread.join();
	}

	Animation::StopAnimation();
	Evaluation::StopFieldLineEvaluations();

	return 0;
}

void SimulationThread() {

	while (isSimulationRunning) {

		if (Evaluation::EvaluationsCount() > 0) {
			Rendering::RenderData scene = Evaluation::PullRenderData();
			std::unique_lock<std::mutex> lock(frame_mutex);
			Rendering::RenderScene(scene);
			//std::this_thread::sleep_for(std::chrono::milliseconds(20));
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(Constants::FrameDuration));
	}
}