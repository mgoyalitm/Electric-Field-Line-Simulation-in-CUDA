#include "Graphix.hpp"
#include "Acoustics.h"

#include <GLFW/glfw3.h>
#include <iostream>
#include "RenderData.hpp"
#include <thread>
#include <mutex>
#include <chrono>

namespace Rendering {

	GLFWwindow* window;
	constexpr float background = 1.0f;
	RenderData scene;
	std::thread thread;
	std::mutex mutex;
	double time = 0;
	double start_time = 0;

	void DrawFieldLines();
	void DrawSphere(float cx, float cy, float cz, float radius = 0.2f, int slices = 12, int stacks = 12);

	bool RenderSimulation()
	{
		if (!glfwInit()) {
			return false;
		}

		//window = glfwCreateWindow(1600, 1080, "Electric Field Simulation", NULL, NULL);
		GLFWmonitor* monitor = glfwGetPrimaryMonitor();
		const GLFWvidmode* mode = glfwGetVideoMode(monitor);
		GLFWwindow* window = glfwCreateWindow(mode->width, mode->height, "Electric Field Simulation", monitor, NULL);

		if (!window) {
			glfwTerminate();
			return false;
		}

		glfwMakeContextCurrent(window);
		glEnable(GL_DEPTH_TEST);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		start_time = glfwGetTime();
		const float fov = 45.0f;
		const float aspect = 1600.0f / 1080.0f;
		const float n = 0.1f, f = 100.0f;
		float t = n * tanf(fov * 0.5f * 3.1415926f / 180.0f);
		float r = t * aspect;
		glFrustum(-r, r, -t, t, n, f);

		glMatrixMode(GL_MODELVIEW);

		while (!glfwWindowShouldClose(window))
		{
			time = glfwGetTime() - start_time;

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			int width, height;
			glfwGetFramebufferSize(window, &width, &height);
			if (height == 0) height = 1;

			glViewport(0, 0, width, height);

			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();

			float aspect = (float)width / (float)height;
			float fovY = 60.0f;
			float near = 0.1f;
			float far = 10000.0f;
			float top = tanf(fovY * 0.5f * 3.1415926f / 180.0f) * near;
			float bottom = -top;
			float right = top * aspect;
			float left = -right;
			glFrustum(left, right, bottom, top, near, far);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glTranslatef(0.0f, 0.0f, -Constants::CameraDistance);
			glRotatef((float)(time * Constants::CameraAngularVelocityDegree), 0, 1, 0);

			glLineWidth(1.4f);

			{
				std::unique_lock<std::mutex> lock(mutex);
				Acoustics::GenerateAcoustics(scene, time);
				DrawFieldLines();
			}

			glfwSwapBuffers(window);
			glfwPollEvents();
		}

		glfwTerminate();
		Acoustics::ShutdownAcoustics();
		return true;
	}

	void DrawFieldLines() {

		if (Rendering::validate(scene) == false) {
			return;
		}

		static float min_alpha = 0.1f;
		static float max_alpha = 0.4f;

		for (int line_index = 0; line_index < Constants::FieldLinesTotal; line_index++) {

			int length = scene.field_line_lengths[line_index];

			bool connected = length < Constants::MaxFieldLineLength - 1;
			glBegin(GL_LINE_STRIP);

			for (int i = 0; i < length; i++) {

				float t = static_cast<float>(i) / (length - 1);
				float alpha = connected
					? max_alpha - (max_alpha - min_alpha) * (1.0f - 4.0f * (t - 0.5f) * (t - 0.5f))
					: max_alpha - (max_alpha - (min_alpha / 4.0f)) * t;

				glColor4f(1.0f, 0.725f, 0.0f, alpha);
				Geometry::Vector3f point = scene.field_lines[line_index][i];

				if (i > 0)
				{
					Geometry::Vector3f point_prev = scene.field_lines[line_index][i - 1];
					float distance = Geometry::dist(point, point_prev);
					distance /= Geometry::mag(scene.poles[i / Constants::FieldLinesPerPole].position) * Constants::CameraDistance;

					if (Geometry::dist(point, point_prev) > 4) {
						break;
					}
				}

				glVertex3f(point.x, point.y, point.z);
			}

			glEnd();

		}

		for (int i = 0; i < Constants::PolesCount; i++)
		{
			Geometry::Vector3f point = scene.poles[i].position;
			glColor3f(1.0f, 1.0f, 0.5f);
			DrawSphere(point.x, point.y, point.z, 0.04f);
		}
	}

	void DrawSphere(float cx, float cy, float cz, float radius, int slices, int stacks) {
		for (int i = 0; i <= stacks; ++i) {
			float lat0 = 3.1415926f * (-0.5f + (float)(i - 1) / stacks);
			float z0 = sinf(lat0);
			float zr0 = cosf(lat0);

			float lat1 = 3.1415926f * (-0.5f + (float)i / stacks);
			float z1 = sinf(lat1);
			float zr1 = cosf(lat1);

			glBegin(GL_QUAD_STRIP);
			for (int j = 0; j <= slices; ++j) {
				float lng = 2 * 3.1415926f * (float)(j - 1) / slices;
				float x = cosf(lng);
				float y = sinf(lng);

				glVertex3f(cx + radius * x * zr0, cy + radius * y * zr0, cz + radius * z0);
				glVertex3f(cx + radius * x * zr1, cy + radius * y * zr1, cz + radius * z1);
			}
			glEnd();
		}
	}

	void RenderScene(Rendering::RenderData data) {
		std::unique_lock<std::mutex> lock(mutex);
		Rendering::destroy(scene);
		scene = data;
	}

	double GetAnimationTime() {
		return time;
	}
}