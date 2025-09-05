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
	bool show_bounding_box = false;
	bool show_axes = false;
	bool full_screen = false;
	bool show_rainbow_lines = false;


	void DrawFieldLines();
	void DrawSphere(float cx, float cy, float cz, float radius = 0.2f, int slices = 12, int stacks = 12);

	bool RenderSimulation()
	{
		if (!glfwInit()) {
			return false;
		}

		if (full_screen)
		{
			GLFWmonitor* monitor = glfwGetPrimaryMonitor();
			const GLFWvidmode* mode = glfwGetVideoMode(monitor);
			GLFWwindow* window = glfwCreateWindow(mode->width, mode->height, "Electric Field Simulation", monitor, NULL);
		}
		else
		{
			window = glfwCreateWindow(1600, 1080, "Electric Field Simulation", NULL, NULL);

		}

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
		double start_time = glfwGetTime();
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

			float aspect_ratio = (float)width / (float)height;
			float top = Constants::MinDrawDistance * tanf(Constants::FieldOfViewRadians / 2.0f);
			float bottom = -top;
			float right = top * aspect_ratio;
			float left = -right;

			glFrustum(left, right, bottom, top, Constants::MinDrawDistance, Constants::MaxDrawDistance);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			float radius = Constants::CameraDistance;
			float azimuth = (float)time * Constants::CameraAngularVelocityRadians;
			float elevation = Constants::CameraElevationRadians;
			float camX = radius * cosf(elevation) * sinf(azimuth);
			float camY = radius * sinf(elevation);
			float camZ = radius * cosf(elevation) * cosf(azimuth);


			// Build forward (view) vector
			float dirX = -camX;
			float dirY = -camY;
			float dirZ = -camZ;

			// Normalize forward
			float cameraDistance = 1.0f / sqrtf(dirX * dirX + dirY * dirY + dirZ * dirZ);
			dirX *= cameraDistance;
			dirY *= cameraDistance;
			dirZ *= cameraDistance;

			// Up vector world-space
			float upX = 0.0f, upY = 1.0f, upZ = 0.0f;

			// Right = dir × up
			float rightX = dirY * upZ - dirZ * upY;
			float rightY = dirZ * upX - dirX * upZ;
			float rightZ = dirX * upY - dirY * upX;

			// Recompute up = right × dir
			upX = rightY * dirZ - rightZ * dirY;
			upY = rightZ * dirX - rightX * dirZ;
			upZ = rightX * dirY - rightY * dirX;

			// Construct the view matrix manually
			GLfloat m[16] = {
				rightX,  upX, -dirX, 0.0f,
				rightY,  upY, -dirY, 0.0f,
				rightZ,  upZ, -dirZ, 0.0f,
				0.0f,    0.0f, 0.0f, 1.0f
			};

			glMultMatrixf(m);
			glTranslatef(-camX, -camY, -camZ);


			glLineWidth(Constants::FieldLineThickness);

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


		for (int line_index = 0; line_index < Constants::FieldLinesTotal; line_index++) {

			int length = scene.field_line_lengths[line_index];

			bool connected = length < Constants::MaxFieldLineLength - 1;
			glBegin(GL_LINE_STRIP);

			for (int i = 0; i < length; i++) {

				float t = static_cast<float>(i) / (length - 1);

				float alpha = connected
					? Constants::MaxAlpha - Constants::AlphaChange * 4.0f * t * (1.0f - t)
					: Constants::MaxAlpha * (1 - t);


				if (show_rainbow_lines) {

					if (scene.poles[line_index / Constants::FieldLinesPerPole].strength > 0.0f) {
						t = 1 - t;
					}



					float r = 1.5 - 0.6 * t, g = 1.0 - 0.4 * (1 - t), b = 0;
					glColor4f(r, g, b, alpha);
				}
				else {
					glColor4f(Constants::LineColorRed, Constants::LineColorGreen, Constants::LineColorBlue, alpha);
				}
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
			glColor3f(Constants::PoleColorGreen, Constants::PoleColorGreen, Constants::PoleColorBlue);
			DrawSphere(point.x, point.y, point.z, 0.04f);
		}

		if (show_axes || show_bounding_box) {
			glBegin(GL_LINES);
		}

		if (show_axes) {
			float r = Constants::CameraDistance / 5.0f;

			glColor4f(1.0f, 0.0f, 0.0f, 0.2f);
			glVertex3f(-r, 0, 0);
			glVertex3f(r, 0, 0);

			glColor4f(0.0f, 1.0f, 0.0f, 0.2f);
			glVertex3f(0.0f, -r, 0.0f);
			glVertex3f(0.0f, r, 0.0f);

			glColor4f(0.0f, 0.0f, 1.0f, 0.2f);
			glVertex3f(0.0f, 0.0f, -r);
			glVertex3f(0.0f, 0.0f, r);
		}

		if (show_bounding_box) {
			float r = Constants::CameraDistance;
			glColor4f(1.0f, 1.0f, 1.0f, 0.1f);

			for (int i = 0; i < 2; i++) {
				float x1 = (i == 0) ? r : -r;

				for (int j = 0; j < 2; j++) {
					float y1 = (j == 0) ? r : -r;
					float z1 = (i + j) % 2 == 0 ? r : -r;

					for (int k = 0; k < 3; k++) {
						float x2 = (k == 0) ? -x1 : x1;
						float y2 = (k == 1) ? -y1 : y1;
						float z2 = (k == 2) ? -z1 : z1;

						glVertex3f(x1, y1, z1);
						glVertex3f(x2, y2, z2);
					}
				}
			}


		}

		if (show_axes || show_bounding_box) {
			glEnd();
		}

	}

	void DrawSphere(float cx, float cy, float cz, float radius, int slices, int stacks) {

		for (int i = 0; i <= stacks; ++i) {

			float lat0 = Constants::PI * (-0.5f + (float)(i - 1) / stacks);
			float z0 = sinf(lat0);
			float zr0 = cosf(lat0);

			float lat1 = Constants::PI * (-0.5f + (float)i / stacks);
			float z1 = sinf(lat1);
			float zr1 = cosf(lat1);

			glBegin(GL_QUAD_STRIP);

			for (int j = 0; j <= slices; ++j) {
				float lng = Constants::TWO_PI * (float)(j - 1) / slices;
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