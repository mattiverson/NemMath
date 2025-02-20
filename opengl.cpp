#include "common.h"
#include "nem_math.h"
#include <glad4_3/glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstring>
#include <limits>
#include <cmath>
#include <random>

struct Mem
{
    U32 hProg;
    float x[1920];
    float y[1920];
    double ref[1920];
};

int main()
{
    Mem* pMem = new Mem;
    Mem& mem = *pMem;
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
    GLFWwindow* window = glfwCreateWindow(1920, 1080, "Henlo uwu", nullptr, nullptr);
    if (!window)
    {
        puts("Couldn't create GLFW window.");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        puts("Failed to init glad.");
    }

    delete pMem;
}