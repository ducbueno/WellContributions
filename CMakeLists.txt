cmake_minimum_required(VERSION "3.16.3")

project("WellContributions")

link_directories("/usr/lib/x86_64-linux-gnu")

add_executable("${PROJECT_NAME}" "src/WellContributions.cpp" "src/main.cpp")
target_link_libraries("${PROJECT_NAME}" OpenCL)
