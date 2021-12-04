# CMakeList.txt : CMake project for Neural_Net, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Neural_Net")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
 
# Add source to this project's executable.
add_executable (${PROJECT_NAME} "src/Neural_Net.cpp" "src/Neural_Net.h")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
target_include_directories(${PROJECT_NAME} PRIVATE "include")
target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)

# TODO: Add tests and install targets if needed.
