# Require the cmake minimum version. This is the cmake version in x40 server.
# But I suggest using the newest version of cmake to get better support.
cmake_minimum_required(VERSION 3.10)


# Set project properties.
project(
	x40-cuda-example
	VERSION 1.0.0
	DESCRIPTION "A project to show how to use cuda in x40 server."
	LANGUAGES CXX CUDA
)


# Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if (EXISTS "${LOC_PATH}")
	message(FATAL_ERROR "You cannot build in a source directory (or any directory with"
						"CmakeLists.txt file). Please make a build subdirectory. Feel free"
						"to remove CMakeCache.txt and CMakeFiles.")
endif()

# Set the build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()


# Only do these if this is the main project, and not if it is included
# through add_subdirectory
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

	# Let's ensure -std=c++xx instead of -std=g++xx
	set(CMAKE_CXX_EXTENSIONS OFF)

	# Set the default cxx standard
	set(CMAKE_CXX_STANDARD 17)

	# Set default compile options
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra")

	# Let's nicely support folders in IDE's
	set_property(GLOBAL PROPERTY USE_FOLDERS ON)

	# check cuda
	include(CheckLanguage)
	check_language(CUDA)

	# Setup cuda architecture, 70 or 72 for volta architecture in TITAN V.
	# This is necessary since cmake version 3.18 .
	set(CMAKE_CUDA_ARCHITECTURES 72-real)

endif()

# multi add library wrtitten in cuda
add_library(multi_add multi_add.cu)

# executable file
add_executable(multi_add_cuda multi_add_cuda.cpp multi_add.h)
# link the library compiled by nvcc
target_link_libraries(multi_add_cuda PRIVATE multi_add)

