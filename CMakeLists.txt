#########################################
#										#
#			SIMPLE TAG VISUALIZER		#
#										#
#########################################

cmake_minimum_required (VERSION 2.6)
project(SimpleTagVisualizer)

# Collect all files
file(GLOB_RECURSE SIMPLETAGVISUALIZER_SOURCE_FILES_SOURCE_FILES "src/*.cpp" "src/*.h" "src/*.inl")

# Compiler specific options
if(CMAKE_COMPILER_IS_GNUCXX)
	add_definitions(-std=c++11)
endif()

## Load PCL
find_package(PCL 1.8 QUIET REQUIRED)
if(${PCL_FOUND} AND ${VTK_FOUND} AND ${Boost_FOUND} AND ${FLANN_FOUND} STREQUAL "TRUE")
	MESSAGE( STATUS "Adding properly PCL and its dependencies")
	add_definitions(${PCL_DEFINITIONS})
else()
	MESSAGE( FATAL_ERROR "Couldnt Add PCL and/or it's dependencies")
endif()

# Find 3rd party packages
find_package(OpenCV 3.1 REQUIRED)
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
LINK_DIRECTORIES(${OpenCV_LIB_DIR})

# Configure application
set(APP_NAME SimpleTagVisualizer)

add_executable(${APP_NAME} ${SIMPLETAGVISUALIZER_SOURCE_FILES_SOURCE_FILES})
target_include_directories(${APP_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_include_directories(${APP_NAME} PUBLIC ${PCL_INCLUDE_DIRS})
target_link_libraries(${APP_NAME} LINK_PUBLIC ${PCL_LIBRARIES})

target_include_directories(${APP_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${APP_NAME} LINK_PUBLIC ${OpenCV_LIBS})
