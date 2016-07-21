cmake_minimum_required(VERSION 2.8.4)
#set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -ftemplate-depth-1024")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread -lpng -ljpeg")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/lib")

project(ddfd)
#cmake_policy(SET CMP0048 OLD)

find_package(JPEG)
include_directories(${JPEG_INCLUDE_DIR})

include(../cpplib/dlib-19.0/dlib/cmake NO_POLICY_SCOPE)
set(DLIB_JPEG_SUPPORT ON CACHE "" INTERNAL FORCE)
set(DLIB_PNG_SUPPORT ON CACHE "" INTERNAL FORCE)
set(DLIB_GIF_SUPPORT ON CACHE "" INTERNAL FORCE)
set(DLIB_NO_GUI_SUPPORT ON CACHE "" INTERNAL FORCE)
set(USE_AVX_INSTRUCTIONS ON CACHE "" INTERNAL FORCE)

set(source_files imagenet_finetune.cpp)

include_directories(${CMAKE_SOURCE_DIR})

add_executable(ddfd_train ddfd_train.cpp ../cpplib/dlib-19.0/dlib/all/source.cpp)
target_link_libraries(ddfd_train dlib)

add_executable(fhog_extract_faces fhog_extract_faces.cpp ../cpplib/dlib-19.0/dlib/all/source.cpp)
target_link_libraries(fhog_extract_faces dlib)

add_executable(crop_faces crop_faces.cpp ../cpplib/dlib-19.0/dlib/all/source.cpp)
target_link_libraries(crop_faces dlib)

add_executable(resnet34_train resnet34_train.cpp ../cpplib/dlib-19.0/dlib/all/source.cpp)
target_link_libraries(resnet34_train dlib)