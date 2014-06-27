project(cucs)
cmake_minimum_required(VERSION 2.8)

# Supported build types:
#  Debug
#  Release
#  RelWithDebInfo
#
# to set build type: -DCMAKE_BUILD_TYPE=...

find_package(PkgConfig)

set(PROJECT_SOURCES_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Enable CUDA
find_package(CUDA)

#pkg_check_modules(SDL2 REQUIRED sdl2 SDL2_image SDL2_ttf gl glu glew)
#set(INCLUDEDIRS ${INCLUDEDIRS} ${SDL2_INCLUDE_DIRS})
#set(PROJECTLIBS ${PROJECTLIBS} ${SDL2_LIBRARIES} )
set(INCLUDEDIRS ${INCLUDEDIRS} )
set(PROJECTLIBS ${PROJECTLIBS} )

#Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes)
include_directories(${INCLUDEDIRS})

##Project sources
file(GLOB_RECURSE SRC_LIST ${PROJECT_SOURCES_PATH}/*)
set(SRC_LIST ${SRC_LIST} )

file(GLOB_RECURSE HEADERS_LIST ${CMAKE_CURRENT_SOURCE_DIR}/includes/*)

#Library
#add_library(${PROJECT_NAME} STATIC ${SRC_LIST})
cuda_add_library(${PROJECT_NAME} SHARED ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${PROJECTLIBS})

#Installation
message("Installation dir: ${CMAKE_INSTALL_PREFIX}")

# TODO: Consider: macro? for each PC file.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pkgconfig/cucs.pc.in ${CMAKE_CURRENT_SOURCE_DIR}/pkgconfig/cucs.pc @ONLY)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/pkgconfig/cucs.pc DESTINATION lib/pkgconfig)

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )
install(FILES ${HEADERS_LIST} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
