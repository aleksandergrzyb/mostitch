FILE (GLOB_RECURSE project_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)
SET (project_BIN ${PROJECT_NAME})

ADD_EXECUTABLE(${project_BIN} ${project_SRCS} ${project_MOC_SRCS_GENERATED})
TARGET_LINK_LIBRARIES(${project_BIN} ${project_LIBS})
SET_TARGET_PROPERTIES(${project_BIN} PROPERTIES VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}" OUTPUT_NAME ${project_BIN} CLEAN_DIRECT_OUTPUT 1)

INSTALL(TARGETS ${project_BIN} DESTINATION bin)

#
# Third party libraries
#
INCLUDE_DIRECTORIES ("${MAINFOLDER}/thirdparty/include")
#LINK_DIRECTORIES ("${MAINFOLDER}/thirdparty/lib")

#TARGET_LINK_LIBRARIES (mostitch ibconfig.dylib, config++, opencv_calib3d, libopencv_contrib.dylib, libopencv_core.dylib, libopencv_features2d.dylib)

FILE (GLOB OPEN_CV "${MAINFOLDER}/thirdparty/lib/libopencv*.dylib")
FILE (GLOB CONFIG "${MAINFOLDER}/thirdparty/lib/libconfig*.dylib")
TARGET_LINK_LIBRARIES (mostitch PRIVATE ${OPEN_CV})
TARGET_LINK_LIBRARIES (mostitch PRIVATE ${CONFIG})