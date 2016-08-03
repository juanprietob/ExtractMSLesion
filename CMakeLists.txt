cmake_minimum_required(VERSION 2.8)
cmake_policy(VERSION 2.8)

project(ExtractMSLesion)
option(BUILD_TESTING "tests" ON)

# Setting paths
set(CMAKE_MODULE_PATH
${${PROJECT_NAME}_SOURCE_DIR}/CMake
${${PROJECT_NAME}_BINARY_DIR}/CMake
${CMAKE_MODULE_PATH}
)

# find ITK headers
find_package(ITK COMPONENTS ITKIOImageBase ITKImageFilterBase ITKStatistics ITKIONIFTI ITKIONRRD REQUIRED)
include(${ITK_USE_FILE})

# find SlicerExecutionModel headers
find_package(SlicerExecutionModel REQUIRED)
include(${SlicerExecutionModel_USE_FILE})


# find SlicerExecutionModel headers
FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)

message(STATUS ${PYTHON_INCLUDE_PATH})

INCLUDE_DIRECTORIES("/usr/local/Cellar/python/2.7.11/Frameworks/Python.framework/Versions/2.7/include/python2.7"
	"/usr/local/Cellar/python/2.7.11/Frameworks/Python.framework/Versions/2.7/include/python2.7")
LINK_DIRECTORIES("/usr/local/Cellar/python/2.7.11/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config")

SET(PYTHON_LIBRARIES python2.7 dl)

add_subdirectory(src)
add_subdirectory(sh)

# Tests
if(BUILD_TESTING)
  #include(CTest)
  #include(ExternalData)
  #add_SUBDIRECTORY(Testing)
endif()

if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/FADTTS)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")
  
configure_file(CMake/ExtractMSLesion.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/ExtractMSLesion.cmake" @ONLY)
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/ExtractMSLesion.cmake"  
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)