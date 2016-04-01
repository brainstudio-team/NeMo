##############################################################################
# CUDA configuration

SET(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
SET(CUDA_HOST_COMPILATION_CPP TRUE)
FIND_PACKAGE(CUDA REQUIRED)

IF(CUDA_BUILD_EMULATION)
	ADD_DEFINITIONS(-D__DEVICE_EMULATION__)
ENDIF()

SET(NEMO_CUDA_DEVICE_MAJOR "1"
	CACHE STRING "Major device generation number for CUDA code generation. Default=1")
SET(NEMO_CUDA_DEVICE_MINOR "2"
	CACHE STRING "Minor device generation number for CUDA code generation. Default=2")

IF(NEMO_CUDA_DEVICE_MAJOR EQUAL 1 AND NEMO_CUDA_DEVICE_MINOR LESS 2)
	# The 1.2 minimum is because of the need for shared memory atomics
	MESSAGE(FATAL_ERROR "Minimum CUDA device generation is 1.2")
ENDIF()

LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_${NEMO_CUDA_DEVICE_MAJOR}${NEMO_CUDA_DEVICE_MINOR}")


IF(NEMO_CUDA_DEBUG_TRACE)
	IF(NOT CUDA_BUILD_EMULATION AND NEMO_CUDA_DEVICE_MAJOR LESS 2)
		IF(HAVE_CUPRINTF)
			CUDA_INCLUDE_DIRECTORIES(${CUPRINTF_INCLUDE_DIR})
			INCLUDE_DIRECTORIES(${CUPRINTF_INCLUDE_DIR})
		ELSE()
			MESSAGE(WARNING "Cannot provide debugging trace for CUDA backend. Either device emulation must be enabled (CUDA_BUILD_EMULATION) or cuPrintf must be available")
		ENDIF(HAVE_CUPRINTF)
	ENDIF(NOT CUDA_BUILD_EMULATION AND NEMO_CUDA_DEVICE_MAJOR LESS 2)
ENDIF(NEMO_CUDA_DEBUG_TRACE)



# FindCUDA sets a number of variables which or not of relevance to our build.
# Mark these as advanced, so as not to clutter the configuration.
MARK_AS_ADVANCED(FORCE
	FOUND_CUT
	FOUND_CUT_INCLUDE
	CUDA_SDK_ROOT_DIR
	CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE
	CUDA_HOST_COMPILATION_CPP)

CUDA_INCLUDE_DIRECTORIES(
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_BINARY_DIR}/src # for configuration files
	${Boost_INCLUDE_DIR}
  )

# C++ only includes
INCLUDE_DIRECTORIES(
	${CMAKE_SOURCE_DIR}
	${CMAKE_BINARY_DIR} # for configuration files
	${CUDA_INCLUDE_DIRS}
	${Boost_INCLUDE_DIR}
  )


##############################################################################
SET(BUILD_SHARED_LIBS TRUE)

CUDA_ADD_LIBRARY(nemo_cuda SHARED
	kernel.cu
	Bitvector.cpp
	ConnectivityMatrix.cpp
	create_simulation.cpp
	Simulation.cpp
	DeviceAssertions.cpp
	FiringBuffer.cpp
	FiringStimulus.cpp
	GlobalQueue.cpp
	LocalQueue.cpp
	Neurons.cpp
	Outgoing.cpp
	construction/FcmIndex.cpp
	runtime/Delays.cpp
	runtime/RCM.cpp
	device_memory.cpp
	devices.cpp
	log.cpp
  )

SET_TARGET_PROPERTIES(nemo_cuda PROPERTIES DEFINE_SYMBOL NEMO_CUDA_EXPORTS)
TARGET_LINK_LIBRARIES(nemo_cuda nemo_base)

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()

INSTALL(TARGETS nemo_cuda DESTINATION ${INSTALL_LIB_DIR})

ADD_SUBDIRECTORY(plugins)
