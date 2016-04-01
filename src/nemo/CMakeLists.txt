OPTION(NEMO_WEIGHT_FIXED_POINT_SATURATION "Use saturating arithmetic when summing weights, to avoid overflow" ON)
MARK_AS_ADVANCED(NEMO_WEIGHT_FIXED_POINT_SATURATION)

SET(BACKEND_LIBS "nemo_cpu")
IF(NEMO_CUDA_ENABLED)
	IF(NOT NEMO_CUDA_DYNAMIC_LOADING)
		LIST(APPEND BACKEND_LIBS "nemo_cuda")
	ENDIF(NOT NEMO_CUDA_DYNAMIC_LOADING)
ENDIF(NEMO_CUDA_ENABLED)

IF(NOT WIN32)
	SET(FIND_LIBRARY_USE_LIB64_PATHS TRUE)
	FIND_PATH(LTDL_INCLUDE_DIR ltdl.h)
	FIND_LIBRARY(LTDL_LIBRARY NAMES ltdl)
	IF(NOT LTDL_INCLUDE_DIR OR NOT LTDL_LIBRARY)
    MESSAGE(STATUS "The ltdl library does not seem to be installed in your system. CMake will try to download it and build it. If you have root access in this machine, you might want to install libltdl-dev.")
		FIND_PROGRAM(M4_bin m4)
		IF(NOT M4_bin)
			MESSAGE(STATUS "To build ltdl we need m4, and it's not in your machine either. I'll try to build it too.")
			INCLUDE(ExternalProject)
			ExternalProject_Add(M4
				# TODO: should we download this particular version or m4-latest.tar.gz?
				URL http://ftp.gnu.org/gnu/m4/m4-1.4.17.tar.gz
				PREFIX ./m4
				CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
				)
			ExternalProject_Get_Property(M4 install_dir)
			SET(M4_bin ${install_dir}/bin/m4)
		ENDIF(NOT M4_bin)
		INCLUDE(ExternalProject)
		ExternalProject_Add(LTDL
			DEPENDS M4
			# TODO: should we write this url (with version nb included) or fetch it from the development repo? (there is no latest for ltdl)
			URL http://ftpmirror.gnu.org/libtool/libtool-2.4.6.tar.gz
			PREFIX ./libtool
			CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> M4=${M4_bin}
			)
		ExternalProject_Get_Property(LTDL install_dir)
		SET(LTDL_INCLUDE_DIR ${install_dir}/include)
		SET(LTDL_LIBRARY ${install_dir}/lib/libltdl.so)
	ENDIF(NOT LTDL_INCLUDE_DIR OR NOT LTDL_LIBRARY)
ELSE(NOT WIN32)
	SET(LTDL_INCLUDE_DIR "")
	SET(LTDL_LIBRARY "")
ENDIF(NOT WIN32)

LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
INCLUDE_DIRECTORIES(
	${CMAKE_BINARY_DIR}/src  # for configuration files
	${CMAKE_SOURCE_DIR}/src
	${Boost_INCLUDE_DIR}
	${LTDL_INCLUDE_DIR}
)

SET(BUILD_SHARED_LIBS TRUE)

LINK_STATIC_MSVCR()

ADD_LIBRARY(nemo_base
	Axon.cpp
	ConfigurationImpl.cpp
	ConnectivityMatrix.cpp
	FiringBuffer.cpp
	fixedpoint.cpp
	Network.cpp
	NetworkImpl.cpp
	Neuron.cpp
	Neurons.cpp
	NeuronType.cpp
	Simulation.cpp
	SimulationBackend.cpp
	OutgoingDelays.cpp
	Plugin.cpp
	ReadableNetwork.cpp
	RNG.cpp
	runtime/RCM.cpp
	StdpFunction.cpp
	StdpProcess.cpp
)

TARGET_LINK_LIBRARIES(nemo_base ${LTDL_LIBRARY} ${Boost_LIBRARIES})
SET_TARGET_PROPERTIES(nemo_base PROPERTIES DEFINE_SYMBOL NEMO_BASE_EXPORTS)


ADD_LIBRARY(nemo nemo.cpp nemo_c.cpp Configuration.cpp)
SET_TARGET_PROPERTIES(nemo PROPERTIES DEFINE_SYMBOL NEMO_EXPORTS)
TARGET_LINK_LIBRARIES(nemo nemo_base ${BACKEND_LIBS})


INSTALL(TARGETS nemo nemo_base DESTINATION ${INSTALL_LIB_DIR})

INSTALL(FILES
		exception.hpp
		types.h
		Configuration.hpp
		Network.hpp
		Simulation.hpp
		ReadableNetwork.hpp
		${CMAKE_CURRENT_BINARY_DIR}/config.h
	DESTINATION ${INSTALL_INCLUDE_DIR}/nemo
)


##############################################################################
# General system configuration

INCLUDE(CheckCXXSourceCompiles)
INCLUDE(CheckFunctionExists)

# CheckFunctionExists does not seem to work with GCC builtins
CHECK_CXX_SOURCE_COMPILES("int main() { __builtin_clzll(0); }" HAVE_BUILTIN_CLZLL)
CHECK_CXX_SOURCE_COMPILES("int main() { __builtin_clz(0); }" HAVE_BUILTIN_CLZ)
CHECK_CXX_SOURCE_COMPILES("int main() { __builtin_ctzll(0); }" HAVE_BUILTIN_CTZLL)
CHECK_FUNCTION_EXISTS(posix_memalign HAVE_POSIX_MEMALIGN)

# log2 is part of C99 standard, so Microsoft support is patchy
CHECK_CXX_SOURCE_COMPILES("#include <math.h>\nint main() { log2(double(0)); }" HAVE_LOG2)

# CUDA-specific configuration
# TODO: ideally this should be done in the cuda subdirectory
SET(CUDA_CUPRINTF_DIR "" CACHE PATH "Directory in which to find cuPrintf source and header file")

IF(CUDA_CUPRINTF_DIR)
	FIND_PATH(CUPRINTF_INCLUDE_DIR cuPrintf.cuh PATHS ${CUDA_CUPRINTF_DIR})
	FIND_FILE(CUPRINTF_SOURCE cuPrintf.cu ${CUDA_CUPRINTF_DIR})
ELSE()
	FIND_PATH(CUPRINTF_INCLUDE_DIR cuPrintf.cuh)
	FIND_FILE(CUPRINTF_SOURCE cuPrintf.cu)
ENDIF(CUDA_CUPRINTF_DIR)

SET(HAVE_CUPRINTF FALSE)
IF(CUPRINTF_INCLUDE_DIR AND CUPRINTF_SOURCE)
	SET(HAVE_CUPRINTF TRUE)
ENDIF(CUPRINTF_INCLUDE_DIR AND CUPRINTF_SOURCE)

MARK_AS_ADVANCED(FORCE CUPRINTF_INCLUDE_DIR CUPRINTF_SOURCE)

IF(NOT Boost_LIBRARY_DIRS)
	MESSAGE(WARNING "Boost_LIBRARY_DIRS is not set, which may cause linker errors. Either set this variable or set BOOST_ROOT")
ELSE(NOT Boost_LIBRARY_DIRS)
	LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
ENDIF(NOT Boost_LIBRARY_DIRS)


# We rely on 32-bit floats
INCLUDE(CheckTypeSize)
CHECK_TYPE_SIZE(float FLOAT_SIZE)
IF(NOT ${FLOAT_SIZE} EQUAL 4)
	MESSAGE(SEND_ERROR "float does not have expected size (4B)")
ENDIF(NOT ${FLOAT_SIZE} EQUAL 4)


# Unless we're on a windows/MSVC platform we need stdint.h
IF(NOT HAVE_STDINT_H AND NOT MSVC)
	MESSAGE(FATAL_ERROR "No valid stdint.h file found on this platform")
ENDIF(NOT HAVE_STDINT_H AND NOT MSVC)


SET(NEMO_SYSTEM_PLUGIN_DIR "${INSTALL_LIB_DIR}/nemo/plugins")
IF(WIN32)
	SET(NEMO_SYSTEM_PLUGIN_DIR "plugins")
ENDIF(WIN32)

SET(NEMO_USER_PLUGIN_DIR ".nemo/plugins")
IF(WIN32)
	SET(NEMO_USER_PLUGIN_DIR "Application Data/NeMo/plugins")
ENDIF(WIN32)

CONFIGURE_FILE(
	${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
	${CMAKE_CURRENT_BINARY_DIR}/config.h)

ADD_SUBDIRECTORY(plugins)
ADD_SUBDIRECTORY("cpu")
IF(NEMO_CUDA_ENABLED)
	ADD_SUBDIRECTORY("cuda")
ENDIF(NEMO_CUDA_ENABLED)
