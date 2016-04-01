FIND_PACKAGE(OpenMP)

IF(OPENMP_FOUND)
	# User may still choose to build without OpenMP support, for whatever reason.
	# On MSVC OpenMP introduces additional dynamic library requirements.
	OPTION(NEMO_CPU_OPENMP_ENABLED "Use OpenMP for parallelisation of the CPU backend" TRUE)
ELSE(OPENMP_FOUND)
	MESSAGE(WARNING "OpenMP not found. CPU backend will be compiled without")
ENDIF(OPENMP_FOUND)

IF(NEMO_CPU_OPENMP_ENABLED AND OPENMP_FOUND)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF(NEMO_CPU_OPENMP_ENABLED AND OPENMP_FOUND)

INCLUDE_DIRECTORIES(
	${CMAKE_BINARY_DIR}/src  # for configuration files
	${CMAKE_SOURCE_DIR}/src
	${Boost_INCLUDE_DIR}
)

ADD_LIBRARY(nemo_cpu Simulation.cpp Neurons.cpp)
SET_TARGET_PROPERTIES(nemo_cpu PROPERTIES DEFINE_SYMBOL NEMO_CPU_EXPORTS)
TARGET_LINK_LIBRARIES(nemo_cpu nemo_base ${Boost_LIBRARIES})
INSTALL(TARGETS nemo_cpu DESTINATION ${INSTALL_LIB_DIR})

ADD_SUBDIRECTORY(plugins)
