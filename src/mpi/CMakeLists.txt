# check for MPI library
FIND_PACKAGE(MPI REQUIRED)
FIND_PACKAGE(CUDA)

IF(NOT MPI_FOUND)
	MESSAGE(SEND_ERROR "MPI enabled, but no MPI implementation found on this system")
ENDIF(NOT MPI_FOUND)

INCLUDE_DIRECTORIES(
	${MPI_INCLUDE_PATH}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_SOURCE_DIR}/src/examples
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_BINARY_DIR}/src # for configuration
	${Boost_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
)

LIST(APPEND MPI_SRCS Master.cpp Worker.cpp Mapper.cpp SpikeQueue.cpp MpiTimer.cpp mpiUtils.cpp WorkerLogger.cpp)

IF(NEMO_MPI_DEBUG_TIMING)
LIST(APPEND MPI_SRCS MpiTimer.cpp)
ENDIF(NEMO_MPI_DEBUG_TIMING)

ADD_LIBRARY(nemo_mpi SHARED ${MPI_SRCS})

SET_TARGET_PROPERTIES(nemo_mpi PROPERTIES
		DEFINE_SYMBOL NEMO_EXPORTS
)

add_dependencies(nemo_mpi nemo nemo_base  nemo_cuda nemo_cpu)

TARGET_LINK_LIBRARIES(nemo_mpi nemo nemo_base nemo_cuda nemo_cpu ${MPI_LIBRARIES} ${Boost_LIBRARIES})

 #TODO: can share this code with examples cmakefile
IF(UNIX)
	SET(INSTALL_EXAMPLE_BIN_DIR "share/nemo/examples")
ENDIF(UNIX)

IF(WIN32)
	SET(INSTALL_EXAMPLE_BIN_DIR "examples")
ENDIF(WIN32)

INSTALL(TARGETS nemo_mpi DESTINATION ${INSTALL_LIB_DIR})
