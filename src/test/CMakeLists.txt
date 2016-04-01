FIND_PACKAGE(Boost REQUIRED COMPONENTS unit_test_framework)
SET(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/src/examples)
INCLUDE_DIRECTORIES(
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_BINARY_DIR}/src # for configuration files
	${EXAMPLES_DIR}
	${Boost_INCLUDE_DIR}
)

LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

IF(NOT Boost_USE_STATIC_LIBS)
	ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
ENDIF(NOT Boost_USE_STATIC_LIBS)

LINK_STATIC_MSVCR()

ADD_EXECUTABLE(test
	test.cpp
	stdp.cpp
	utils.cpp
	networks.cpp
	rtest.cpp
	c_api.cpp
	${EXAMPLES_DIR}/kuramoto.cpp
	${EXAMPLES_DIR}/torus.cpp
	${EXAMPLES_DIR}/random.cpp
	${EXAMPLES_DIR}/randomNetUtils.cpp)
TARGET_LINK_LIBRARIES(test nemo ${Boost_LIBRARIES})

ADD_EXECUTABLE(create_rtest_data
	create_rtest_data.cpp
	rtest.cpp
	utils.cpp
	${EXAMPLES_DIR}/kuramoto.cpp
	${EXAMPLES_DIR}/torus.cpp
	${EXAMPLES_DIR}/random.cpp
	${EXAMPLES_DIR}/randomNetUtils.cpp)
TARGET_LINK_LIBRARIES(create_rtest_data nemo ${Boost_LIBRARIES})

IF(INCLUDE_MPI)
	ADD_EXECUTABLE(stest serialisation.cpp)
	TARGET_LINK_LIBRARIES(stest nemo ${Boost_LIBRARIES})
ENDIF(INCLUDE_MPI)


IF(NEMO_MPI_ENABLED)

	FIND_PACKAGE(MPI)

	IF(NOT MPI_FOUND)
		MESSAGE(SEND_ERROR "MPI enabled, but no MPI implementation found on this system")
	ENDIF(NOT MPI_FOUND)

	INCLUDE_DIRECTORIES(
		${CMAKE_SOURCE_DIR}
		${MPI_INCLUDE_PATH}
		${Boost_INCLUDE_DIR})
	LINK_DIRECTORIES("${CMAKE_BINARY_DIR}/mpi")
	ADD_EXECUTABLE(mpi_test
		mpi_test.cpp
		${EXAMPLES_DIR}/torus.cpp
		utils.cpp
	)
	# TODO: do we need both nemo_mpi and nemo?
	TARGET_LINK_LIBRARIES(mpi_test nemo_mpi nemo ${MPI_LIBRARIES} ${Boost_LIBRARIES})

	# TODO: remove this again
	INSTALL(TARGETS mpi_test DESTINATION ${INSTALL_LIB_DIR})

ENDIF(NEMO_MPI_ENABLED)
