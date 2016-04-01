FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options)
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
LINK_STATIC_MSVCR()
INCLUDE_DIRECTORIES(
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_BINARY_DIR}/src
	${Boost_INCLUDE_DIRS}
)
ADD_EXECUTABLE(nemo-app nemo.cpp)
TARGET_LINK_LIBRARIES(nemo-app nemo ${Boost_LIBRARIES})
SET_TARGET_PROPERTIES(nemo-app PROPERTIES OUTPUT_NAME "nemo")

IF(UNIX)
	INSTALL(TARGETS nemo-app DESTINATION bin)
ENDIF(UNIX)

# TODO: set up install on windows. We'll end up with the same name for the
# library and the application.
