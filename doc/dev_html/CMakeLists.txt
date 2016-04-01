##############################################################################
# Full source code documentation
##############################################################################
#
# The source code documentation is not included in either the source or binary
# packages because:
#
# 1. it is quite large (>6MB)
# 2. it is only useful to people who want to hack the internals of NeMo. Such
#    people would compile from source, so can generate this documentation as
#    needed.
#
##############################################################################

OPTION(NEMO_DOC_SOURCE_ENABLED "Create full source code documentation using Doxygen" ON)

IF(NEMO_DOC_SOURCE_ENABLED)

	SET(NEMO_DOC_SOURCE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/doc/html" CACHE STRING "Directory in which to place generated source code documentation")

	FIND_PACKAGE(Doxygen REQUIRED)

	SET(DOXYGEN_INPUT
		${CMAKE_SOURCE_DIR}/src/nemo.h
		${CMAKE_SOURCE_DIR}/src/nemo.hpp
		${CMAKE_SOURCE_DIR}/src/nemo
		${CMAKE_SOURCE_DIR}/src/nemo/cpu
		${CMAKE_SOURCE_DIR}/src/nemo/cuda
		${CMAKE_SOURCE_DIR}/src/nemo/network
		${CMAKE_SOURCE_DIR}/src/nemo/network/programmatic
	)

    IF(NEMO_MPI_ENABLED)
        LIST(APPEND DOXYGEN_INPUT ${CMAKE_SOURCE_DIR}/src/mpi)
    ENDIF(NEMO_MPI_ENABLED)
    
	# Convert dependency list to whitespace separated string
	SET(DOXYGEN_INPUT_STR "")
	FOREACH(ARG ${DOXYGEN_INPUT})
		SET(DOXYGEN_INPUT_STR "${DOXYGEN_INPUT_STR} ${ARG}")
	ENDFOREACH(ARG ${DOXYGEN_INPUT})

	# Fill in name, version number etc.
	CONFIGURE_FILE(
		${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
		${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
		ESCAPE_QUOTES @ONLY
	)

	# Doxygen creates a very large number of output files. List just the
	# top-level one.
	SET(DOXYGEN_OUTPUT ${NEMO_DOC_SOURCE_OUTPUT_DIR}/index.html)

	# Run Doxygen
	ADD_CUSTOM_COMMAND(
		OUTPUT ${DOXYGEN_OUTPUT}
		COMMAND ${DOXYGEN_EXECUTABLE}
		DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile ${DOXYGEN_INPUT}
	)

	# Need a named target in order for dependencies to work out
	ADD_CUSTOM_TARGET(api-doc-source
		DEPENDS ${DOXYGEN_OUTPUT}
		COMMENT "Generating source code documentation"
		VERBATIM
	)

	# If enabled in configuration, always create source code docs
	ADD_DEPENDENCIES(nemo api-doc-source)

ENDIF(NEMO_DOC_SOURCE_ENABLED)
