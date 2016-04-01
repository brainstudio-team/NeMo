##############################################################################
# C++/C API reference documentation
#
# This is generates HTML files for the public API for C and C++. A fuller
# documentation of the internals can also be generated, but this is found in a
# separate directory, as that is only useful for developers, whereas the
# documentation here is useful for users.
##############################################################################

OPTION(NEMO_DOC_API_REBUILD "Create C API HTML reference manual using Doxygen" FALSE)

IF(NEMO_DOC_API_REBUILD)

	FIND_PACKAGE(Doxygen REQUIRED)

	SET(DOXYGEN_SOURCES 
		${CMAKE_SOURCE_DIR}/src/nemo.h
		${CMAKE_SOURCE_DIR}/src/nemo/Configuration.hpp
		${CMAKE_SOURCE_DIR}/src/nemo/Network.hpp
		${CMAKE_SOURCE_DIR}/src/nemo/Simulation.hpp
		${CMAKE_SOURCE_DIR}/src/nemo/types.h
		# Some classes required for correct processing, but not really desired
		# in the output.
		${CMAKE_SOURCE_DIR}/src/nemo/network/Generator.hpp
		${CMAKE_SOURCE_DIR}/src/nemo/ReadableNetwork.hpp
	) 

	# Convert source list to space-separated string
	FOREACH(arg ${DOXYGEN_SOURCES})
		SET(DOXYGEN_SOURCES_SS "${DOXYGEN_SOURCES_SS} ${arg}")
	ENDFOREACH(arg ${DOXYGEN_SOURCES})

	# Create Doxyfile
	CONFIGURE_FILE(
		${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
		${CMAKE_CURRENT_BINARY_DIR}/Doxyfile 
		ESCAPE_QUOTES @ONLY
	)

	# Run doxygen to generate HTML
	ADD_CUSTOM_COMMAND(
		OUTPUT html/index.html 
			# A bunch of other files are also generated, but that above output
			# is sufficient for dependency resolution
		COMMAND ${DOXYGEN_EXECUTABLE}
		DEPENDS ${DOXYGEN_SOURCES}
	)

	ADD_CUSTOM_TARGET(api-doc-html
		DEPENDS html/index.html
		COMMENT "Generating HTML reference for C/C++ API"
		VERBATIM
	)

	ADD_DEPENDENCIES(nemo api-doc-html)

	IF(UNIX)
		SET(DOC_API_INSTALL_DIR share/doc/nemo)
	ENDIF(UNIX)

	IF(WIN32)
		SET(DOC_API_INSTALL_DIR Documentation)
	ENDIF(WIN32)

	INSTALL(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION ${DOC_API_INSTALL_DIR})

ENDIF(NEMO_DOC_API_REBUILD)
