# Users shouldn't normally need to build the documentation themselves.
OPTION(NEMO_DOC_MAN_REBUILD "Rebuild API documentation (man pages)" OFF)

SET(MAN_DIR ${CMAKE_CURRENT_BINARY_DIR}/man3)
SET(GENERATED_MAN_PAGES
	${MAN_DIR}/nemo-c.3
	${MAN_DIR}/nemo_Network.3
	${MAN_DIR}/nemo_Simulation.3
	${MAN_DIR}/nemo_Configuration.3
)


IF(NEMO_DOC_MAN_REBUILD)

	FIND_PACKAGE(Doxygen REQUIRED)

	SET(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
	SET(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/..)
	CONFIGURE_FILE(${DOXYFILE_IN} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile ESCAPE_QUOTES @ONLY)

	# Run Doxygen
	ADD_CUSTOM_COMMAND(
		OUTPUT
			# Files we are interested in
			${MAN_DIR}/nemo.h.3
			${MAN_DIR}/nemo_Network.3
			${MAN_DIR}/nemo_Configuration.3
			${MAN_DIR}/nemo_Simulation.3
			# Extra crud
			${MAN_DIR}/nemo.hpp.3
		COMMAND ${DOXYGEN_EXECUTABLE}
		DEPENDS
			${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
			${CMAKE_SOURCE_DIR}/src/nemo.h
			${CMAKE_SOURCE_DIR}/src/nemo/Network.hpp
			${CMAKE_SOURCE_DIR}/src/nemo/Configuration.hpp
			${CMAKE_SOURCE_DIR}/src/nemo/Simulation.hpp
	)

	# Name C API man page correctly
	ADD_CUSTOM_COMMAND(
		OUTPUT ${MAN_DIR}/nemo-c.3
		COMMAND ${CMAKE_COMMAND} -E copy ${MAN_DIR}/nemo.h.3 ${MAN_DIR}/nemo-c.3
		DEPENDS ${MAN_DIR}/nemo.h.3
	)

	ADD_CUSTOM_TARGET(api-doc-man
		DEPENDS ${GENERATED_MAN_PAGES}
		COMMENT "Generating man pages for API"
		VERBATIM
	)

	FIND_PROGRAM(SED sed)
	FOREACH(MANPAGE ${GENERATED_MAN_PAGES})

		# Doxygen produces rather cruddy man-pages. We do some replacements to fix header etc.
		IF(SED)
			# TODO: get date from somewhere
			STRING(REGEX REPLACE "${MAN_DIR}/(.*).3" "\\1" MANPAGE_TITLE ${MANPAGE})
			SET(SED_SET_HEADER -e s/^\\.TH.*/.TH\ ${MANPAGE_TITLE}\ 3\ \"Mar\ 2010\"\ \"\"\ \"Nemo\ Reference\ Manual\"/)
			SET(SED_SET_AUTHOR -e s/Generated\ automatically\ by\ Doxygen.*/Andreas\ Fidjeland\ \(using\ Doxygen\)/)

			# The title page is incorrect as it referes to filename or
			# class. Just hard-code the relevant replacements.
			SET(SED_SET_TITLE_C -e s/^nemo.h\ \\\\-/${MANPAGE_TITLE}\ \\\\-/)
			SET(SED_SET_TITLE ${SED_SET_TITLE_C})

			# Add reference to main man-page and upper-case author header
			SET(SED_SET_SEE -e /.SH\ .Author/\ c.SH\ SEE\ ALSO\\nnemo\(3\)\ for\ library\ overview\\n.SH\ AUTHOR)

			ADD_CUSTOM_COMMAND(
				TARGET api-doc-man POST_BUILD
				COMMAND ${SED} -i ${SED_SET_HEADER} ${SED_SET_AUTHOR} ${SED_SET_TITLE} ${SED_SET_SEE} ${MANPAGE}
				VERBATIM
			)
		ENDIF(SED)

		# Copy final generated files into source directory, for inclusion in source package
		ADD_CUSTOM_COMMAND(
			TARGET api-doc-man POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy ${MANPAGE} ${CMAKE_SOURCE_DIR}/doc/man/man3
		)

	ENDFOREACH(MANPAGE ${GENERATED_MAN_PAGES})

ELSE(NEMO_DOC_MAN_REBUILD)

	# The generated documentation should already be present in the source tree.
	# Copy it into the build tree.
	ADD_CUSTOM_COMMAND(
		OUTPUT ${GENERATED_MAN_PAGES}
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/doc/man/man3 ${CMAKE_BINARY_DIR}/doc/man/man3
	)

	ADD_CUSTOM_TARGET(api-doc-man
		DEPENDS ${GENERATED_MAN_PAGES}
		COMMENT "Using existing man pages"
		VERBATIM
	)

ENDIF(NEMO_DOC_MAN_REBUILD)


# Always create API man pages
ADD_DEPENDENCIES(nemo api-doc-man)

IF(UNIX)
	SET(MAN_INSTALL_DIR share/man/man3)
	INSTALL(FILES ${GENERATED_MAN_PAGES} DESTINATION ${MAN_INSTALL_DIR})
	INSTALL(FILES man3/nemo.3 DESTINATION ${MAN_INSTALL_DIR})
ENDIF(UNIX)

