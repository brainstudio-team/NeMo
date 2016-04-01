# TODO: move these into a separate CMake module

IF(WIN32)
	FIND_PROGRAM(MEX NAMES mex.bat)
	FIND_PROGRAM(MEXEXT NAMES mexext.bat)
ELSE(WIN32)
	FIND_PROGRAM(MEX NAMES matlab-mex mex)
	FIND_PROGRAM(MEXEXT NAMES mexext)
ENDIF(WIN32)


IF(NOT MEX STREQUAL MEX-NOTFOUND)

	# The input sources to MEX is mostly auto-generated. This is not part of
	# the main build process since user builds should not need to do this.

	IF(MEXEXT)
		# Set the MEX extension automatically
		EXECUTE_PROCESS(
			COMMAND ${MEXEXT}
			OUTPUT_VARIABLE SUFFIX
			OUTPUT_STRIP_TRAILING_WHITESPACE
		)
	ELSE(MEXEXT)
		# No mexext utility available so set extension by hand. Old versions of
		# Matlab lack the utility.
		IF(MEX_LIB_SUFFIX)
			# User has already set this by hand
			SET(SUFFIX ${MEX_LIB_SUFFIX})
		ELSE(MEX_LIB_SUFFIX)
			# Ask user to set the variable
			MESSAGE(SEND_ERROR "Could not find the mexext utility, required to build the Matlab bindings. Either set the MEX library extension on your platform by hand (in variable MEX_LIB_SUFFIX) or disable Matlab bindings altogether (set NEMO_MATLAB_ENABLED to OFF)")
			SET(MEX_LIB_SUFFIX "MEX_LIB_SUFFIX-NOTFOUND"
				CACHE
				STRING "Mex library extension on this platform (e.g. dll, mexa64, etc.)"
			)
		ENDIF(MEX_LIB_SUFFIX)
	ENDIF(MEXEXT)

	SET(MEX_INCLUDES
		-I${CMAKE_SOURCE_DIR}/src
		-I${CMAKE_BINARY_DIR}/src # for configuration file
		-I${Boost_INCLUDE_DIRS}
	)
	SET(MEX_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/nemo_mex.${SUFFIX})
	SET(MEX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/nemo_mex.cpp)
	SET(NEMO_LIB_DIR ${CMAKE_BINARY_DIR}/src/nemo/${CMAKE_CFG_INTDIR})
	SET(MEX_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

	FILE(MAKE_DIRECTORY ${MEX_OUTPUT_DIR})

	ADD_CUSTOM_COMMAND(
		OUTPUT ${MEX_OUTPUT}
		COMMAND ${MEX} ${MEX_INCLUDES} -L${NEMO_LIB_DIR} -lnemo ${MEX_SOURCE}
		WORKING_DIRECTORY ${MEX_OUTPUT_DIR}
		DEPENDS ${MEX_SOURCE}
	)

	ADD_CUSTOM_TARGET(mex-api ALL DEPENDS ${MEX_OUTPUT})

	ADD_DEPENDENCIES(mex-api nemo)

	# Sometimes the m-files changes and the install list along with it, but the
	# installed targets are deemed up-to-date. This might be because the list
	# of install targets is loaded from a file, but not sure why. Just force
	# all files to always be installed to work around this issue.
	SET(CMAKE_INSTALL_ALWAYS ON)

	IF(WIN32)
		SET(MATLAB_INSTALL_DIR Matlab)
	ELSE(WIN32)
		SET(MATLAB_INSTALL_DIR share/nemo/matlab)
	ENDIF(WIN32)

	FILE(STRINGS "install.cmake" AUTO_MATLAB_FILES)

	SET(DEPRECATED_MATLAB_FILES nemoGetDelays.m nemoGetPlastic.m nemoGetTargets.m nemoGetWeights.m)

	INSTALL(
		FILES ${MEX_OUTPUT} ${AUTO_MATLAB_FILES} ${DEPRECATED_MATLAB_FILES} nemo.m example.m
		DESTINATION ${MATLAB_INSTALL_DIR}
	)

ELSE(NOT MEX STREQUAL MEX-NOTFOUND)
	MESSAGE("MEX compiler not found. Skipping Matlab API generation") 
ENDIF(NOT MEX STREQUAL MEX-NOTFOUND)

SUBDIRS(kuramoto)
