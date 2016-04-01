CMAKE_MINIMUM_REQUIRED(VERSION 2.8.1)

PROJECT(NeMo)

IF(UNIX AND NOT APPLE)
	SET(CPACK_PACKAGE_NAME nemo)
ENDIF(UNIX AND NOT APPLE)



SET(CPACK_PACKAGE_VERSION_MAJOR 0)
SET(CPACK_PACKAGE_VERSION_MINOR 7)
SET(CPACK_PACKAGE_VERSION_PATCH 2)
SET(CPACK_PACKAGE_CONTACT "Andreas Fidjeland <andreas.fidjeland@imperial.ac.uk>")
SET(CPACK_PACKAGE_DESCRIPTION_FILE ${CMAKE_SOURCE_DIR}/description.txt)

SET(NEMO_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")

SUBDIRS(src doc)

# read description from file
FILE(READ ${CPACK_PACKAGE_DESCRIPTION_FILE} NEMO_DESCRIPTION)

IF(APPLE OR WIN32)
	# PackageMaker cannot handle files without an extension.
	# Windows won't fall over, but probably a good idea there as well.
	SET(README "ReadMe.txt")
	SET(COPYING "Copying.txt")
	SET(INSTALLATION "Install.txt")
ELSE(APPLE OR WIN32)
	SET(README "README")
	SET(COPYING "COPYING")
	SET(INSTALLATION "INSTALL")
ENDIF(APPLE OR WIN32)

CONFIGURE_FILE(README.in ${CMAKE_BINARY_DIR}/${README} ESCAPE_QUOTES @ONLY)
CONFIGURE_FILE(README.in ${CMAKE_SOURCE_DIR}/${README} ESCAPE_QUOTES @ONLY)
CONFIGURE_FILE(COPYING ${CMAKE_BINARY_DIR}/${COPYING} ESCAPE_QUOTES @ONLY)
CONFIGURE_FILE(INSTALL ${CMAKE_BINARY_DIR}/${INSTALLATION} ESCAPE_QUOTES @ONLY)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

# Packaging

SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_BINARY_DIR}/${COPYING}")
SET(CPACK_RESOURCE_FILE_README "${CMAKE_BINARY_DIR}/${README}")

IF(WIN32)
	SET(CPACK_GENERATOR "NSIS")
	# Not that for the path setting to work, the libraries *must* be in the bin
	# folder, as CMake is hard-coded this way.
	SET(CPACK_NSIS_MODIFY_PATH ON)
	# By default CMake puts the installed version into a directory which
	# includes the version in the name. We can reasonably assume that users
	# won't have more than a single version installed at a time, so always
	# install into the same directory, to make upgrades easier.
	SET(CPACK_PACKAGE_INSTALL_DIRECTORY "NeMo")
	# TODO: Could add links to manual here using CPACK_NSIS_MENU_LINKS. Did not immideatly work
ENDIF(WIN32)

IF(APPLE)
	SET(CPACK_GENERATOR "PackageMaker")
ENDIF(APPLE)

SET(CPACK_SOURCE_GENERATOR "TGZ" "ZIP")

SET(CPACK_SOURCE_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${NEMO_VERSION}")

# By default cpack grabs everything in the source tree. However, the local
# source tree may contain user-specific files which should not be included.
# A list of such files (or patterns) can be specified in the file
# CMake/cpack.ignore
IF(EXISTS CMake/cpack.ignore)
	FILE(READ CMake/cpack.ignore USER_CPACK_SOURCE_IGNORE_FILES)
	SET(CPACK_SOURCE_IGNORE_FILES ${USER_CPACK_SOURCE_IGNORE_FILES} ${CPACK_SOURCE_IGNORE_FILES})
ENDIF(EXISTS CMake/cpack.ignore)

SET(CPACK_SOURCE_IGNORE_FILES
	"\\\\.gitignore"
	".git"
	build
	src/mpi
	src/api/autogen
	src/api/matlab/sources
	src/api/python/utils
	src/gui
	doc/manual/src
	doc/manual/CMakeLists.txt
	ReadMe.txt # for PackageMaker only
	${CPACK_SOURCE_IGNORE_FILES})

INCLUDE(CPack)
