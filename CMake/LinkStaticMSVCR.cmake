#
# When building with MSVC the resulting binaries and libraries ends up with
# runtime dependencies on the MSVC runtime library. Users may not have this
# installed. To avoid runtime link errors, call this macro to link statically
# against the runtime library.
#
MACRO(LINK_STATIC_MSVCR)
	IF(MSVC)
		FOREACH(flag_var
			CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
			CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
			IF(${flag_var} MATCHES "/MD")
				string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
			ENDIF(${flag_var} MATCHES "/MD")
		ENDFOREACH(flag_var)
	ENDIF(MSVC)
ENDMACRO(LINK_STATIC_MSVCR)
