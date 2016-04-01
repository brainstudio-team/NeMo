/*! \file plugin.ipp OS-specific dynamic loading routines */

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! Initialise loading routines, returning success */
bool dl_init();

/*! Shut down loading routines, returning success */
bool dl_exit();

/*! Load library, returning handle to library. Returns NULL in case of failure */
dl_handle dl_load(const char* name);

/*! Unload library. Return success. */
bool dl_unload(dl_handle h);

/*! Return description of last error */
const char* dl_error();

/* Return function pointer to given symbol or NULL if there's an error. */
void* dl_sym(dl_handle, const char* name);

/*! \return library name with any required extension */
std::string dl_libname(std::string baseName);


#ifdef _MSC_VER

bool
dl_init()
{
	return true;
}

bool
dl_exit()
{
	return true;
}

dl_handle
dl_load(const char* name)
{
	return LoadLibrary(name);
}

bool
dl_unload(dl_handle h)
{
	return FreeLibrary(h) != 0;
}

const char*
dl_error()
{
	const char* str;
	FormatMessage(
		FORMAT_MESSAGE_FROM_SYSTEM
		| FORMAT_MESSAGE_ALLOCATE_BUFFER
		| FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL, GetLastError(), 0, (LPTSTR) &str, 0, NULL);
	return str;
}

void*
dl_sym(dl_handle hdl, const char* name)
{
	return GetProcAddress(hdl, name);
}

std::string
dl_libname(std::string baseName)
{
	return baseName.append(".dll");
}

#else

bool
dl_init()
{
	return lt_dlinit() == 0;
}

bool
dl_exit()
{
	return lt_dlexit() == 0;
}

dl_handle
dl_load(const char* name)
{	
#if __APPLE__
	/* Until recently (March 2011) LTDL tried to open '.so' on OSX rather than
	 * the more common '.dylib'. We're only ever loading our own libraries
	 * here, so force the extension */ 
	std::string fullname = std::string(name).append(".dylib");
	return lt_dlopen(fullname.c_str());
#else
	return lt_dlopenext(name);
#endif
}

bool
dl_unload(dl_handle h)
{
	return lt_dlclose(h) == 0;
}

const char*
dl_error()
{
	return lt_dlerror();
}

void*
dl_sym(dl_handle hdl, const char* name)
{
	return lt_dlsym(hdl, name);
}

std::string
dl_libname(std::string baseName)
{
	/* Leave libltdl to work out the extension */
	return std::string("lib").append(baseName);
}

#endif
