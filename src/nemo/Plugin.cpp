/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <cstdlib>
#include <boost/format.hpp>

#include <nemo/config.h>
#include "Plugin.hpp"
#include "exception.hpp"

#include "plugin.ipp"

#ifdef _MSC_VER
const char* HOME_ENV_VAR = "userprofile";
#else
const char* HOME_ENV_VAR = "HOME";
#endif

namespace nemo {


std::set<boost::filesystem::path> Plugin::s_extraPaths;


Plugin::Plugin(const std::string& name) :
	m_handle(NULL)
{
	init(name);
	try {
		load(name);
	} catch(nemo::exception&) {
		dl_exit();
		throw;
	}
}



Plugin::Plugin(const boost::filesystem::path& dir, const std::string& name) :
	m_handle(NULL)
{
	init(name);
	try {
		load(dir, name);
	} catch(nemo::exception&) {
		dl_exit();
		throw;
	}
}



Plugin::~Plugin()
{
	/* Both the 'unload' and the 'exit' can fail. There's not much we can do
	 * about either situation, so just continue on our merry way */
	dl_unload(m_handle);
	dl_exit();
}



void
Plugin::init(const std::string& name)
{
	using boost::format;
	if(!dl_init()) {
		/* Nothing to clean up in case of failure here */
		throw nemo::exception(NEMO_DL_ERROR,
				str(format("Error when loading plugin %s: %s")
					% dl_libname(name) % dl_error()));
	}
}



boost::filesystem::path
Plugin::userDirectory()
{
	char* home = getenv(HOME_ENV_VAR);
	if(home == NULL) {
		throw nemo::exception(NEMO_DL_ERROR,
				"Could not locate user's home directory when searching for plugins");
	}

	return boost::filesystem::path(home) / NEMO_USER_PLUGIN_DIR;
}



boost::filesystem::path
Plugin::systemDirectory()
{
	using boost::format;
	using namespace boost::filesystem;

#if defined _WIN32 || defined __CYGWIN__
	/* On Windows, where there aren't any standard library locations, NeMo
	 * might be relocated. To support this, look for a plugin directory relative
	 * to the library location path rather than relative to the hard-coded
	 * installation prefix.  */
	HMODULE dll = GetModuleHandle("nemo_base.dll");
	TCHAR dllPath[MAX_PATH];
	GetModuleFileName(dll, dllPath, MAX_PATH);
	path systemPath = path(dllPath).parent_path().parent_path() / NEMO_SYSTEM_PLUGIN_DIR;
#else
	path systemPath(NEMO_SYSTEM_PLUGIN_DIR);
#endif
	if(!exists(systemPath)) {
		throw nemo::exception(NEMO_DL_ERROR,
				str(format("System plugin path does not exist: %s") % systemPath));
	}
	return systemPath;
}



Plugin::path_iterator
Plugin::extraPaths_begin()
{
	return s_extraPaths.begin();
}



Plugin::path_iterator
Plugin::extraPaths_end()
{
	return s_extraPaths.end();
}




void
Plugin::load(const std::string& name)
{
	using boost::format;
	m_handle = dl_load(dl_libname(name).c_str());
	if(m_handle == NULL) {
		throw nemo::exception(NEMO_DL_ERROR,
				str(format("Error when loading plugin %s: %s")
					% dl_libname(name) % dl_error()));
	}
}



void
Plugin::load(const boost::filesystem::path& dir, const std::string& name)
{
	using boost::format;
	using namespace boost::filesystem;

	path fullpath = dir / path(dl_libname(name));
	m_handle = dl_load(fullpath.string().c_str());
	if(m_handle == NULL) {
		throw nemo::exception(NEMO_DL_ERROR,
				str(format("Error when loading plugin %s: %s")
					% dl_libname(name) % dl_error()));
	}
}



void*
Plugin::function(const std::string& name) const
{
	void* fn = dl_sym(m_handle, name.c_str());
	if(fn == NULL) {
		throw nemo::exception(NEMO_DL_ERROR, dl_error());
	}
	return fn;
}



void
Plugin::addPath(const std::string& dir)
{
	using boost::format;

	boost::filesystem::path path(dir);
	if(!exists(path) && !is_directory(path)) {
		throw nemo::exception(NEMO_DL_ERROR,
				str(format("User-specified path %s could not be found") % dir));
	}
	s_extraPaths.insert(path);
}


}
