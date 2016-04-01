#ifndef NEMO_PLUGIN_HPP
#define NEMO_PLUGIN_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#ifdef _MSC_VER
/* Suppress generation of MSVC-specific min/max macros which otherwise break
 * std::min and std::max */
#	define NOMINMAX
#	include <windows.h>
typedef HMODULE dl_handle;
#else
#	include <ltdl.h>
typedef lt_dlhandle dl_handle;
#endif
#include <set>
#include <string>
#include <boost/utility.hpp>
#include <boost/filesystem.hpp>
#include <nemo/config.h>


namespace nemo {


/* Wrapper for a dynamically loaded library or plugin */
class NEMO_BASE_DLL_PUBLIC Plugin : private boost::noncopyable
{
	public :

		/*! Load a plugin from the default library path.
		 *
		 * \param name
		 * 		base name of the library, i.e. without any system-specific
		 * 		prefix or file extension. For example the library libfoo.so on
		 * 		a UNIX system has the  base name 'foo'. 
		 *
		 * \throws nemo::exception for load errors
		 */
		explicit Plugin(const std::string& name);

		/*! Load a plugin from a subdirectory of the set of NeMo-specific plugin directories
		 *
		 * Plugins are always located in a subdirectory, as they are backend-specific.
		 * There is one system plugin directory and one per-user system directory.
		 */
		Plugin(const boost::filesystem::path& dir, const std::string& name);

		~Plugin();

		/*! \return function pointer for a named function
		 *
		 * The user needs to cast this to the appropriate type.
		 *
		 * \throws nemo::exception for load errors
		 */
		void* function(const std::string& name) const;

		/*! \return path to user plugin directory
		 *
		 * The path may not exist
		 */
		static boost::filesystem::path userDirectory();

		/*! \return path to system plugin directory
		 *
		 * \throws if the directory does not exist
		 */
		static boost::filesystem::path systemDirectory();

		/*! Add a directory to the NeMo plugin search path
		 *
		 * \param dir name of a directory containing NeMo plugins
		 *
		 * Paths added manually are searched before the default user and system
		 * paths. If multiple paths are added, the most recently added path is
		 * searched first.
		 *
		 * \throws nemo::exception if the directory is not found
		 */
		static void addPath(const std::string& dir);

		typedef std::set<boost::filesystem::path> path_collection;
		typedef path_collection::const_iterator path_iterator;

		static path_iterator extraPaths_begin();
		static path_iterator extraPaths_end();

	private:

		dl_handle m_handle;

		/*! Initialise the loader */
		void init(const std::string& name);

		/*! Load the library, using standard search paths */
		void load(const std::string& name);

		/*! Load the library based on absolute path */
		void load(const boost::filesystem::path& dir, const std::string& name);

		/*! Additional paths where to look for plugins */
		static std::set<boost::filesystem::path> s_extraPaths;
};

}

#endif
