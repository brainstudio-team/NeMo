/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/* This is a top-level utility app for NeMo, which currently only prints
 * information about the library and the parts of the systems on which it can
 * run.
 */

#include <iostream>
#include <boost/program_options.hpp>

#include <nemo.hpp>
#include <nemo/config.h>


void
listCudaDevices()
{
#ifdef NEMO_CUDA_ENABLED
	unsigned dcount  = nemo::cudaDeviceCount();

	if(dcount == 0) {
		std::cout << "No CUDA devices available\n";
		return;
	}

	for(unsigned d = 0; d < dcount; ++d) {
		std::cout << d << ": " << nemo::cudaDeviceDescription(d) << std::endl;
	}
#else
	std::cout << "No CUDA devices available\n";
#endif
}



int
main(int argc, char* argv[])
{
	namespace po = boost::program_options;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "print this message")
		("version,v", "print version number")
		("list-devices,l", "print the available simulation devices")
		("system-plugin-path", "print the path to the system plugin path for NeMo")
		("user-plugin-path", "print the path to the user plugin path for NeMo")
	;

	po::variables_map vm;
	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
	} catch(boost::program_options::error& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		exit(1);
	}
	po::notify(vm);

	if(vm.count("help") != 0) {
		std::cout << "Usage:\n\t" << argv[0] << " [OPTIONS]\n\n";
		std::cout << desc << std::endl;
		exit(0);
	}

	if(vm.count("version") != 0) {
		std::cout << nemo::version() << std::endl;
		//! \todo list exact build as well
		//! \todo list build options as well
		exit(0);
	}

	if(vm.count("list-devices") != 0) {
		listCudaDevices();
		exit(0);
	}

	if(vm.count("system-plugin-path") != 0) {
		std::cout << NEMO_SYSTEM_PLUGIN_DIR << std::endl;
		exit(0);
	}

	if(vm.count("user-plugin-path") != 0) {
		std::cout << NEMO_USER_PLUGIN_DIR << std::endl;
		exit(0);
	}

	/* No options chosen, so return error */

	std::cout << "Usage:\n\t" << argv[0] << " [OPTIONS]\n\n";
	std::cout << desc << std::endl;
	return 1;
}
