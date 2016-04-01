#include "mpiUtils.hpp"

namespace nemo {
namespace mpi {

namespace utils {

void logInit(logger& lg, int rank, const char * logDir) {
	char nodir[] = "logConsole";
	if ( strcmp(logDir, nodir) != 0 ) {
		std::ostringstream oss;
		oss << logDir << "/process_" << rank << ".log";
		boost::log::add_file_log(boost::log::keywords::file_name = oss.str(), boost::log::keywords::auto_flush = true);
	}

	boost::log::add_common_attributes();
	lg.add_attribute("Rank", attrs::make_constant(rank));
}

}
}
}

