#ifndef MPIUTILS_HPP_
#define MPIUTILS_HPP_

#include <MpiTypes.hpp>

namespace logging = boost::log;
namespace triv = boost::log::trivial;
namespace attrs = boost::log::attributes;
namespace src = boost::log::sources;

typedef src::severity_logger<triv::severity_level> logger;

namespace nemo {
namespace mpi {

namespace utils {

void logInit(logger& lg, int rank, const char * logDir);

}
}
}
#endif /* MPIUTILS_HPP_ */
