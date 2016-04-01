#ifndef SIM_RUNNER_HPP
#define SIM_RUNNER_HPP

#include <ostream>
#include <boost/program_options.hpp>
#include <nemo.hpp>

/*! Run simulation and report performance results
 *
 * The run does a bit of warming up and does measures performance for some time
 * without reading data back.
 *
 * \param n
 * 		Number of neurons in the network
 * \param m
 * 		Number of synapses per neuron. This must be the same for each neuron in
 * 		order for the throughput-measurement to work out. This constraint is
 * 		not checked.
 *
 * \return milliseconds elapsed for the actual simulation (wallclock)
 */
long benchmark(nemo::Simulation* sim, unsigned n, unsigned m,
				boost::program_options::variables_map& opts,
				unsigned seconds=10);


/*! Run simulation for some time, writing data to output stream
 *
 * \param time_ms
 * 		Number of milliseconds simulation should be run
 * \param stdp
 * 		Period (in ms) between STDP applications. If 0, run without STDP.
 */
void
simulate(nemo::Simulation* sim, unsigned time_ms, unsigned stdp=0, std::ostream& out=std::cout);


/*! Run simulation for some time, writing data to output to file. Existing file
 * contents will be overwritten.
 *
 * \see simulate
 */
void
simulateToFile(nemo::Simulation* sim, unsigned time_ms, unsigned stdp, const char* firingFile);


/*! \return configuration with STDP, logging, and backend possibly set via
 * command-line parameters */
nemo::Configuration configuration(boost::program_options::variables_map& opts);


/* Return common program options */
boost::program_options::options_description
commonOptions();

typedef boost::program_options::variables_map vmap;

vmap
processOptions(int argc, char* argv[], const boost::program_options::options_description& desc);




#endif
