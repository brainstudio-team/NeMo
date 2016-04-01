/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/program_options.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/format.hpp>

#include <nemo/config.h>

#include "NeuronType.hpp"
#include "exception.hpp"
#include "Plugin.hpp"
#include<string>
#include <vector>


namespace nemo {


NeuronType::NeuronType(const std::string& name) :
	m_nParam(0), m_nState(0),
	m_name(name), m_membranePotential(0),
	m_nrand(false),
	m_rcmSources(false),
	m_rcmDelays(false),
	m_rcmForward(false),
	m_rcmWeights(false),
	m_stateHistory(1)
{
	parseConfigurationFile(name);
}


/*! Return full name of .ini file for the given plugin
 *
 * Files are searched in both the user and the system plugin directories, in
 * that order, returning the first match.
 *
 * \throws nemo::exception if no plugin configuration file is found
 */
boost::filesystem::path
configurationFile(const std::string& name)
{
	using boost::format;
	using namespace boost::filesystem;

	for(Plugin::path_iterator i = Plugin::extraPaths_begin(); i != Plugin::extraPaths_end(); ++i ) {
		path extraPath = *i / (name + ".ini");
		if(exists(extraPath)) {
			return extraPath;
		}
	}

	path userPath = Plugin::userDirectory() / (name + ".ini");
	if(exists(userPath)) {
		return userPath;
	}

	path systemPath = Plugin::systemDirectory() / (name + ".ini");
	if(exists(systemPath)) {
		return systemPath;
	}

	throw nemo::exception(NEMO_INVALID_INPUT,
			str(format("Could not find .ini file for plugin %s") % name));
}


template<typename T>
T
getRequired(boost::program_options::variables_map vm,
		const std::string& name,
		boost::filesystem::path file)
{
	using boost::format;

	if(vm.count(name) != 1) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Missing parameter '%s' in configuration file %s")
					% name % file));
	}
	return vm[name].as<T>();
}



void
NeuronType::parseConfigurationFile(const std::string& name)
{
	using boost::format;
	namespace po = boost::program_options;
	namespace fs = boost::filesystem;

	po::options_description desc("Allowed options");
	desc.add_options()
		/* required fields, no defaults */
		("parameters", po::value<unsigned>(),
			"number of neuron parameters")
		("state-variables", po::value<unsigned>(),
			"number of neuron state variables")
		("membrane-potential", po::value<unsigned>(),
			"index of membrane potential variable")
		("rng.normal", po::value<bool>(),
			"is normal RNG required?")
		/* optional fields */
		("rcm.sources", po::value<bool>()->default_value(false),
			"are sources required in the reverse connectivity matrix")
		("rcm.delays", po::value<bool>()->default_value(false),
			"are delays required in the reverse connectivity matrix")
		("rcm.forward", po::value<bool>()->default_value(false),
			"are links to synapses in the forward matrix required in the reverse connectivity matrix")
		("rcm.weights", po::value<bool>()->default_value(false),
			"are weights required in the reverse connectivity matrix")
		("history", po::value<unsigned>()->default_value(1),
			"index of membrane potential variable")
		("backends.cpu", po::value<bool>()->default_value(false),
			"support for CPU backend")
		("backends.cuda", po::value<bool>()->default_value(false),
			"support for CUDA backend")
	;

	fs::path filename = configurationFile(name);

	m_pluginDir = filename.parent_path();

	fs::fstream file(filename, std::ios::in);
	if(!file.is_open()) {
		throw nemo::exception(NEMO_IO_ERROR,
				str(format("Failed to open neuron model configuration file %s: %s")
					% filename % strerror(errno)));
	}

	try {
		po::variables_map vm;
		po::store(po::parse_config_file(file, desc, true), vm);
		po::notify(vm);

		m_nParam = getRequired<unsigned>(vm, "parameters", filename);
		m_nState = getRequired<unsigned>(vm, "state-variables", filename);
		m_membranePotential = getRequired<unsigned>(vm, "membrane-potential", filename);
		m_nrand = getRequired<bool>(vm, "rng.normal", filename);
		m_rcmSources = vm["rcm.sources"].as<bool>();
		m_rcmDelays = vm["rcm.delays"].as<bool>();
		m_rcmForward = vm["rcm.forward"].as<bool>();
		m_rcmWeights = vm["rcm.weights"].as<bool>();
		m_stateHistory = vm["history"].as<unsigned>();
	} catch (po::error& e) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Error parsing neuron model configuration file %s: %s")
					% filename % e.what()));
	}
}


}
