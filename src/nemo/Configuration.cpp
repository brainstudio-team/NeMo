#include "Configuration.hpp"
#include "ConfigurationImpl.hpp"

#include <boost/format.hpp>
#include <nemo/internals.hpp>
#include <nemo/cpu/Simulation.hpp>
#include <iostream>
namespace nemo {
#ifdef NEMO_MPI_ENABLED
using boost::property_tree::ptree;
#endif

Configuration::Configuration() :
		m_impl(new ConfigurationImpl()) {
	setDefaultHardware (*m_impl);
	setBackendDescription();
}

#ifdef NEMO_MPI_ENABLED

const ptree& empty_ptree() {
	static ptree t;
	return t;
}

Configuration::Configuration(const nemo::mpi::rank_t rank, const char * filename) :
m_impl(new ConfigurationImpl()) {
	ptree pt;
	try
	{
		read_xml(filename, pt);
	}
	catch(std::exception const& ex)
	{
		std::cout << "Warning: Invalid path to configuration XML. Default configuration is used for node " << rank << "." << std::endl;
		setDefaultHardware (*m_impl);
		setBackendDescription();
		return;
	}

	//in case configuration does not provide information about this node.
	bool setupOK = false;
	const ptree & nodes = pt.get_child("config.nodes", empty_ptree());

	BOOST_FOREACH(const ptree::value_type& node, nodes) {
		if (node.first == "node") {
			nemo::mpi::node_config nc;

			const ptree& nodeTree = node.second;

			nc.rank = nodeTree.get<unsigned>("rank");
			nc.logging = nodeTree.get<bool>("logging");

			const ptree& backendTree = nodeTree.get_child("backend", empty_ptree());
			BOOST_FOREACH(const ptree::value_type& node, nodes) {
				nc.bc.type = backendTree.get<std::string>("type");
				nc.bc.description = nodeTree.get_optional<std::string>("description");
			}

			nc.writeOnlySynapses = nodeTree.get_optional<bool>("writeOnlySynapses");

			if(nc.rank == rank) {
				setParameters(nc);
				setupOK = true;
			}
		}
	}

	if(!setupOK)
	{
		std::cout << "Warning: Invalid path to configuration XML. Default configuration is used for node " << rank << "." << std::endl;
		setDefaultHardware (*m_impl);
		setBackendDescription();
		return;
	}
}

void Configuration::setParameters(const nemo::mpi::node_config& nc) {
	if(nc.logging)
	enableLogging();
	else
	disableLogging();

	if(nc.bc.type.compare("cpu") == 0) {
		setCpuBackend();
	}
	else if(nc.bc.type.compare("cuda") == 0) {
		setCudaBackend();

		if(nc.bc.device)
		setCudaBackend(*nc.bc.device);

		if(nc.bc.partitionSize) {
			setCudaPartitionSize(*nc.bc.partitionSize);
		}
	}
	else {
		throw nemo::exception(NEMO_INVALID_INPUT, "Configuration backend type is wrong.");
	}

	setBackendDescription();

	if(nc.writeOnlySynapses) {
		if(*nc.writeOnlySynapses) {
			setWriteOnlySynapses();
		}
	}
}
#endif

Configuration::Configuration(const Configuration& other) :
		m_impl(new ConfigurationImpl(*other.m_impl)) {
	;
}

Configuration::Configuration(const ConfigurationImpl& other, bool ignoreBackendOptions) :
		m_impl(new ConfigurationImpl(other)) {
	if (ignoreBackendOptions) {
		setDefaultHardware (*m_impl);
		setBackendDescription();
	}
}

Configuration::~Configuration() {
	delete m_impl;
}

void Configuration::enableLogging() {
	m_impl->enableLogging();
}

void Configuration::disableLogging() {
	m_impl->disableLogging();
}

bool Configuration::loggingEnabled() const {
	return m_impl->loggingEnabled();
}

void Configuration::setCudaPartitionSize(unsigned ps) {
	m_impl->setCudaPartitionSize(ps);
}

unsigned Configuration::cudaPartitionSize() const {
	return m_impl->cudaPartitionSize();
}

void Configuration::setStdpFunction(const std::vector<float>& prefire,
		const std::vector<float>& postfire, float minWeight, float maxWeight) {
	setStdpFunction(prefire, postfire, 0, maxWeight, 0, minWeight);
}

void Configuration::setStdpFunction(const std::vector<float>& prefire,
		const std::vector<float>& postfire, float minE, float maxE, float minI, float maxI) {
	m_impl->setStdpFunction(prefire, postfire, minE, maxE, minI, maxI);
}

void Configuration::setStdpEnabled(bool isEnabled) {
	m_impl->setStdpEnabled(isEnabled);
}

void Configuration::setStdpPeriod(unsigned period) {
	m_impl->setStdpPeriod(period);
}

void Configuration::setStdpReward(float reward) {
	m_impl->setStdpReward(reward);
}

/* Utility method that setups the STDP params and enables the STDP */
void Configuration::setStdpParams(unsigned period, float reward) {
	m_impl->setStdpEnabled(true);
	m_impl->setStdpPeriod(period);
	m_impl->setStdpReward(reward);
}

void Configuration::setWriteOnlySynapses() {
	m_impl->setWriteOnlySynapses();
}

bool Configuration::writeOnlySynapses() const {
	return m_impl->writeOnlySynapses();
}

unsigned Configuration::stdpPeriod() const {
	return m_impl->stdpPeriod();
}

float Configuration::stdpReward() const {
	return m_impl->stdpReward();
}

bool Configuration::stdpEnabled() const {
	return m_impl->stdpEnabled();
}

void Configuration::setCpuBackend() {
	m_impl->setBackend(NEMO_BACKEND_CPU);
	setBackendDescription();
}

void Configuration::setCudaBackend(int device) {
	setCudaDeviceConfiguration(*m_impl, device);
	setBackendDescription();
}

backend_t Configuration::backend() const {
	return m_impl->backend();
}

int Configuration::cudaDevice() const {
	if (m_impl->backend() == NEMO_BACKEND_CUDA) {
		return m_impl->cudaDevice();
	} else {
		return -1;
	}
}

const char*
Configuration::backendDescription() const {
	return m_impl->backendDescription();
}

void Configuration::setBackendDescription() {
	switch (m_impl->backend()) {
	case NEMO_BACKEND_CUDA:
		m_impl->setBackendDescription(cudaDeviceDescription(m_impl->cudaDevice()));
		break;
	case NEMO_BACKEND_CPU:
		m_impl->setBackendDescription(cpu::deviceDescription());
		break;
	default:
		throw std::runtime_error("Invalid backend selected");
	}
}

} // end namespace nemo

std::ostream& operator<<(std::ostream& o, nemo::Configuration const& conf)
{
	return o << *conf.m_impl;
}

