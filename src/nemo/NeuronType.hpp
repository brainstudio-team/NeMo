#ifndef NEMO_NEURON_TYPE_HPP
#define NEMO_NEURON_TYPE_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <string>
#include <boost/filesystem.hpp>
#include <nemo/config.h>

#ifdef NEMO_MPI_ENABLED
#include <boost/serialization/serialization.hpp>
#include <boost/mpi/datatype.hpp>
#endif

namespace nemo {

/*! \brief General neuron type
 *
 * A neuron type is specified by:
 *
 * - the data it contains
 * - its dynamics 
 *
 * This class is concerned only with the type of data it contains. The
 * simulation data can be set up based on this, regardless of the neuron
 * dynamics. The neuron dynamics are specified in a plugin which is loaded when
 * the simulation is set up. The neuron format is described in a seperate
 * configuration file from which NeuronType instances are initialized.
 */
class NEMO_BASE_DLL_PUBLIC NeuronType
{
	public :

	/*! Initialise a neuron type from a neuron type description file, in
	 * .ini format located in one of NeMo's plugin directories */
	explicit NeuronType(const std::string& name);

	/* \return number of floating point parameters */
	size_t parameterCount() const {return m_nParam;}

	/* \return number of floating point state variables */
	size_t stateVarCount() const {return m_nState;}

	/*! Return the name of the neuron model
	 *
	 * This should match the name of the plugin which implements this
	 * neuron type */
	std::string name() const {return m_name;}

	/*! Return the index of the state variable representing the membrane potential */
	unsigned membranePotential() const {return m_membranePotential;}

	bool usesNormalRNG() const {return m_nrand;}

	/*! How much history (of the state) do we need? In a first order system
	 * only the latest state is available. In a second order system, a
	 * double buffer is used. The \it previous state is available to the
	 * whole system, whereas the current state is available to some subset
	 * of the system (e.g. a thread). */
	unsigned stateHistory() const {return m_stateHistory;}

	bool usesRcmSources() const {return m_rcmSources;}
	bool usesRcmDelays() const {return m_rcmDelays;}
	bool usesRcmForward() const {return m_rcmForward;}
	bool usesRcmWeights() const {return m_rcmWeights;}

	const boost::filesystem::path& pluginDir() const {return m_pluginDir;}

	private :

	size_t m_nParam;
	size_t m_nState;
	std::string m_name;

	unsigned m_membranePotential;

	/*! Does this neuron type require a per-neuron gaussian random number
	 * generator? */
	bool m_nrand;

	/* Fields of the reverse connectivity matrix */
	bool m_rcmSources;
	bool m_rcmDelays;
	bool m_rcmForward;
	bool m_rcmWeights;

	/*! How much history (of the state) do we need? In a first order system
	 * only the latest state is available. In a second order system, a
	 * double buffer is used. The \it previous state is available to the
	 * whole system, whereas the current state is available to some subset
	 * of the system (e.g. a thread). */
	unsigned m_stateHistory;

	/*! Directory where .ini file was found */
	boost::filesystem::path m_pluginDir;

	void parseConfigurationFile(const std::string& name);
};

}

#endif
