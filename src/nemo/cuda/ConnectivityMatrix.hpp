#ifndef CONNECTIVITY_MATRIX_IMPL_HPP
#define CONNECTIVITY_MATRIX_IMPL_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stddef.h>
#include <cuda_runtime.h>

#include <map>
#include <vector>
#include <deque>

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>

#include <nemo/types.hpp>
#include <nemo/network/Generator.hpp>
#include <nemo/cuda/runtime/RCM.hpp>
#include <nemo/cuda/runtime/Delays.hpp>

#include "types.h"
#include "kernel.cu_h"
#include "Mapper.hpp"
#include "Outgoing.hpp"
#include "GlobalQueue.hpp"
#include "NVector.hpp"

struct param_t;

namespace nemo {

	class ConfigurationImpl;

	namespace cuda {

		class AxonTerminalAux;

		namespace construction {
			class FcmIndex;
		}

		namespace runtime {
			class Delays;
		}

/*! \brief Connectivity matrix
 *
 * The connectivity matrix (CM) specifies how neurons are connected. The CM has
 * both a forward version (connections from presynaptic to postsynaptic) and a
 * reverse version (connections from postsynaptic to presynaptic). The reverse
 * version may be required if synapses are modified at run time.
 *
 * Functions are prefixed 'f' or 'r' depending on which version it affects.
 * Furthermore, functions are prefixed 'd' or 'h' depending on whether it
 * affects data on the device or on the host.
 *
 * \section fcm Forward connectivity
 *
 * In the forward connectivity matrix (FCM) each synapse is represented by two
 * pieces of data: the synapse weight and the target neuron (within a
 * partition). Other per-synapse data, such as source partition/neuron, target
 * partition, and delay, are implicit in the storage, as it is used as a key.
 *
 * The FCM can get very large and make up a major source of memory traffic in
 * the kernel. To facilitate coalesced access to this data we perform some
 * grouping of synapses. A \e synapse \e group consists of all the synapses
 * sharing the same source partition, source neuron, target partition, and
 * delay. All synapses in such a group must be delivered during the same
 * simulation step between the same pair of partitions. Synapse groups are
 * further split into \e synapse \e warps which are simply groupings of \ref
 * WARP_SIZE synapses. The synapse warp is the basic unit of spike delivery.
 *
 * On the device the FCM is stored as an \e s x \ref WARP_SIZE matrix, in
 * a contigous chunk of memory. The value of \e s depends on the connectivity
 * in the network. Each synapse warp thus has a unique index which is simply
 * the offset from the start of the matrix. These indices are stored in the
 * \ref nemo::cuda::Outgoing outgoing data.
 *
 * The size of each synapse groups is not necessarily a multiple of \ref
 * WARP_SIZE so some memory is wasted on padding. The amount of padding varies
 * with the network connectivity, but generally a high degree of clustering
 * (spatially and/or temporally) will give a low amount of padding.
 *
 * \section rcm Reverse connectivity
 *
 * The reverse connectivity matrix (RCM) stores the target synapse and an
 * accumulator variable (for STDP) for each synapse. The data are organised in
 * rows indexed by neuron. To avoid excessive padding due to potentially
 * variable-width rows, the RCM is split into separate chunks for each
 * partition, each with its own pitch. This leads to a rather inelegant method
 * for indexing the RCM. The organisation of the RCM will be changed in future
 * versions.
 *
 * \todo Move reverse and forward connectivity into separate classes
 */
class ConnectivityMatrix
{
	public:

		ConnectivityMatrix(
				const nemo::network::Generator&,
				const nemo::ConfigurationImpl&,
				const Mapper&);

		delay_t maxDelay() const { return m_maxDelay; }

		/*! \copydoc nemo::Simulation::getSynapsesFrom */
		const std::vector<synapse_id>& getSynapsesFrom(unsigned neuron);

		/*! \copydoc nemo::Simulation::getSynapseTarget */
		unsigned getTarget(const synapse_id& synapse) const;

		/*! \copydoc nemo::Simulation::getSynapseDelay */
		unsigned getDelay(const synapse_id& synapse) const;

		/*! \return synapse weight
		 *
		 * Calling function may involve a large copy from the device to host.
		 */
		float getWeight(cycle_t cycle, const synapse_id& synapse) const;

		/*! \copydoc nemo::Simulation::getSynapsePlastic */
		unsigned char getPlastic(const synapse_id& synapse) const;

		/*! Clear one plane of connectivity matrix on the device */
		void clearStdpAccumulator();

		size_t d_allocated() const;

		synapse_t* d_fcm() const { return md_fcm.get(); }

		/*! \return pointer to device data containing outgoing spike data for
		 * each neuron */
		outgoing_t* d_outgoing() const { return m_outgoing.d_data(); }

		/*! \return pointer to device data containing the number of outgoing
		 * spike groups for each neuron */
		outgoing_addr_t* d_outgoingAddr() const { return m_outgoing.d_addr(); }

		/*! \copydoc nemo::cuda::GlobalQueue::d_data */
		gq_entry_t* d_gqData() const { return m_gq.d_data(); }

		/*! \copydoc nemo::cuda::GlobalQueue::d_fill */
		unsigned* d_gqFill() const { return m_gq.d_fill(); }

		/*! \return number of fractional bits used for weights. */
		unsigned fractionalBits() const { return m_fractionalBits; }

		void printMemoryUsage(std::ostream&) const;

		delay_dt* d_ndData() const { return md_delays->d_data(); }

		unsigned* d_ndFill() const { return md_delays->d_fill(); }

		/*! Fill in all relevant fields in global parameters data structure */
		void setParameters(param_t*) const;

		/*! \return RCM device pointers */
		rcm_dt* d_rcm() { return md_rcm.d_rcm(); }

	private:

		const Mapper& m_mapper;

		delay_t m_maxDelay;

		/*! Compact forward connectivity matrix on device */
		boost::shared_ptr<synapse_t> md_fcm;

		/*! Compact reverse connectivity matrix on device */
		runtime::RCM md_rcm;

		/*! Host-side copy of the weight data. This is mutable since it acts as
		 * a buffer for synapse getters */
		mutable std::vector<weight_dt> mhf_weights;

		/* \post The weight of every synapse in 'synapses' is found up-to-date
		 * in mhf_weights. */
		const std::vector<weight_dt>& syncWeights(cycle_t) const;
		mutable cycle_t m_lastWeightSync;

		size_t md_fcmPlaneSize; // in words
		size_t md_fcmAllocated; // in bytes

		void moveFcmToDevice(size_t totalWarps,
				const std::vector<synapse_t>& h_targets,
				const std::vector<weight_dt>& h_weights);

		/*! For each neuron, record the delays for which there are /any/
		 * outgoing connections */
		boost::scoped_ptr<runtime::Delays> md_delays;

		/* For spike delivery we need to keep track of all target partitions
		 * for each neuron */
		Outgoing m_outgoing;

		/* We also need device memory for the global queue */
		GlobalQueue m_gq;

		unsigned m_fractionalBits;

		/* Additional synapse data which is only needed for runtime queries.
		 * Static FCM data for each neuron, required for synapse queries.
		 * Neuron indices are global rather than the partition/neuron scheme
		 * used on the device, so no decoding needs to take place at run-time.
		 */
		typedef std::deque<AxonTerminalAux> aux_row;
		typedef boost::unordered_map<nidx_t, aux_row> aux_map;
		aux_map m_cmAux;

		/* The data required for reading back synapses is quite costly, since
		 * it is essentially a full copy of the FCM. In cases where this data
		 * is not needed it therefore makes sense to never construct this data
		 * in the first place. This can be set via a configuration option*/
		bool m_writeOnlySynapses;

		void addAuxillary(const Synapse&, size_t addr);

#ifndef NDEBUG
		/* Count synapses to verify that m_cmAux contains dense rows */
		std::map<nidx_t, unsigned> m_synapsesPerNeuron;
#endif

		/*! Internal buffer for synapse queries */
		std::vector<synapse_id> m_queriedSynapseIds;

		/*! Add synapse to forward matrix
		 *
		 * \return synapse address, i.e. full word offset into FCM
		 */
		size_t addForward(
				const Synapse&,
				const DeviceIdx& source,
				const DeviceIdx& target,
				size_t& nextFreeWarp,
				construction::FcmIndex&,
				std::vector<synapse_t>& h_targets,
				std::vector<weight_dt>& h_weights);

		void setMaxDelay(const Synapse& s);

		void verifySynapseTerminals(const aux_map&, const Mapper& mapper);

		const AxonTerminalAux& axonTerminalAux(const synapse_id& id) const;
};



/* The parts of the synapse data is only needed if querying synapses at
 * run-time. This data is stored separately */
class AxonTerminalAux
{
	public :

		unsigned target() const { return m_target; }
		unsigned delay() const { return m_delay; }
		unsigned char plastic() const { return (unsigned char) m_plastic; }
		size_t addr() const { return m_addr; }

		AxonTerminalAux(const Synapse& s, size_t addr) :
			m_target(s.target()), m_delay(s.delay), m_plastic(s.plastic() != 0), m_addr(addr) { }

		AxonTerminalAux() :
			m_target(~0), m_delay(~0), m_plastic(false), m_addr(~0) { }

	private :

		/* Global target index */
		unsigned m_target;

		unsigned m_delay;
		bool m_plastic;

		/* Address into FCM on device */
		size_t m_addr;

};



	} // end namespace cuda
} // end namespace nemo

#endif
