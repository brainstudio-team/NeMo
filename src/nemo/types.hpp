#ifndef NEMO_TYPES_HPP
#define NEMO_TYPES_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stddef.h>

/* The basic types in nemo_types are also used without an enclosing namespace
 * inside the kernel (which is pure C). */
#include <nemo/config.h>
#include "internal_types.h"
#include "Neuron.hpp"

#ifdef NEMO_MPI_ENABLED
#include <boost/serialization/utility.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/mpi/datatype.hpp>

namespace nemo {
namespace mpi {
typedef int rank_t;
}
}
#endif


#ifdef NEMO_MPI_ENABLED
namespace boost {
	namespace serialization {
		class access;
	}
}
#endif

namespace nemo {


struct AxonTerminal
{
	public :
		id32_t id;
		nidx_t target;
		float weight;
		bool plastic;

		AxonTerminal():
			id(~0), target(~0), weight(0.0f), plastic(false) { }

		AxonTerminal(id32_t id, nidx_t t, float w, bool p):
			id(id), target(t), weight(w), plastic(p) { }

	private :
#ifdef NEMO_MPI_ENABLED
		friend class boost::serialization::access;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & id;
			ar & target;
			ar & weight;
			ar & plastic;
		}
#endif
};



class Synapse
{
	public :

		Synapse() : source(0), delay(0) {}

		Synapse(nidx_t source, delay_t delay, const AxonTerminal& terminal) :
			source(source), delay(delay), terminal(terminal) { }

		nidx_t source;
		delay_t delay;
		AxonTerminal terminal;

		id32_t id() const { return terminal.id; }

		nidx_t target() const { return terminal.target; }

		unsigned char plastic() const { return terminal.plastic; }

		float weight() const { return terminal.weight; }

	private :

#ifdef NEMO_MPI_ENABLED
		friend class boost::serialization::access;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & source;
			ar & delay;
			ar & terminal;
		}
#endif
};



class RSynapse
{
	public :

		nidx_t source;
		unsigned delay;

		RSynapse(nidx_t s, unsigned d) : source(s), delay(d) { }
};



struct SynapseAddress
{
	size_t row;
	sidx_t synapse;

	SynapseAddress(size_t row, sidx_t synapse):
		row(row), synapse(synapse) { }

	SynapseAddress():
		row(~0), synapse(~0) { }
};


} // end namespace nemo


#ifdef NEMO_MPI_ENABLED
BOOST_IS_MPI_DATATYPE(nemo::NeuronType);
BOOST_CLASS_IMPLEMENTATION(nemo::NeuronType, object_serializable)
BOOST_CLASS_TRACKING(nemo::NeuronType, track_never)

BOOST_IS_MPI_DATATYPE(nemo::AxonTerminal);
BOOST_CLASS_IMPLEMENTATION(nemo::AxonTerminal, object_serializable)
BOOST_CLASS_TRACKING(nemo::AxonTerminal, track_never)

BOOST_IS_MPI_DATATYPE(nemo::Synapse);
BOOST_CLASS_IMPLEMENTATION(nemo::Synapse, object_serializable)
BOOST_CLASS_TRACKING(nemo::Synapse, track_never)

//not fixed length BOOST_IS_MPI_DATATYPE(nemo::Neuron);
BOOST_CLASS_IMPLEMENTATION(nemo::Neuron, object_serializable)
BOOST_CLASS_TRACKING(nemo::Neuron, track_never)
#endif


#endif
