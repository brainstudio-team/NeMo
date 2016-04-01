#ifndef NEMO_MPI_MAPPER_HPP
#define NEMO_MPI_MAPPER_HPP

#include <utility>
#include <math.h>

#include <nemo/util.h>
#include <nemo/exception.hpp>
#include <nemo/internal_types.h>
#include "MpiTypes.hpp"
#include <boost/random.hpp>

namespace nemo {
namespace mpi {

typedef boost::mt19937 rng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_real<double> > urng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_int<> > uirng_t;

typedef unsigned mapperType;
typedef std::pair<nidx_t, nidx_t> nrange;
typedef std::vector<nrange> Ranges;
/* Translate between global neuron indices and rank indices
 *
 * Each neuron is processed on a single node. The index of a neuron can thus be
 * specified either in a global index or with a rank/local index pair.
 */
class Mapper {
public:

	Mapper();

	/*! Create a new mapper.
	 *
	 * \param neurons total number of neurons in the network
	 * \param world reference to MPI communicator
	 * \param net reference to the constructed network
	 * \param type number that indicates the type of partitioning
	 */
	Mapper(unsigned neurons, boost::mpi::communicator& world, const nemo::Network& net, mapperType type);

	void setRank(rank_t rank);

	/*! \return the rank of the process which should process a particular neuron */
	rank_t rankOf(nidx_t) const;

	bool isLocal(nidx_t n) const;

	/*! \return global number of neurons in simulation  */
	unsigned neuronCount() const;

	const std::vector<rank_t>& getRankMap() const;

	std::string mapperTypeDescr() const;
private:
	enum Partition {
		UNIFORM, NEWMAN, RANDOM
	};

	void makeRanges();

	unsigned getAvgNeurons();

	unsigned getNodesize(rank_t rank);

	void uniformPartition();

	void newmansPartition(const nemo::Network& net);

	void newmansPartition(const nemo::Network& net, std::vector<unsigned>& partition, std::vector<unsigned>& subPart1,
			std::vector<unsigned>& subPart2);

	void randomPartition();

	rank_t m_rank;

	unsigned m_workers;

	/* Global number of neurons in simulation */
	unsigned m_neuronsCount;

	Ranges m_ranges;

	mapperType m_type;

	std::vector<rank_t> m_rankMap;

	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & m_workers;
		ar & m_neuronsCount;
		ar & m_ranges;
		ar & m_type;
		ar & m_rankMap;
	}

};

} // end namespace mpi
} // end namespace nemo

BOOST_CLASS_IMPLEMENTATION(nemo::mpi::Mapper, object_serializable)
BOOST_CLASS_TRACKING(nemo::mpi::Mapper, track_never)

#endif
