#include "Mapper.hpp"
#include <iostream>
#include <algorithm>

#define THRESHOLD 10
#define STRENGTH 15

namespace nemo {
namespace mpi {

bool isInRange(nidx_t n, const nrange& range);

Mapper::Mapper() {

}
Mapper::Mapper(unsigned neurons, boost::mpi::communicator& world, const nemo::Network& net, mapperType type) :
		m_rank(0), m_workers(world.size() - 1), m_neuronsCount(neurons), m_type(type), m_rankMap(neurons, 0) {

	if ( type == UNIFORM ) {
		uniformPartition();
	}
	else if ( type == NEWMAN ) {
		newmansPartition(net);
	}
	else if ( type == RANDOM ) {
		randomPartition();
	}
}

std::string Mapper::mapperTypeDescr() const {
	if ( m_type == UNIFORM ) {
		return "Uniform";
	}
	else if ( m_type == NEWMAN ) {
		return "Newman";
	}
	else if ( m_type == RANDOM ) {
		return "RANDOM";
	}
	return "Invalid";
}

void Mapper::setRank(rank_t rank) {
	m_rank = rank;
}

/* All workers have network size equal the returned size expect the last worker who has less neurons */
unsigned Mapper::getAvgNeurons() {
	return ceil(((float) m_neuronsCount) / ((float) m_workers));
}

unsigned Mapper::getNodesize(rank_t rank) {
	if ( ((unsigned) rank != m_workers + 1) )
		return getAvgNeurons();

	return m_neuronsCount - ((rank - 1) * getAvgNeurons());
}

void Mapper::makeRanges() {
	unsigned start = 0;

	m_ranges.push_back(std::make_pair(0, 0));
	for ( unsigned r = 1; r <= m_workers; r++ ) {
		start = (r - 1) * getAvgNeurons();
		std::pair<unsigned, unsigned> range(start, start + getNodesize(r) - 1);
		m_ranges.push_back(range);
	}
}

bool isInRange(nidx_t n, const nrange& range) {
	return (n >= range.first && n <= range.second);
}

void Mapper::uniformPartition() {
	makeRanges();

	for ( unsigned n = 0; n < m_neuronsCount; n++ ) {
		for ( unsigned r = 1; r <= m_workers; r++ ) {
			if ( isInRange(n, m_ranges.at(r)) ) {
				m_rankMap[n] = r;
				break;
			}
		}
	}
}

bool Mapper::isLocal(nidx_t n) const {
	return rankOf(n) == m_rank;
}

rank_t Mapper::rankOf(nidx_t n) const {
	return m_rankMap[n];
}

unsigned Mapper::neuronCount() const {
	return m_neuronsCount;
}

const std::vector<rank_t>& Mapper::getRankMap() const {
	return m_rankMap;
}

void Mapper::newmansPartition(const nemo::Network& net) {
	if ( m_workers == 1 ) {
		m_rankMap = std::vector<rank_t>(m_neuronsCount, 1);
		return;
	}

	std::vector<std::vector<unsigned> > subPartitions(1, std::vector<unsigned>());

	for ( unsigned n = 0; n < m_neuronsCount; n++ )
		subPartitions.back().push_back(n);

	unsigned w = 0;
	unsigned availablePartitions = 1;
	while (w >= 0) {
		std::cout << "\n\niter = " << w << std::endl;
		subPartitions.push_back(std::vector<unsigned>());
		subPartitions.push_back(std::vector<unsigned>());
		newmansPartition(net, subPartitions[w], subPartitions[2 * w + 1], subPartitions[2 * w + 2]);
//		std::cout << "\n\npartitions size: " << subPartitions.size() << std::endl;
		w++;
		availablePartitions++;
		if ( m_workers == availablePartitions )
			break;
	}

//	std::cout << "\n\npartitions: " << std::endl;
//	for (unsigned i = 0; i < subPartitions.size(); i++) {
//		std::cout << "P" << i << ":  ";
//		for (unsigned j = 0; j < subPartitions[i].size(); j++) {
//			std::cout << subPartitions[i][j] << " ";
//		}
//		std::cout << std::endl << std::endl;
//	}

	typedef std::vector<std::vector<unsigned> >::reverse_iterator outItType;
	typedef std::vector<unsigned>::const_iterator inItType;
//	std::cout << "m_rankMap: " << std::endl;

	for ( unsigned currentNeuron = 0; currentNeuron < m_neuronsCount; currentNeuron++ ) {
		bool assigned = false;
		unsigned w = 1;
		for ( outItType outIt = subPartitions.rbegin(); outIt != subPartitions.rend(); ++outIt, ++w ) {
			if ( std::find(outIt->begin(), outIt->end(), currentNeuron) != outIt->end() ) {
				m_rankMap[currentNeuron] = w;
				assigned = true;
//				std::cout << "n = " << currentNeuron << " --> " << m_rankMap[currentNeuron]
//						<< std::endl;
			}
			if ( assigned )
				break;
		}
	}
}

// Mapping Algorithm implementation
void Mapper::newmansPartition(const nemo::Network& net, std::vector<unsigned>& partition,
		std::vector<unsigned>& subPart1, std::vector<unsigned>& subPart2) {
	nemo::network::NetworkImpl netImpl = *net.m_impl;

	unsigned pSize = partition.size();
	// used for both adjacency and modularity
	std::vector<std::vector<float> > q_matrix(pSize, std::vector<float>(pSize, 0));
	std::vector<unsigned> degrees(pSize, 0);
	std::vector<float> eigenvector(pSize, 0);
	std::vector<float> tmp(pSize, 0);

	//give a valid vector index to each global neuron index
	std::map<float, float> aux;
	for ( unsigned i = 0; i < pSize; ++i )
		aux[partition[i]] = i;

	std::cout << "step 1" << std::endl;
	// Step 1 = Populating Adjacency Matrix
	unsigned edges = 0;
	for ( unsigned i = 0; i < pSize; ++i ) {
		std::vector<synapse_id> synapses = netImpl.getSynapsesFrom(partition[i]);
		for ( unsigned j = 0; j < synapses.size(); ++j ) {
			unsigned target = netImpl.getSynapseTarget(synapses[j]);
			//if ( std::find(partition.begin(), partition.end(), target) == partition.end() )
			//	break;
		  if (!std::binary_search (partition.begin(), partition.end(), target))
		  	break;

			q_matrix[i][aux[target]] = 1;
			q_matrix[aux[target]][i] = 1;
			degrees[i] += 1;
			degrees[aux[target]] += 1;
			edges++;
		}
	}

	std::cout << "step 2" << std::endl;
	// Step 2 = Population of Modularity Matrix using previously collected values for adjacency and degrees per neuron
	for ( unsigned i = 0; i < pSize; ++i ) {
		for ( unsigned j = 0; j < pSize; ++j ) {
			if ( i != j )
				q_matrix[i][j] = q_matrix[i][j] - ((float) (degrees[i] * degrees[j]) / (float) (2 * edges));
			else
				q_matrix[i][j] = STRENGTH;
		}
	}

	std::cout << "step 3" << std::endl;
	// Step 3 = Finding leading eigenvector (Von Moses algorithm, power-iteration)
	float norm = 0, norm_sq = 0, step = 0;
	while (step < THRESHOLD) {
		norm_sq = 0;
		for ( unsigned i = 0; i < pSize; ++i ) {
			tmp[i] = 0;
			for ( unsigned j = 0; j < pSize; ++j )
				tmp[i] += q_matrix[i][j] * eigenvector[j];
			norm_sq += tmp[i] * tmp[i];
		}
		norm = sqrt(norm_sq);
		for ( unsigned i = 0; i < pSize; ++i )
			eigenvector[i] = tmp[i] / norm;
		step++;
	}

	std::cout << "step 4" << std::endl;
	// Step 4 = Partitioning the graph
	for ( unsigned i = 0; i < partition.size(); ++i ) {
		if ( eigenvector[i] > 0 ) {
			subPart1.push_back(partition[i]);
		}
		else {
			subPart2.push_back(partition[i]);
		}
	}

	std::cout << "step 5" << std::endl;
	// The check for empty graphs - if one of partitions is empty - do a uniform split
	if ( subPart1.size() == 0 || subPart2.size() == 0 ) {
		subPart1.clear();
		subPart2.clear();
		for ( unsigned i = 0; i < pSize / 2; ++i )
			subPart1.push_back(partition[i]);
		for ( unsigned i = pSize / 2; i < pSize; ++i )
			subPart2.push_back(partition[i]);
	}
}

void Mapper::randomPartition() {
	rng_t rng;
	uirng_t randomWorker(rng, boost::uniform_int<>(1, m_workers));
	for(unsigned n=0; n < m_neuronsCount; n++) {
		m_rankMap[n] = randomWorker();
	}
}
} // end namespace mpi
} // end namespace nemo
