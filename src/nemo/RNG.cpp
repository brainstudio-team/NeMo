#include "RNG.hpp"

#include <cmath>
#include <boost/random.hpp>

#include "exception.hpp"


unsigned
urand(RNG* rng)
{
	unsigned t = (rng->state[0]^(rng->state[0]<<11));
	rng->state[0] = rng->state[1];
	rng->state[1] = rng->state[2];
	rng->state[2] = rng->state[3];
	rng->state[3] = (rng->state[3]^(rng->state[3]>>19))^(t^(t>>8));
	return rng->state[3];
}



/* For various reasons this generates a pair of samples for each call. If nesc.
 * then you can just stash one of them somewhere until the next time it is
 * needed or something.  */
float
nrand(RNG* rng)
{
	float a = urand(rng) * 1.4629180792671596810513378043098e-9f;
	float b = urand(rng) * 0.00000000023283064365386962890625f;
	float r = sqrtf(-2*logf(1-b));
	// cosf(a) * r // ignore the second random
	return sinf(a) * r;
}


namespace nemo {

void
initialiseRng(nidx_t minNeuronIdx, nidx_t maxNeuronIdx, std::vector<RNG>& rngs)
{
	assert_or_throw(minNeuronIdx <= maxNeuronIdx,
			"Invalid neuron range when initialising RNG");

	//! \todo allow users to seed this RNG
	typedef boost::mt19937 rng_t;
	rng_t rng;

	boost::variate_generator<rng_t, boost::uniform_int<unsigned long> >
		seed(rng, boost::uniform_int<unsigned long>(0, 0x7fffffff));

	for(unsigned gidx=0; gidx < 4 * minNeuronIdx; ++gidx) {
		seed();
	}

	for(unsigned gidx = minNeuronIdx, gidx_end = maxNeuronIdx;
			gidx <= gidx_end; ++gidx) {
		// some of these neuron indices may be invalid
		RNG& seeds = rngs.at(gidx - minNeuronIdx);
		for(unsigned plane=0; plane < 4; ++plane) {
			seeds.state[plane] = seed();
		}
	}
}


} // end namespace
