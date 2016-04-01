#include <vector>
#include <set>
#include <boost/random.hpp>
#include <nemo.hpp>
#include <nemo/types.hpp>

typedef boost::mt19937 rng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_real<double> > urng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_int<> > uirng_t;

namespace nemo {
namespace random {

void addExcitatoryNeuron(nemo::Network* net, unsigned nidx, urng_t& param);

void addInhibitoryNeuron(nemo::Network* net, unsigned nidx, urng_t& param);

nemo::Network* constructUniformRandom(unsigned ncount, unsigned scount, unsigned dmax, bool stdp);

nemo::Network* constructWattsStrogatz(unsigned k, unsigned p, unsigned ncount, unsigned dmax, bool stdp);

/* Used for debugging */
nemo::Network* simpleNet(unsigned ncount, unsigned dmax, bool stdp);

/* Used for debugging */
nemo::Network* constructSemiRandom(unsigned ncount, unsigned scount, unsigned dmax, bool stdp, unsigned workers, float ratio);
} // namespace random
} // namespace nemo
