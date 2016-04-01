#define BOOST_TEST_MODULE nemo test

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/random.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

#include <nemo/ConfigurationImpl.hpp>


/* When serialising to file the binary representation of floating-point values
 * is not preserved. The best we can do is therefore to make sure we are very
 * close */

float tolerance = 0.0001f; // percent

template<typename T>
void
check_close(const std::vector<T>& lhs,
		const std::vector<T>& rhs)
{
	BOOST_CHECK_EQUAL(lhs.size(), rhs.size());

	for(size_t i = 0; i < lhs.size(); ++i) {
		BOOST_CHECK_CLOSE(lhs.at(i), rhs.at(i), tolerance);
	}
}


namespace nemo {


void
check_close(const StdpFunction& lhs, const StdpFunction& rhs)
{
	::check_close<float>(lhs.prefire(), rhs.prefire());
	::check_close<float>(lhs.postfire(), rhs.postfire());
	BOOST_CHECK_CLOSE(lhs.maxWeight(), rhs.maxWeight(), tolerance);
	BOOST_CHECK_CLOSE(lhs.minWeight(), rhs.minWeight(), tolerance);
}



void
check_close(const nemo::ConfigurationImpl& lhs,
		const nemo::ConfigurationImpl& rhs)
{
	BOOST_CHECK_EQUAL(lhs.m_logging, rhs.m_logging);
	BOOST_CHECK_EQUAL(bool(lhs.stdpFunction()), bool(rhs.stdpFunction()));
	if(lhs.stdpFunction() && rhs.stdpFunction()) {
		check_close(lhs.stdpFunction().get(), rhs.stdpFunction().get());
	}
	BOOST_CHECK_EQUAL(lhs.m_cudaPartitionSize, rhs.m_cudaPartitionSize);
}

} // namespace nemo



nemo::ConfigurationImpl
randomConfiguration()
{
	typedef boost::mt19937 rng_t;
	typedef boost::variate_generator<rng_t&, boost::uniform_real<float> > ufrng_t;
	typedef boost::variate_generator<rng_t&, boost::uniform_int<> > uirng_t;

	rng_t rng;
	rng.seed(static_cast<unsigned int>(std::time(0)));
	ufrng_t randf(rng, boost::uniform_real<float>(0, 1));
	uirng_t randi(rng, boost::uniform_int<>(0, 1<<15));

	nemo::ConfigurationImpl conf;
	conf.enableLogging();
	conf.setCudaPartitionSize(randi());

	std::vector<float> prefire;
	std::vector<float> postfire;
	for(unsigned i=0; i<20; ++i) {
		prefire.push_back(randf());
		postfire.push_back(randf());
	}
	conf.setStdpFunction(prefire, postfire, -fabs(randf()), fabs(randf()));

	return conf;
}



/* Verify that the configuration is unchanged by a serialisation/
 * deserialisation roundtrip. This test is only as good as the comparison
 * operators defined above */
BOOST_AUTO_TEST_CASE(serialise_configuration)
{
	for(int i=0; i < 10; ++i) {
		const char* filename = "rnd_cfg";
		nemo::ConfigurationImpl oc = randomConfiguration();
		{
			std::ofstream ofs(filename);
			boost::archive::text_oarchive oa(ofs);
			oa << oc;
		}

		nemo::ConfigurationImpl ic;
		{
			std::ifstream ifs(filename);
			boost::archive::text_iarchive ia(ifs);
			ia >> ic;
		}

		check_close(ic, oc);
	}
}
