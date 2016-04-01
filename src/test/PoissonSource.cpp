namespace nemo {
	namespace test {
		namespace poisson {

/* Crudely test that the average rate over a long run approaches the expected value */
void
testRate(backend_t backend, unsigned duration, bool otherNeurons)
{
	nemo::Network net;
	nemo::Configuration conf = configuration(false, 1024, backend);
	if(otherNeurons) {
		/* This population will never fire */
		createRing(&net, 1024, 1);
	}
	unsigned poisson = net.addNeuronType("PoissonSource");
	float rate = 0.010f;
	net.addNeuron(poisson, 0, 1, &rate);
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(net, conf));
	unsigned nfired = 0;
	for(unsigned t=0; t<duration; ++t) {
		const std::vector<unsigned>& fired = sim->step();
		nfired += fired.size();
	}
	unsigned expected = unsigned(fabsf(float(duration)*rate));
	unsigned deviation = nfired - expected;
	//! \todo use a proper statistical test over a large number of runs
	BOOST_REQUIRE(nfired > 0);
	BOOST_REQUIRE(deviation < expected * 2);
}


		}
	}
}

BOOST_AUTO_TEST_SUITE(poisson)
	TEST_ALL_BACKENDS_N(rate1s, nemo::test::poisson::testRate, 1000, false)
	TEST_ALL_BACKENDS_N(rate10s, nemo::test::poisson::testRate, 10000, false)
	TEST_ALL_BACKENDS_N(rate1sMix, nemo::test::poisson::testRate, 1000, true)
	TEST_ALL_BACKENDS_N(rate10sMix, nemo::test::poisson::testRate, 10000, true)
BOOST_AUTO_TEST_SUITE_END()

