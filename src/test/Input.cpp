namespace nemo {
	namespace test {
		namespace input {

/*! Add a number of input neurons to a network, with optionally an equal number
 * of Izhikevich neurons. */
void
addNeurons(nemo::Network& net, unsigned ncount, bool izhikevich)
{
	unsigned input = net.addNeuronType("Input");
	for(unsigned i=0; i<ncount; ++i) {
		net.addNeuron(input, i, 0, NULL);
	}
	if(izhikevich) {
		unsigned iz = net.addNeuronType("Izhikevich");
		float param[7] = { 0.02f, 0.2f, -65.0f, 8.0f, 0.0f, 0.2f*-65.0f, -65.0f };
		for(unsigned i=ncount; i<2*ncount; ++i) {
			net.addNeuron(iz, i, 7, param);
		}
	}
}




/* Test that we're able to create input neurons
 *
 * Input neurons are special in that they have neither parameters nor state */
void
create(backend_t backend, unsigned ncount, bool iz)
{
	nemo::Network net;
	addNeurons(net, ncount, iz);
	nemo::Configuration conf = configuration(false, 1024, backend);
	boost::scoped_ptr<nemo::Simulation> sim(simulation(net, conf));
	sim->step();
}


/* Ensure that the input neurons fire, as instructed */
void
simple(backend_t backend, unsigned ncount, bool iz)
{
	nemo::Network net;
	addNeurons(net, ncount, iz);
	nemo::Configuration conf = configuration(false, 1024, backend);
	boost::scoped_ptr<nemo::Simulation> sim(simulation(net, conf));
	std::vector<unsigned> fstim;
	rng_t rng;
	urng_t random(rng, boost::uniform_real<double>(0, 1.0));
	for(unsigned t=0; t<1000; ++t) {
		double p = 0.5*random();
		fstim.clear();
		for(unsigned n=0; n<ncount; ++n) {
			if(random() < p) {
				fstim.push_back(n);
			}
		}
		const std::vector<unsigned>& fired = sim->step(fstim);
		BOOST_REQUIRE_EQUAL(fired.size(), fstim.size());
		for(size_t i = 0; i < fstim.size(); ++i) {
			BOOST_REQUIRE_EQUAL(fired[i], fstim[i]);
		}
	}
}

		}
	}
}
BOOST_AUTO_TEST_SUITE(input)
	TEST_ALL_BACKENDS_N(create1,   nemo::test::input::create,    1, false)
	TEST_ALL_BACKENDS_N(create1k,  nemo::test::input::create, 1000, false)
	TEST_ALL_BACKENDS_N(create1N,  nemo::test::input::create,    1, true )
	TEST_ALL_BACKENDS_N(create1kN, nemo::test::input::create, 1000, true )
	TEST_ALL_BACKENDS_N(simple1,   nemo::test::input::simple,    1, false)
	TEST_ALL_BACKENDS_N(simple1k,  nemo::test::input::simple, 1000, false)
	TEST_ALL_BACKENDS_N(simple1N,  nemo::test::input::simple,    1, true )
	TEST_ALL_BACKENDS_N(simple1kN, nemo::test::input::simple, 1000, true )
BOOST_AUTO_TEST_SUITE_END()
