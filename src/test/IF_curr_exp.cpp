#include <boost/scoped_ptr.hpp>

namespace nemo {
	namespace test {
		namespace IF_curr_exp {

enum Types {
	SINGLE,
	MULTIPLE
};


/*! Ensure that it's at all possible to create a network with IF_curr_exp
 * neurons */
void
create(backend_t backend, unsigned duration, Types ntypes)
{
	Network net;
	const unsigned IF_curr_exp = net.addNeuronType("IF_curr_exp");

	const float v_rest = -65.0f;
	const float args[13] = {
		v_rest, 1.0f, 20.0f, 5.0f, 2.0f, 5.0f, 0.1f, -70.0f, -51.0f,
		v_rest, 0.0f, 0.0f, 1000.0f
	};
	net.addNeuron(IF_curr_exp, 0, 13, args);

	if(ntypes == MULTIPLE) {
		/* This population will never fire */
		createRing(&net, 1024, 1);
	}

	Configuration conf = configuration(false, 1024, backend);

	boost::scoped_ptr<Simulation> sim;
	BOOST_REQUIRE_NO_THROW(sim.reset(nemo::simulation(net, conf)));
	for(unsigned i=0; i < duration; ++i){
		BOOST_REQUIRE_NO_THROW(sim->step());
	}
}



BOOST_AUTO_TEST_SUITE(IF_curr_exp)
	// BOOST_AUTO_TEST_CASE(create_s) { create(NEMO_BACKEND_CUDA, 1000, SINGLE); }
	// BOOST_AUTO_TEST_CASE(create_m) { create(NEMO_BACKEND_CUDA, 1000, MULTIPLE); }
	TEST_ALL_BACKENDS_N(create_s, nemo::test::IF_curr_exp::create, 1000, SINGLE)
	TEST_ALL_BACKENDS_N(create_m, nemo::test::IF_curr_exp::create, 1000, MULTIPLE)
BOOST_AUTO_TEST_SUITE_END()

		}
	}
}
