/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with NeMo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <utility>
#include <boost/random.hpp>
#include <boost/test/unit_test.hpp>

#include <nemo.hpp>
#include <nemo.h>
#include <nemo/fixedpoint.hpp>

#include "c_api.hpp"
#include "utils.hpp"

namespace nemo {
	namespace test {
		namespace c_api {

typedef boost::mt19937 rng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_real<double> > urng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_int<> > uirng_t;


void
c_safeCall(nemo_status_t err)
{
	if(err != NEMO_OK) {
		std::cerr << nemo_strerror() << std::endl;
		exit(-1);
	}
}



template<typename T>
T
c_safeAlloc(T ptr)
{
	if(ptr == NULL) {
		std::cerr << nemo_strerror() << std::endl;
		exit(-1);
	}
	return ptr;
}


void
setBackend(nemo_configuration_t conf, backend_t backend)
{
	switch(backend) {
		case NEMO_BACKEND_CPU: nemo_set_cpu_backend(conf); break;
		case NEMO_BACKEND_CUDA: nemo_set_cuda_backend(conf, 0); break;
		default: BOOST_REQUIRE(false);
	}
}



void
addExcitatoryNeuron(
		nemo::Network* net,
		nemo_network_t c_net,
		unsigned c_iz,
		unsigned nidx, urng_t& param)
{
	float v = -65.0f;
	float a = 0.02f;
	float b = 0.2f;
	float r1 = float(param());
	float r2 = float(param());
	float c = v + 15.0f * r1 * r1;
	float d = 8.0f - 6.0f * r2 * r2;
	float u = b * v;
	float sigma = 5.0f;
	net->addNeuron(nidx, a, b, c, d, u, v, sigma);
	float c_args[7] = {a, b, c, d, sigma, u, v};
	nemo_add_neuron(c_net, c_iz, nidx, 7, c_args);
}


void
addExcitatorySynapses(
		nemo::Network* net,
		nemo_network_t c_net,
		unsigned source,
		unsigned ncount,
		unsigned scount,
		uirng_t& rtarget,
		urng_t& rweight)
{
	for(unsigned s = 0; s < scount; ++s) {
		unsigned target = rtarget();
		float weight = 0.5f * float(rweight());
		net->addSynapse(source, target, 1U, weight, 0);
		nemo_add_synapse(c_net, source, target, 1U, weight, 0, NULL);
	}
}


void
addInhibitoryNeuron(
		nemo::Network* net,
		nemo_network_t c_net,
		unsigned c_iz,
		unsigned nidx,
		urng_t& param)
{
	float v = -65.0f;
	float r1 = float(param());
	float a = 0.02f + 0.08f * r1;
	float r2 = float(param());
	float b = 0.25f - 0.05f * r2;
	float c = v;
	float d = 2.0f;
	float u = b * v;
	float sigma = 2.0f;
	net->addNeuron(nidx, a, b, c, d, u, v, sigma);
	float c_args[7] = {a, b, c, d, sigma, u, v};
	nemo_add_neuron(c_net, c_iz, nidx, 7, c_args);
}



void
addInhibitorySynapses(
		nemo::Network* net,
		nemo_network_t c_net,
		unsigned source,
		unsigned ncount,
		unsigned scount,
		uirng_t& rtarget,
		urng_t& rweight)
{
	for(unsigned s = 0; s < scount; ++s) {
		unsigned target = rtarget();
		float weight = float(-rweight());
		net->addSynapse(source, target, 1U, weight, 0);
		nemo_add_synapse(c_net, source, target, 1U, weight, 0, NULL);
	}
}



/* Array arguments to NeMo API should be NULL if there is no data. */
template<typename T>
T*
vectorData(std::vector<T>& vec)
{
	return vec.size() > 0 ? &vec[0] : NULL;
}



void
c_runSimulation(
		const nemo_network_t net,
		nemo_configuration_t conf,
		unsigned seconds,
		std::vector<unsigned>& fstim,
		std::vector<unsigned>& istim_nidx,
		std::vector<float>& istim_current,
		std::vector<unsigned>* fcycles,
		std::vector<unsigned>* fnidx)
{
	nemo_simulation_t sim = c_safeAlloc(nemo_new_simulation(net, conf));

	fcycles->clear();
	fnidx->clear();

	//! todo vary the step size between reads to firing buffer
	
	for(unsigned s = 0; s < seconds; ++s)
	for(unsigned ms = 0; ms < 1000; ++ms) {

		unsigned *fired;
		size_t fired_len;

		if(s == 0 && ms == 0) {
			nemo_step(sim, vectorData(fstim), fstim.size(),
					vectorData(istim_nidx), vectorData(istim_current), istim_nidx.size(),
					&fired, &fired_len);
		} else {
			nemo_step(sim, NULL, 0, NULL, NULL, 0, &fired, &fired_len);
		}
		// read back a few synapses every now and then just to make sure it works
		if(ms % 100 == 0) {
			synapse_id* synapses;
			size_t len;
			nemo_get_synapses_from_s(sim, 1, &synapses, &len);

			float weight;
			nemo_get_synapse_weight_s(sim, synapses[0], &weight);

			unsigned target;
			nemo_get_synapse_target_s(sim, synapses[0], &target);

			unsigned delay;
			nemo_get_synapse_delay_s(sim, synapses[0], &delay);

			unsigned char plastic;
			nemo_get_synapse_plastic_s(sim, synapses[0], &plastic);
		}

		// read back a some membrane potential, just to make sure it works
		if(ms % 100 == 0) {
			float v;
			nemo_get_membrane_potential(sim, 40, &v);
			nemo_get_membrane_potential(sim, 50, &v);
		}

		// push data back onto local buffers
		std::copy(fired, fired + fired_len, back_inserter(*fnidx));
		std::fill_n(back_inserter(*fcycles), fired_len, s*1000 + ms);
	}

	// try replacing a neuron, just to make sure it doesn't make things fall over.
	nemo_set_neuron_iz_s(sim, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -64.0f, 0.0f);
	{
		float v;
		nemo_get_membrane_potential(sim, 0, &v);
		BOOST_REQUIRE_EQUAL(v, -64.0);
	}

	nemo_delete_simulation(sim);
}




/*! Compare simulation runs using C and C++ APIs with optional firing and
 * current stimulus during cycle 100 */
void
compareWithCpp(bool useFstim, bool useIstim)
{
	unsigned ncount = 1000;
	unsigned scount = 1000;
	//! \todo run test with stdp enabled as well
	bool stdp = false;

	rng_t rng;
	/* Neuron parameters and weights are partially randomised */
	urng_t randomParameter(rng, boost::uniform_real<double>(0, 1));
	uirng_t randomTarget(rng, boost::uniform_int<>(0, ncount-1));

	std::cerr << "Nemo version: " << nemo_version() << std::endl;

	std::cerr << "Creating network (C++ API)\n";
	nemo::Network* net = new nemo::Network();
	std::cerr << "Creating network (C API)\n";
	nemo_network_t c_net = nemo_new_network();
	unsigned c_iz;
	nemo_add_neuron_type(c_net, "Izhikevich", &c_iz);

	std::cerr << "Populating networks\n";
	for(unsigned nidx=0; nidx < ncount; ++nidx) {
		if(nidx < (ncount * 4) / 5) { // excitatory
			addExcitatoryNeuron(net, c_net, c_iz, nidx, randomParameter);
			addExcitatorySynapses(net, c_net, nidx, ncount, scount, randomTarget, randomParameter);
		} else { // inhibitory
			addInhibitoryNeuron(net, c_net, c_iz, nidx, randomParameter);
			addInhibitorySynapses(net, c_net, nidx, ncount, scount, randomTarget, randomParameter);
		}
	}

	nemo::Configuration conf;

	unsigned duration = 2;

	std::vector<unsigned> cycles1, cycles2, nidx1, nidx2;

	std::vector<unsigned> fstim;
	if(useFstim) {
		fstim.push_back(100);
	}

	std::vector< std::pair<unsigned, float> > istim;
	std::vector<unsigned> istim_nidx;
	std::vector<float> istim_current;
	if(useIstim) {
		istim.push_back(std::make_pair(20U, 20.0f));
		istim_nidx.push_back(20);
		istim_current.push_back(20.0f);

		istim.push_back(std::make_pair(40U, 20.0f));
		istim_nidx.push_back(40);
		istim_current.push_back(20.0f);

		istim.push_back(std::make_pair(60U, 20.0f));
		istim_nidx.push_back(60);
		istim_current.push_back(20.0f);
	}

	std::cerr << "Running network (C++ API)\n";
	runSimulation(net, conf, duration, &cycles1, &nidx1, stdp, fstim, istim);

	unsigned cuda_dcount;
	if(nemo_cuda_device_count(&cuda_dcount) == NEMO_OK) {
		std::cerr << cuda_dcount << " CUDA devices available\n";

		for(unsigned i=0; i < cuda_dcount; ++i) {
			const char* cuda_descr;
			c_safeCall(nemo_cuda_device_description(i, &cuda_descr));
			std::cerr << "\tDevice " << i << ": " << cuda_descr << "\n";
		}
	}


	nemo_configuration_t c_conf = nemo_new_configuration();
	const char* descr;
	c_safeCall(nemo_backend_description(c_conf, &descr));
	std::cerr << descr << std::endl;
	std::cerr << "Running network (C API)\n";
	c_runSimulation(c_net, c_conf, duration,
			fstim, istim_nidx, istim_current, &cycles2, &nidx2);

	std::cerr << "Checking results\n";
	compareSimulationResults(cycles1, nidx1, cycles2, nidx2);
	nemo_delete_configuration(c_conf);
	nemo_delete_network(c_net);
}



/*! Test that
 *
 * 1. we get back synapse id
 * 2. the synapse ids are correct (based on our knowledge of the internals
 */
void
testSynapseId()
{
	nemo_network_t net = nemo_new_network();

	synapse_id id00, id01, id10;

	nemo_add_synapse(net, 0, 1, 1, 0.0f, 0, &id00);
	nemo_add_synapse(net, 0, 1, 1, 0.0f, 0, &id01);
	nemo_add_synapse(net, 1, 0, 1, 0.0f, 0, &id10);

	BOOST_REQUIRE_EQUAL(id01 - id00, 1U);
	BOOST_REQUIRE_EQUAL(id10 & 0xffffffff, 0U);
	BOOST_REQUIRE_EQUAL(id00 & 0xffffffff, 0U);
	BOOST_REQUIRE_EQUAL(id01 & 0xffffffff, 1U);
	BOOST_REQUIRE_EQUAL(id00 >> 32, 0U);
	BOOST_REQUIRE_EQUAL(id01 >> 32, 0U);
	BOOST_REQUIRE_EQUAL(id10 >> 32, 1U);

	nemo_delete_network(net);
}



/* Both the simulation and network classes have neuron setters. Here we
 * perform the same test for both. */
void
testSetNeuron()
{
	float a = 0.02f;
	float b = 0.2f;
	float c = -65.0f+15.0f*0.25f;
	float d = 8.0f-6.0f*0.25f;
	float v = -65.0f;
	float u = b * v;
	float sigma = 5.0f;
	float val;

	/* Create a minimal network with a single neuron */
	nemo_network_t net = nemo_new_network();

	/* setNeuron should only succeed for existing neurons */
	BOOST_REQUIRE(nemo_set_neuron_iz_n(net, 0, a, b, c, d, u, v, sigma) != NEMO_OK);

	nemo_add_neuron_iz(net, 0, a, b, c-0.1f, d, u, v-1.0f, sigma);

	/* Invalid neuron */
	BOOST_REQUIRE(nemo_get_neuron_parameter_n(net, 1, 0, &val) != NEMO_OK);
	BOOST_REQUIRE(nemo_get_neuron_state_n(net, 1, 0, &val) != NEMO_OK);

	/* Invalid parameter */
	BOOST_REQUIRE(nemo_get_neuron_parameter_n(net, 0, 5, &val) != NEMO_OK);
	BOOST_REQUIRE(nemo_get_neuron_state_n(net, 0, 2, &val) != NEMO_OK);

	float e = 0.1f;
	BOOST_REQUIRE_EQUAL(nemo_set_neuron_iz_n(net, 0, a-e, b-e, c-e, d-e, u-e, v-e, sigma-e), NEMO_OK);
	nemo_get_neuron_parameter_n(net, 0, 0, &val); BOOST_REQUIRE_EQUAL(val, a-e);
	nemo_get_neuron_parameter_n(net, 0, 1, &val); BOOST_REQUIRE_EQUAL(val, b-e);
	nemo_get_neuron_parameter_n(net, 0, 2, &val); BOOST_REQUIRE_EQUAL(val, c-e);
	nemo_get_neuron_parameter_n(net, 0, 3, &val); BOOST_REQUIRE_EQUAL(val, d-e);
	nemo_get_neuron_parameter_n(net, 0, 4, &val); BOOST_REQUIRE_EQUAL(val, sigma-e);
	nemo_get_neuron_state_n(net, 0, 0, &val); BOOST_REQUIRE_EQUAL(val, u-e);
	nemo_get_neuron_state_n(net, 0, 1, &val); BOOST_REQUIRE_EQUAL(val, v-e);

	/* Try setting individual parameters during construction */

	nemo_set_neuron_parameter_n(net, 0, 0, a);
	nemo_get_neuron_parameter_n(net, 0, 0, &val); BOOST_REQUIRE_EQUAL(val, a);

	nemo_set_neuron_parameter_n(net, 0, 1, b);
	nemo_get_neuron_parameter_n(net, 0, 1, &val); BOOST_REQUIRE_EQUAL(val, b);

	nemo_set_neuron_parameter_n(net, 0, 2, c);
	nemo_get_neuron_parameter_n(net, 0, 2, &val); BOOST_REQUIRE_EQUAL(val, c);

	nemo_set_neuron_parameter_n(net, 0, 3, d);
	nemo_get_neuron_parameter_n(net, 0, 3, &val); BOOST_REQUIRE_EQUAL(val, d);

	nemo_set_neuron_parameter_n(net, 0, 4, sigma);
	nemo_get_neuron_parameter_n(net, 0, 4, &val); BOOST_REQUIRE_EQUAL(val, sigma);

	nemo_set_neuron_state_n(net, 0, 0, u);
	nemo_get_neuron_state_n(net, 0, 0, &val); BOOST_REQUIRE_EQUAL(val, u);

	nemo_set_neuron_state_n(net, 0, 1, v);
	nemo_get_neuron_state_n(net, 0, 1, &val); BOOST_REQUIRE_EQUAL(val, v);

	/* Invalid neuron */
	BOOST_REQUIRE(nemo_set_neuron_parameter_n(net, 1, 0, 0.0f) != NEMO_OK);
	BOOST_REQUIRE(nemo_set_neuron_state_n(net, 1, 0, 0.0f) != NEMO_OK);

	/* Invalid parameter */
	BOOST_REQUIRE(nemo_set_neuron_parameter_n(net, 0, 5, 0.0f) != NEMO_OK);
	BOOST_REQUIRE(nemo_set_neuron_state_n(net, 0, 2, 0.0f) != NEMO_OK);

	nemo_configuration_t conf = nemo_new_configuration();

	/* Try setting individual parameters during simulation */
	{
		nemo_simulation_t sim = c_safeAlloc(nemo_new_simulation(net, conf));
		nemo_step(sim, NULL, 0, NULL, NULL, 0, NULL, NULL);

		nemo_set_neuron_state_s(sim, 0, 0, u-e);
		nemo_get_neuron_state_s(sim, 0, 0, &val); BOOST_REQUIRE_EQUAL(val, u-e);

		nemo_set_neuron_state_s(sim, 0, 1, v-e);
		nemo_get_neuron_state_s(sim, 0, 1, &val); BOOST_REQUIRE_EQUAL(val, v-e);

		nemo_step(sim, NULL, 0, NULL, NULL, 0, NULL, NULL);

		nemo_set_neuron_parameter_s(sim, 0, 0, a-e);
		nemo_get_neuron_parameter_s(sim, 0, 0, &val); BOOST_REQUIRE_EQUAL(val, a-e);

		nemo_set_neuron_parameter_s(sim, 0, 1, b-e);
		nemo_get_neuron_parameter_s(sim, 0, 1, &val); BOOST_REQUIRE_EQUAL(val, b-e);

		nemo_set_neuron_parameter_s(sim, 0, 2, c-e);
		nemo_get_neuron_parameter_s(sim, 0, 2, &val); BOOST_REQUIRE_EQUAL(val, c-e);

		nemo_set_neuron_parameter_s(sim, 0, 3, d-e);
		nemo_get_neuron_parameter_s(sim, 0, 3, &val); BOOST_REQUIRE_EQUAL(val, d-e);

		nemo_set_neuron_parameter_s(sim, 0, 4, sigma-e);
		nemo_get_neuron_parameter_s(sim, 0, 4, &val); BOOST_REQUIRE_EQUAL(val, sigma-e);

		/* Invalid neuron */
		BOOST_REQUIRE(nemo_set_neuron_parameter_s(sim, 1, 0, 0.0f) != NEMO_OK);
		BOOST_REQUIRE(nemo_set_neuron_state_s(sim, 1, 0, 0.0f) != NEMO_OK);
		BOOST_REQUIRE(nemo_get_neuron_parameter_s(sim, 1, 0, &val) != NEMO_OK);
		BOOST_REQUIRE(nemo_get_neuron_state_s(sim, 1, 0, &val) != NEMO_OK);

		/* Invalid parameter */
		BOOST_REQUIRE(nemo_set_neuron_parameter_s(sim, 0, 5, 0.0f) != NEMO_OK);
		BOOST_REQUIRE(nemo_set_neuron_state_s(sim, 0, 2, 0.0f) != NEMO_OK);
		BOOST_REQUIRE(nemo_get_neuron_parameter_s(sim, 0, 5, &val) != NEMO_OK);
		BOOST_REQUIRE(nemo_get_neuron_state_s(sim, 0, 2, &val) != NEMO_OK);

		nemo_delete_simulation(sim);
	}

	float v0 = 0.0f;
	{
		nemo_simulation_t sim = nemo_new_simulation(net, conf);
		nemo_step(sim, NULL, 0, NULL, NULL, 0, NULL, NULL);
		nemo_get_membrane_potential(sim, 0, &v0);
		nemo_delete_simulation(sim);
	}

	{
		nemo_simulation_t sim = nemo_new_simulation(net, conf);
		nemo_get_neuron_state_s(sim, 0, 0, &val); BOOST_REQUIRE_EQUAL(val, u);
		nemo_get_neuron_state_s(sim, 0, 1, &val); BOOST_REQUIRE_EQUAL(val, v);
		/* Marginally change the 'c' parameter. This is only used if the neuron
		 * fires (which it shouldn't do this cycle). This modification
		 * therefore should not affect the simulation result (here measured via
		 * the membrane potential) */
		nemo_set_neuron_iz_s(sim, 0, a, b, c+1.0f, d, u, v, sigma);
		nemo_step(sim, NULL, 0, NULL, NULL, 0, NULL, NULL);
		nemo_get_membrane_potential(sim, 0, &val); BOOST_REQUIRE_EQUAL(v0, val);
		nemo_get_neuron_parameter_s(sim, 0, 0, &val); BOOST_REQUIRE_EQUAL(val, a);
		nemo_get_neuron_parameter_s(sim, 0, 1, &val); BOOST_REQUIRE_EQUAL(val, b);
		nemo_get_neuron_parameter_s(sim, 0, 2, &val); BOOST_REQUIRE_EQUAL(val, c+1.0f);
		nemo_get_neuron_parameter_s(sim, 0, 3, &val); BOOST_REQUIRE_EQUAL(val, d);
		nemo_get_neuron_parameter_s(sim, 0, 4, &val); BOOST_REQUIRE_EQUAL(val, sigma);
		nemo_delete_simulation(sim);
	}

	{
		/* Modify membrane potential after simulation has been created.
		 * Again the result should be the same */
		nemo_network_t net1 = nemo_new_network();
		nemo_add_neuron_iz(net1, 0, a, b, c, d, u, v-1.0f, sigma);
		nemo_simulation_t sim = nemo_new_simulation(net1, conf);
		nemo_set_neuron_iz_s(sim, 0, a, b, c, d, u, v, sigma);
		nemo_step(sim, NULL, 0, NULL, NULL, 0, NULL, NULL);
		nemo_get_membrane_potential(sim, 0, &val); BOOST_REQUIRE_EQUAL(v0, val);
		nemo_delete_simulation(sim);
		nemo_delete_network(net1);
	}

	nemo_delete_network(net);
	nemo_delete_configuration(conf);
}



/*! Create simulation and verify that the simulation data contains the same
 * synapses as the input network. Neurons are assumed to lie in a contigous
 * range of indices starting at n0. */
void
testGetSynapses(backend_t backend, unsigned n0)
{
	unsigned fbits = 20;

	nemo_network_t net = c_safeAlloc(nemo_new_network());

	/* Construct a network with a triangular connection matrix and fixed
	 * synapse properties depending on source and target neurons */
	unsigned ncount = 1000;
	for(unsigned n = 0; n < ncount; ++n) {
		unsigned source = n0 + n;
		c_safeCall(nemo_add_neuron_iz(net, source, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
		for(unsigned s = 0; s < n; ++s) {
			unsigned target = n0 + s;
			synapse_id id;
			c_safeCall(nemo_add_synapse(net, source, target, 1 + s%20, float(s % 10), s%2, &id));
		}
	}

	nemo_configuration_t conf = c_safeAlloc(nemo_new_configuration());
	setBackend(conf, backend);

	nemo_simulation_t sim = c_safeAlloc(nemo_new_simulation(net, conf));

	for(unsigned src = n0; src < n0 + ncount; ++src) {

		synapse_id* ids;
		size_t slen;
		size_t nlen;
		c_safeCall(nemo_get_synapses_from_n(net, src, &ids, &nlen));
		c_safeCall(nemo_get_synapses_from_s(sim, src, &ids, &slen));

		BOOST_REQUIRE_EQUAL(nlen, slen);
		BOOST_REQUIRE_EQUAL(slen, src-n0);

		for(unsigned i = 0; i < slen; ++i) {

			synapse_id id = ids[i];

			unsigned ns, ss;
			c_safeCall(nemo_get_synapse_source_s(sim, id, &ss));
			c_safeCall(nemo_get_synapse_source_n(net, id, &ns));
			BOOST_REQUIRE_EQUAL(ns, ss);
			BOOST_REQUIRE_EQUAL(ns, src);

			unsigned nt, st;
			c_safeCall(nemo_get_synapse_target_n(net, id, &nt));
			c_safeCall(nemo_get_synapse_target_s(sim, id, &st));
			BOOST_REQUIRE_EQUAL(nt, st);
			BOOST_REQUIRE_EQUAL(nt, n0 + i);

			unsigned nd, sd;
			c_safeCall(nemo_get_synapse_delay_n(net, id, &nd));
			c_safeCall(nemo_get_synapse_delay_s(sim, id, &sd));
			BOOST_REQUIRE_EQUAL(nd, sd);
			BOOST_REQUIRE_EQUAL(nd, 1 + i % 20);

			float nw, sw;
			c_safeCall(nemo_get_synapse_weight_n(net, id, &nw));
			c_safeCall(nemo_get_synapse_weight_s(sim, id, &sw));
			nw = fx_toFloat(fx_toFix(nw, fbits), fbits);
			BOOST_REQUIRE_EQUAL(nw, sw);
			BOOST_REQUIRE_EQUAL(nw, float(i % 10));

			unsigned char np, sp;
			c_safeCall(nemo_get_synapse_plastic_n(net, id, &np));
			c_safeCall(nemo_get_synapse_plastic_s(sim, id, &sp));
			BOOST_REQUIRE_EQUAL(np, sp);
			BOOST_REQUIRE_EQUAL(np, i % 2);
		}
	}

	nemo_delete_configuration(conf);
	nemo_delete_network(net);
	nemo_delete_simulation(sim);
}



}	}	}
