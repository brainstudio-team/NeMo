#!/usr/bin/env python

import unittest
import random
import nemo

class IzNetwork(nemo.Network):

    def __init__(self):
        nemo.Network.__init__(self)
        self._type = self.add_neuron_type('Izhikevich')

    def add_neuron(self, nidx, a, b, c, d, sigma, u, v):
        nemo.Network.add_neuron(self, self._type, nidx,
                {'a':a ,'b': b,'c': c,'d': d,'sigma': sigma}, {'u': u, 'v': v})


def randomSource():
    return random.randint(0, 999)

def randomTarget():
    return randomSource()

def randomDelay():
    return random.randint(1, 20)

def randomWeight():
    return random.uniform(-1.0, 1.0)

def randomPlastic():
    return random.choice([True, False])

def randomParameterIndex():
    return random.randint(0, 4)

def randomStateIndex():
    return random.randint(0, 1)

def arg(vlen, gen):
    """
    Return either a fixed-length vector or a scalar, with values drawn from 'gen'
    """
    vector = random.choice([True, False])
    if vector:
        return [gen() for n in range(vlen)]
    else:
        return gen()


class TestFunctions(unittest.TestCase):

    def test_network_set_neuron(self):
        """ create a simple network and make sure we can get and set parameters
        and state variables """
        a = 0.02
        b = 0.2
        c = -65.0+15.0*0.25
        d = 8.0-6.0*0.25
        v = -65.0
        u = b * v
        sigma = 5.0

        net = IzNetwork()

        # This should only succeed for existing neurons
        self.assertRaises(RuntimeError, net.set_neuron, 0, a, b, c, d, sigma, u, v)

        net.add_neuron(0, a, b, c-0.1, d, sigma, u, v-1.0)

        # Getters should fail if given invalid neuron or parameter
        self.assertRaises(RuntimeError, net.get_neuron_parameter, 1, 0) # neuron
        self.assertRaises(RuntimeError, net.get_neuron_state, 1, 0)     # neuron
        self.assertRaises(RuntimeError, net.get_neuron_parameter, 0, 5) # parameter
        self.assertRaises(RuntimeError, net.get_neuron_state, 0, 2)     # state

        e = 0.1

        # Test setting whole neuron, reading back by parts
        net.set_neuron(0, a-e, b-e, c-e, d-e, sigma-e, u-e, v-e)

        # Since Python uses double precision and NeMo uses single precision
        # internally, the parameters may not be exactly the same after reading
        # back.
        
        places = 5
        self.assertAlmostEqual(net.get_neuron_parameter(0, 0), a-e, places)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 1), b-e, places)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 2), c-e, places)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 3), d-e, places)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 4), sigma-e, places)

        self.assertAlmostEqual(net.get_neuron_state(0, 0), u-e, places)
        self.assertAlmostEqual(net.get_neuron_state(0, 1), v-e, places)

        # Test setting and reading back neuron by parts

        net.set_neuron_parameter(0, 0, a)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 0), a, places)

        net.set_neuron_parameter(0, 1, b)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 1), b, places)

        net.set_neuron_parameter(0, 2, c)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 2), c, places)

        net.set_neuron_parameter(0, 3, d)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 3), d, places)

        net.set_neuron_parameter(0, 4, sigma)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 4), sigma, places)

        net.set_neuron_state(0, 0, u)
        self.assertAlmostEqual(net.get_neuron_state(0, 0), u, places)

        net.set_neuron_state(0, 1, v)
        self.assertAlmostEqual(net.get_neuron_state(0, 1), v, places)

        # Individual setters should fail if given invalid neuron or parameter
        self.assertRaises(RuntimeError, net.set_neuron_parameter, 1, 0, 0.0) # neuron
        self.assertRaises(RuntimeError, net.set_neuron_state, 1, 0, 0.0)     # neuron
        self.assertRaises(RuntimeError, net.set_neuron_parameter, 0, 5, 0.0) # parameter
        self.assertRaises(RuntimeError, net.set_neuron_state, 0, 2, 0.0)     # state

    def check_neuron_function(self, fun, ncount):
        vlen = random.randint(2, ncount)
        a = arg(vlen, random.random)
        b = arg(vlen, random.random)
        c = arg(vlen, random.random)
        d = arg(vlen, random.random)
        u = arg(vlen, random.random)
        v = arg(vlen, random.random)
        s = arg(vlen, random.random)
        vectorized = any(isinstance(x, list) for x in [a, b, c, d, u, v, s])
        if vectorized:
            fun(range(vlen), a, b, c, d, s, u, v)
        else:
            fun(random.randint(0,1000), a, b, c, d, s, u, v)

    def test_add_neuron(self):
        """
        The add_neuron method supports either vector or scalar input. This
        test calls set_synapse in a large number of ways, checking for
        catastrophics failures in the boost::python layer
        """
        for test in range(1000):
            net = IzNetwork()
            self.check_neuron_function(net.add_neuron, ncount=1000)

    def test_set_neuron(self):
        """
        The set_neuron method supports either vector or scalar input. This
        test calls set_synapse in a large number of ways, checking for
        catastrophics failures in the boost::python layer
        """
        net = IzNetwork()
        ncount = 1000
        net.add_neuron(range(ncount), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for test in range(1000):
            self.check_neuron_function(net.set_neuron, ncount=1000)
        sim = nemo.Simulation(net, nemo.Configuration())
        for test in range(1000):
            self.check_neuron_function(sim.set_neuron, ncount=1000)

    def check_set_neuron_vector(self, obj, pop):
        """
        Test vector/scalar forms of set_neuron for either network or simulation

        pop -- list of neuron
        """
        for test in range(1000):
            vlen = random.randint(2, 100)
            # We need uniqe neurons here, for defined behaviour
            vector = random.choice([True, False])
            if vector:
                neuron = random.sample(pop, vlen)
                value = [random.random() for n in neuron]
            else:
                neuron = random.choice(pop)
                value = random.random()

            def assertListsAlmostEqual(value, ret):
                if vector:
                    self.assertEqual(vlen, len(ret))
                    self.assertEqual(vlen, len(value))
                    self.assertEqual(vlen, len(neuron))
                    [self.assertAlmostEqual(a, b, 5) for (a,b) in zip(value, ret)]
                else:
                    self.assertAlmostEqual(value, ret, 5)

            # check neuron parameter
            param = randomParameterIndex()
            obj.set_neuron_parameter(neuron, param, value)
            ret = obj.get_neuron_parameter(neuron, param)
            assertListsAlmostEqual(value, ret)

            # check neuron state
            var = randomStateIndex()
            obj.set_neuron_state(neuron, var, value)
            ret = obj.get_neuron_state(neuron, var)
            assertListsAlmostEqual(value, ret)


    def test_network_set_neuron_vector(self):
        """
        Test for failures in vector/scalar form of set_neuron

        The set_neuron_parameter methods supports either vector or scalar
        input. This test calls this function in a large number of ways,
        checking for catastrophics failures in the boost::python layer
        """
        net = IzNetwork()
        pop = range(1000)
        for n in pop:
            net.add_neuron(n, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.check_set_neuron_vector(net, pop)

    def test_sim_set_neuron_vector(self):
        """
        Test for failures in vector/scalar form of set_neuron

        The set_neuron_parameter methods supports either vector or scalar
        input. This test calls this function in a large number of ways,
        checking for catastrophics failures in the boost::python layer
        """
        net = IzNetwork()
        conf = nemo.Configuration()
        pop = range(1000)
        for n in pop:
            net.add_neuron(n, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sim = nemo.Simulation(net, conf)
        self.check_set_neuron_vector(sim, pop)

    def simple_network(self):
        net = IzNetwork()
        net.add_neuron(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        net.add_neuron(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        net.add_synapse(0, 1, 1, 5.0, False)
        net.add_synapse(1, 0, 1, 5.0, False)
        return (net, nemo.Simulation(net, nemo.Configuration()))

    def test_get_neuron_scalar(self):
        """
        Test that singleton arguments to neuron getters work as either scalar
        or singleton list.
        """
        def check(x):
            x.get_neuron_state([0], 0)
            x.get_neuron_state(0, 0)
            x.get_neuron_parameter([0], 0)
            x.get_neuron_parameter(0, 0)
        (net, sim) = self.simple_network()
        check(net)
        check(sim)

    def test_set_neuron_scalar(self):
        """
        Test that singleton arguments to neuron setters work as either scalar
        or singleton list.
        """
        def check(x):
            x.set_neuron_state([0], 0, [0])
            x.set_neuron_state(0, 0, 0)
            x.set_neuron_parameter([0], 0, [0])
            x.set_neuron_parameter(0, 0, 0)
        (net, sim) = self.simple_network()
        check(net)
        check(sim)

    def test_get_synapse_scalar(self):
        """
        Test that singleton arguments to synapse getters work as either scalar
        or singleton list.
        """
        def check(x):
            x.get_synapse_source(0)
            x.get_synapse_source([0])
            x.get_synapse_target(0)
            x.get_synapse_target([0])
            x.get_synapse_delay(0)
            x.get_synapse_delay([0])
            x.get_synapse_weight(0)
            x.get_synapse_weight([0])
            x.get_synapse_plastic(0)
            x.get_synapse_plastic([0])
        (net, sim) = self.simple_network()
        check(net)
        check(sim)

    def test_add_synapse(self):
        """
        The add_synapse method supports either vector or scalar input. This
        test calls set_synapse in a large number of ways, checking for
        catastrophics failures in the boost::python layer
        """
        net = IzNetwork()
        for test in range(1000):
            vlen = random.randint(2, 500)
            source = arg(vlen, randomSource)
            target = arg(vlen, randomTarget)
            delay = arg(vlen, randomDelay)
            weight = arg(vlen, randomWeight)
            plastic = arg(vlen, randomPlastic)
            ids = net.add_synapse(source, target, delay, weight, plastic)
            vectorized = any(isinstance(n, list) for n in [source, target, delay, weight, plastic])
            if vectorized:
                self.assertTrue(isinstance(ids, list))
                self.assertEqual(len(ids), vlen)
            else:
                self.assertFalse(isinstance(ids, list))


    def test_get_synapses_from_unconnected(self):
        net = IzNetwork()
        net.add_neuron(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(len(net.get_synapses_from(0)), 0)
        sim = nemo.Simulation(net, nemo.Configuration())
        self.assertEqual(len(sim.get_synapses_from(0)), 0)

    def test_get_synapse(self):
        """
        Test scalar and vector form of synapse getters

        Synapse getters have both scalar and vector forms. To test these,
        construct a network with fixed connectivity where all synapse
        properties are functions of the source and target, then read back and
        verify that the values are as expected.
        """

        def delay(source, target):
            return 1 + ((source + target) % 20)

        def plastic(source, target):
            return (source + target) % 1 == 0

        def weight(source, target):
            return float(source) + float(target)

        ncount = 100

        net = IzNetwork()
        for src in range(ncount):
            net.add_neuron(src, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for tgt in range(src+1):
                net.add_synapse(src, tgt, delay(src, tgt), weight(src, tgt), plastic(src, tgt))

        conf = nemo.Configuration()
        sim = nemo.Simulation(net, conf)

        def check_scalar(x, known_source, sid, source, target):
            self.assertEqual(known_source, source)
            self.assertEqual(x.get_synapse_delay(sid), delay(source, target))
            self.assertEqual(x.get_synapse_weight(sid), weight(source, target))
            self.assertEqual(x.get_synapse_plastic(sid), plastic(source, target))

        def check(x):
            for src in range(ncount):
                all_synapses = x.get_synapses_from(src)
                # read a random number of these out-of-order
                n_queried = random.randint(1, len(all_synapses))
                queried = random.sample(all_synapses, n_queried)
                if len(queried) == 1:
                    queried = queried[0]
                sources = x.get_synapse_source(queried)
                targets = x.get_synapse_target(queried)
                if n_queried == 1:
                    check_scalar(x, src, queried, sources, targets)
                else:
                    for (sid, qsrc, tgt) in zip(queried, sources, targets):
                        check_scalar(x, src, sid, qsrc, tgt)

        def check_iterator(x):
            # Make synapse getter can deal with the iterator returned by the
            # the synapse query
            for src in range(ncount):
                srcs = x.get_synapse_source(x.get_synapses_from(src))

        check(net)
        check(sim)
        check_iterator(net)
        check_iterator(sim)


if __name__ == '__main__':
    unittest.main()
