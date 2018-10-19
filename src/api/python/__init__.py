"""
The NeMo spiking neural network simulator
=========================================

NeMo is a fast spiking neural network simulator which can run on CUDA-enabled
GPUs. The ``nemo`` module provides an object-oriented interface to the C++
class library. The interface is based around three classes: Network,
Configuration, and Simulation.

Basic usage is as follows:

1. create a configuration;
2. create and populate a network by adding individual neurons and synapses; and
3. create a simulation from the configuration and network object, and run the
   simulation providing stimulus and reading outputs as appropriate

More details can be found in the documentation for each of these classes.

The following example shows how a small network of 1000 neurons is created and
simulated for one second::

    import nemo
    import random

    net = nemo.Network()
    iz = net.add_neuron_type('Izhikevich')

    # Excitatory neurons
    for nidx in range(800):
        r = random.random()**2
        c = -65.0+15*r
        d = 8.0 - 6.0*r
        net.add_neuron(iz, nidx, 0.02, 0.2, c, d, 5.0, 0.2*c, c);
        targets = range(1000)
        weights = [0.5*random.random() for tgt in targets]
        net.add_synapse(nidx, targets, 1, weights, False);

    # Inhibitory neurons
    for n in range(200):
        nidx = 800 + n
        r = random.random()
        a = 0.02+0.08*r
        b = 0.25-0.05*r
        c = -65.0
        net.add_neuron(iz, nidx, a, b, c, 2.0, 2.0, b*c, c)
        targets = range(1000)
        weights = [-random.random() for tgt in targets]
        net.add_synapse(nidx, targets, 1, weights, False);

    conf = nemo.Configuration()
    sim = nemo.Simulation(net, conf)
    for t in range(1000):
        fired = sim.step()
        print t, ":", fired

There is also a higher-level interface using the PyNN common simulator
interface.  PyNN is a large separate project which is documented in full
elsewhere.
"""

import sys
import warnings


# from _nemo import *

# TODO: Is init() necesary?
# init()

from wrappers import *

# Simulation.step = step

# Old synapse getters. These were originally vector-only, and were only exposed
# for the simulation class. The new getters support both vector and scalar
# forms, and use more consistent naming.

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func

__version__ = '@NEMO_VERSION@'
__nemo_path__='@CMAKE_INSTALL_PREFIX@/@INSTALL_DIR@'

"""The above replacements are not working so hard coding 
	path need for Brain-Studio below."""
__nemo_path__='/usr/local'
__all__ = ['Network', 'Simulation', 'Configuration']
