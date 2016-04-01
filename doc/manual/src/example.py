import nemo
from numpy import random as rn

net = nemo.Network()
iz  = net.add_neuron_type('Izhikevich')
Ne  = 800
Ni  = 200
N   = Ne + Ni

# Excitatory neurons
re = rn.random(Ne)**2
c  = -65.0 + 15*re
d  = 8.0 - 6.0*re
paramDictEx = {'a': 0.02, 'b': 0.2, 'c': list(c),
               'd': list(d), 'sigma': 5.0}
stateDictEx = {'v': list(c), 'u': list(0.2*c)}
net.add_neuron(iz, range(Ne), paramDictEx, stateDictEx)

# Inhibitory neurons
ri = rn.random(Ni)
a  = list(0.02 + 0.08*ri)
b  = list(0.25 - 0.05*ri)
c  = -65.0
paramDictIn = {'a': a, 'b': b, 'c': c,
               'd': 2.0, 'sigma': 2.0}
stateDictIn = {'v': c, 'u': 0.2*c}
net.add_neuron(iz, range(Ne, N), paramDictIn, stateDictIn)

# Excitatory connections
for nidx in range(Ne):
    targets = range(N)
    weights = list(0.5*rn.random(N))
    delay = 1
    net.add_synapse(nidx, targets, delay, weights, False)

# Inhibitory connections
for nidx in range(Ne, N):
    targets = range(N)
    weights = list(-1*rn.random(N))
    delay = 1
    net.add_synapse(nidx, targets, delay, weights, False)

conf = nemo.Configuration()
sim = nemo.Simulation(net, conf)

# Run simulation and print firings
for t in range(1000):
    fired = sim.step()
    print t, ":", fired

