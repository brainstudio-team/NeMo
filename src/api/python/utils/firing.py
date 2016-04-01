#!/usr/bin/python

"""Plot firing data

Options:
--help -h            produce this message
--input -i <file>    read input from file rather than stdin
--output -o <file>   write output to a postscript file rather than to GUI.
--start <n>          only consider data from simulation cycle n onwards 
                     (default: 0)
--duration <n>       only consider data for n cycles after start cycle
                     (default: until end)

Data format:
The input data is in the form of whitespace-separated time-neuron pairs 
"""

import sys
import getopt

import matplotlib
if "--output=" in sys.argv[1:] or "-o" in sys.argv[1:] :
    matplotlib.use('PS')
from pylab import *


def plot_data(t, nn, start=0, duration=0, outfile=None):
    """ Plot and display firing data """ 
    plot(t, nn, 'b.')
    xlabel('time (ms)')
    ylabel('neuron')
    title('Neuron firings')
    xlim(xmin=start)
    if duration > 0:
        xlim(xmax=start+duration)
    if outfile:
        print "Plotting to file", outfile
        savefig(outfile)
    else:
        print "Plotting to GUI"
        show()


def plot_file(infile, start=0, duration=0, outfile=None):
    """ Plot and display firing data from file """
    firing = load(infile)
    t = firing[:,0]
    nn = firing[:,1]
    plot_data(t, nn, start, duration, outfile)


def plot_stdin_hs(start=0, duration=0, outfile=None):
    """ Plot and display firing data from stdin in haskell list format:
        one line per cycle with a list of spiking neurons. E.g.:
        [1,49,50,980]
        []
        [7]
    """
	print "plot_stdin_hs"

    ts = []
    nns = []
    # TODO: do this by chunks in order to deal with large files, see e.g.
    # http://oreilly.com/catalog/lpython/chapter/ch09.html
    for (cycle, line) in enumerate(sys.stdin.readlines()):
        if line.strip('[]\n') == '':
            continue
        try :
            nn = map(int, line.strip('[]\n').split(','))
        except ValueError :
            print "Malformed data in line %s: %s" % (cycle, line)
            return 
        except Exception, e:
            raise e
        if cycle>=start and (duration<=0 or cycle<=start+duration):
            for n in nn :
                ts.append(cycle)
                nns.append(n)
    plot_data(ts, nns, start, duration, outfile)


def plot_stdin_raw(start=0, duration=0, outfile=None):
    """ Plot and display firing data from stdin """
    ts = []
    nns = []
    # TODO: do this by chunks in order to deal with large files, see e.g.
    # http://oreilly.com/catalog/lpython/chapter/ch09.html
    for (lineno, line) in enumerate(sys.stdin.readlines()):
        try :
            (t, nn) = map(int, string.split(line))
        except ValueError :
            print "Malformed data in line %s: %s" % (lineno, line)
            return 
        except Exception, e:
            raise e
        if t>=start and (duration<=0 or t<=start+duration):
            ts.append(t)
            nns.append(nn)
    plot_data(ts, nns, start, duration, outfile)


def main(args):

    start = 0
    duration = 0
    infile = None
    outfile = None

    try:
        opts, args = getopt.getopt(args, 
                "hi:o:s:d:", 
                ["help", "input=", "output=", "start=", "duration="])
    except getopt.GetoptError:
        print __doc__
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h" ,"--help"):
            print __doc__
            sys.exit()
        elif opt in ("-i", "--input"):
            infile = arg
        elif opt in ("-o", "--output"):
            outfile = arg
        elif opt in ("-s", "--start"):
            start = int(arg)
        elif opt in ("-d", "--duration"):
            duration = int(arg)

    if infile:
        plot_file(infile, start, duration, outfile)
    else:
        plot_stdin_hs(start, duration, outfile)
        

if __name__ == "__main__":
    main(sys.argv[1:])
