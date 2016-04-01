#!/usr/bin/python

"""Plot connectivity matrix 

Usage: connectivity [-i <file>] [-o <file>]

Options:

 --input -i <file>    Read data from file rather than stdin
 --output -o <file>   Write output to postscript file rather than to GUI. The
					  file suffix specifies the format which can be either ps
                      or eps.
 --title <string>     Use specified title instead of default ("Neuron
                      Connectivity")
 --notitle            Don't print a title
 --nolegend           Don't print a legend
 --help -h            Print this message

By default data are read from stdin. The first line of the input should contain
a single value which specifies the number of neurons to plot. The rest of the
input contains one synapse per line in the following format:

<presynaptic index>  <postsynaptic index> <value>

Value can be anything of interest (connection strength, delay, etc), and is
interpreted as a floating point value.
"""

import sys
import string
import getopt
import matplotlib.numerix.ma as M
import matplotlib
if "--output=" in sys.argv[1:] or "-o" in sys.argv[1:] :
	matplotlib.use('PS')
import pylab
import numpy


def parse_header(infile):
	header = infile.readline()
	try:
		size = int(header)
	except ValueError:
		print "Malformed header: %s" % header
		return
	except Exception, e:
		raise e
	return size


def parse_data(infile, size):
	""" Populate matrix with data from infile """
	m = numpy.zeros((size,size))
	for (lineno, line) in enumerate(infile.readlines()):
		try :
			(pre, post, val) = string.split(line)
		except ValueError :
			print "Malformed data in line %s: %s" % (lineno, line)
			return 
		except Exception, e:
			raise e
		m[(int(pre), int(post))] = float(val)
	return M.masked_where(m == 0, m)


def color_map():
    """ Return colour map using blues for negative values and reds for positive
        values """
    cdict = {
        'red': ((0.0, 0.0, 0.0),
                (0.5, 0.0, 0.1),
                (1.0, 1.0, 1.0)),
        'green': ((0.0, 0.3, 0.3),
                (0.5, 0.0, 0.0),
                (1.0, 0.3, 0.3)),
        'blue': ((0.0, 1.0, 1.0),
                (0.5, 0.1, 0.0),
                (1.0, 0.0, 0.0))}
    colors = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)
    colors.set_bad((0.0, 0.0, 0.0)) # ignore 0 values, i.e. no synapse
    return colors



def plot(m, outfile=None, title=None, use_legend=False):
    pylab.figure(1)
    colors = color_map()
    max_elem = abs(m).max()
    pylab.imshow(m, cmap=colors, interpolation='nearest', vmin=-max_elem, vmax=max_elem) 
    if use_legend:
        pylab.colorbar()
    if title:
        pylab.title(title)
    if outfile:
        pylab.savefig(outfile)
    else:
        pylab.show()


def plot_file(infile, outfile, title, use_legend):
    size = parse_header(infile)
    m = parse_data(infile, size)
    plot(m, outfile, title, use_legend)


if __name__ == "__main__":
	infile = sys.stdin
	outfile = None
	title = "Neuron connectivity"
	use_legend = True
	try:
		opts, args = getopt.getopt(sys.argv[1:], 
				"hi:o:", 
				["help", "input=", "output=", "title=", "notitle", "nolegend"])
	except getopt.GetoptError:
		print __doc__
		sys.exit(2)

	for opt, arg in opts:
		if opt in ("-h" ,"--help"):
			print __doc__
			sys.exit()
		elif opt in ("-i", "--input"):
			infile = open(arg, 'r')
		elif opt in ("-o", "--output"):
			outfile = arg
		elif opt in ("--title"):
			title = arg
		elif opt in ("--notitle"):
			title = None
		elif opt in ("--nolegend"):
			use_legend = False

	plot_file(infile, outfile, title, use_legend)
