#ifndef NEMO_CUDA_DOUBLE_BUFFER_CU
#define NEMO_CUDA_DOUBLE_BUFFER_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file double_buffer.cu Basic indexing for double buffers
 *
 * Data structures which are used for communication between different
 * partitions need to be double buffered so as to avoid race conditions.  These
 * functions return the double buffer index (0 or 1) for the given cycle, for
 * either the read or write part of the buffer */


/*! \return read buffer index (0 or 1) for current \a cycle */
__device__
unsigned
readBuffer(unsigned cycle)
{
    return (cycle & 0x1) ^ 0x1;
}


/*! \return write buffer index (0 or 1) for current \a cycle */
__device__
unsigned
writeBuffer(unsigned cycle)
{
    return cycle & 0x1;
}

#endif
