#ifndef NEMO_CUDA_EXCEPTION_HPP
#define NEMO_CUDA_EXCEPTION_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <sstream>

#include <cuda_runtime.h>

#include <nemo/exception.hpp>

#ifdef NDEBUG
#define LINE_INFO ""
#else
#define LINE_INFO "Cuda error in file " << __FILE__ << " in line " << __LINE__ << ": "
#endif

#define CUDA_SAFE_CALL(call) {                                             \
    cudaError err = call;                                                  \
    if(cudaSuccess != err) {                                               \
        std::ostringstream msg;                                            \
        msg << LINE_INFO << cudaGetErrorString(err);                       \
        throw nemo::exception(NEMO_CUDA_ERROR, msg.str().c_str());         \
    } }


#endif
