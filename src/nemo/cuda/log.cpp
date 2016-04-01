/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <nemo/config.h>
#if defined(NEMO_CUDA_DEBUG_TRACE) && defined(HAVE_CUPRINTF)
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuPrintf.cuh>
#include "exception.hpp"
#endif

void
initLog()
{
#if defined(NEMO_CUDA_DEBUG_TRACE) && defined(HAVE_CUPRINTF)
	CUDA_SAFE_CALL(cudaPrintfInit(1<<24));
#endif
}


void
flushLog()
{
#if defined(NEMO_CUDA_DEBUG_TRACE) && defined(HAVE_CUPRINTF)
	CUDA_SAFE_CALL(cudaPrintfDisplay(stdout, true));
#endif
}


void
endLog()
{
#if defined(NEMO_CUDA_DEBUG_TRACE) && defined(HAVE_CUPRINTF)
	cudaPrintfEnd();
#endif
}
