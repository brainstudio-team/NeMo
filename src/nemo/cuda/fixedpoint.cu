#ifndef FIXED_POINT_CU
#define FIXED_POINT_CU

/*! \file fixedpoint.cu Routines for fixed point manipulation
 *
 * All functions are prefixed \c fx
 */

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <nemo/fixedpoint.hpp>

#include "types.h"
#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
#	include "log.cu_h"
#endif

#include "device_assert.cu"
#include "bitvector.cu"

#define FX_SIGN_BIT 0x80000000


__device__
float
fx_tofloat(fix_t v, unsigned scale)
{
	//! \todo any way to avoid division here. Perhaps precompute the fraction here?
	//! \todo check if it makes any difference to use 1<<c here instead
	return float(v) / scale;
}



/*! \return saturated value (i.e. maximally positive or maximally negative)
 * with sign as indicated by \a negative */
__device__
fix_t
fx_saturate(bool negative)
{
	return negative ? fx_max : fx_min;
}



/*! Cast fixed-point to floating point, with saturation as indicated by the
 * flags \a overflow and \a negative */
__device__
float
fx_saturatedTofloat(fix_t v, bool overflow, bool negative, unsigned scale)
{
	//! \todo any way to avoid division here? Perhaps precompute the fraction here?
	//! \todo check if it makes any difference to use 1<<c here instead
	return float(overflow ? fx_saturate(negative) : v) / scale;
}


/*! Add atomically to shared memory fixed-point value, returning true if an
 * overflow occurred */
__device__
bool
fx_atomicAdd(fix_t* s_a, fix_t b)
{
	fix_t a = atomicAdd(s_a, b);
	/* It seems it's not possible to access the carry bit, even in PTX code.
	 * (from PTX manual under ADDC). We therefore have to manually check for
	 * overflow. */
	fix_t aSign = a & FX_SIGN_BIT;
	fix_t bSign = b & FX_SIGN_BIT;
	/* We cannot rely on *s_a here, due to race conditions */
	fix_t outputSign = (a+b) & FX_SIGN_BIT;
	/*! \note could use 64-bit addition here for overflow detection */
	return (aSign == bSign) && (aSign != outputSign);
}



__device__
fix_t
fx_isNegative(fix_t v)
{
	return v & FX_SIGN_BIT;
}



/* Convert shared-memory array from fixed-point to floating point format and
 * perform fixed-point saturation. The conversion can be done in-place, i.e.
 * the fixed-point input and floating-point outputs arrays can be the same. */
__device__
void
fx_arrSaturatedToFloat(
		uint32_t* s_overflow, // bit-vector
		bool negative,
		fix_t* s_fix,
		float* s_float,
		unsigned scale)
{
	/* If any accumulators overflow, clamp to max positive or minimum value */
	for(unsigned nbase=0; nbase < MAX_PARTITION_SIZE; nbase += THREADS_PER_BLOCK) {
		unsigned nidx = nbase + threadIdx.x;
#ifndef NEMO_WEIGHT_FIXED_POINT_SATURATION
		s_float[nidx] = fx_tofloat(s_fix[nidx], scale);
#else
		bool overflow = bv_isSet(nidx, s_overflow);
		s_float[nidx] = fx_saturatedTofloat(s_fix[nidx], overflow, negative, scale);
#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
		if(overflow) {
			DEBUG_MSG("c%u p?+%un%u input current overflow. Saturated to %+f (%08x)\n",
					s_cycle, CURRENT_PARTITION, nidx,
					s_float[nidx], s_fix[nidx]);
		}
#endif
#endif
	}
	__syncthreads();
}





#endif
