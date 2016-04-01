#ifndef FIXED_POINT_HPP
#define FIXED_POINT_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <nemo/internal_types.h>
#include <nemo/config.h>


/* Convert floating point to fixed-point */
NEMO_BASE_DLL_PUBLIC
fix_t
fx_toFix(float f, unsigned fractionalBits);


/* Convert fixed-point to floating point */
NEMO_BASE_DLL_PUBLIC
float
fx_toFloat(fix_t v, unsigned fractionalBits);



/* Convert wide fixed-point to floating point */
NEMO_BASE_DLL_PUBLIC
float
wfx_toFloat(wfix_t v, unsigned fractionalBits);



inline
#ifdef __CUDACC__
__host__ __device__
#endif
fix_t
fx_mul(fix_t a, fix_t b, unsigned fractionalBits)
{
	int64_t r = int64_t(a) * int64_t(b);
	return fix_t(r >> fractionalBits);
}


const fix_t fx_min =   fix_t(1) << 31;
const fix_t fx_max = ~(fix_t(1) << 31);


#endif
