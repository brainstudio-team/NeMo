#ifndef NEMO_BITOPS_H
#define NEMO_BITOPS_H

#include <limits.h>
#include <cmath>
#include <nemo/config.h>

/* Compute leading zeros for type T which should have B bits.
 *
 * This could be done faster using one of the other bit-twiddling hacks from
 * http://graphics.stanford.edu/~seander/bithacks.html */
template<typename T, int B>
int
clzN(T v)
{
	unsigned r = 0;
	while (v >>= 1) {
		r++;
	}
	return (B - 1) - r;
}



/* Count leading/trailing zeros in 64-bit word. Unfortunately the gcc builtins
 * to deal with this are not explicitly 64 bit. Instead it is defined for long
 * long. In C99 this is required to be /at least/ 64 bits. However, we require
 * it to be /exactly/ 64 bits. */
#if LLONG_MAX == 9223372036854775807 && defined(HAVE_BUILTIN_CLZLL)
inline int clz64(uint64_t val) { return __builtin_clzll(val); }
#else
inline int clz64(uint64_t val) { return clzN<uint64_t, 64>(val); }
#endif


/* Ditto for 32 bits */
#if UINT_MAX == 4294967295U && defined(HAVE_BUILTIN_CLZ)
inline int clz32(uint32_t val) { return __builtin_clz(val); }
#else
inline int clz32(uint32_t val) { return clzN<uint32_t, 32>(val); }
#endif // LONG_MAX


/* Compute trailing zeros for type T which should have B bits.
 *
 * This is taken from "bit-twiddling hacks", which also has
 * some faster methods
 * http://graphics.stanford.edu/~seander/bithacks.html */
template<typename T, int B>
int
ctzN(T v)
{
	if(v) {
		int c;
		v = (v ^ (v - 1)) >> 1;  // Set v's trailing 0s to 1s and zero rest
		for (c = 0; v; c++) {
			v >>= 1;
		}
		return c;
	}
	else {
		return B;
	}
}


#ifdef HAVE_BUILTIN_CTZLL
/* Count trailing zeros. This should work even if long long is greater than
 * 64-bit. The uint64_t will be safely extended to the appropriate length */
inline int ctz64(uint64_t val) { return __builtin_ctzll(val); }
#else
inline int ctz64(uint64_t val) { return ctzN<uint64_t, 64>(val); }
#endif




/* compute the next highest power of 2 of 32-bit v. From "bit-twiddling hacks".  */
inline
uint32_t
ceilPowerOfTwo(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}


#ifndef HAVE_LOG2

/* log2 is part of the C99 spec, so Microsoft support is patchy */
inline double log2(double n) { return log(n) / log(double(2)); }
inline float log2f(float n) { return logf(n) / logf(2.0f); }

#endif



#endif // BITOPS_H
