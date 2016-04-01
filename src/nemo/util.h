#ifndef UTIL_H
#define UTIL_H

#include <math.h>
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

#define IS_POWER_OF_TWO(v) (!((v) & ((v) - 1)) && (v))

#define DIV_CEIL(a, b) (((a) + (b) - 1) / (b))

/* Round a up to the nearest multiple of b */
#define ALIGN(a, b) (b) * DIV_CEIL((a), (b))

#define SWAP(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))

#define MASK(bits) (~(~0 << (bits)))

#endif
