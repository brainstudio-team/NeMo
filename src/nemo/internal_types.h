#ifndef NEMO_INTERNAL_TYPES_H
#define NEMO_INTERNAL_TYPES_H

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stddef.h>
#include "types.h"

typedef int32_t fix_t;
typedef int64_t wfix_t;  // 'wide' fixed-point type
typedef unsigned nidx_t; // neuron index
typedef unsigned sidx_t; // synapse index
typedef unsigned delay_t;

typedef uint32_t id32_t;
typedef uint64_t id64_t;

#endif
