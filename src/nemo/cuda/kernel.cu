//! \file kernel.cu

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "kernel.cu_h"
#include "log.cu_h"

#include "delays.cu"
#include "device_assert.cu"
#include "bitvector.cu"
#include "double_buffer.cu"
#include "outgoing.cu"
#include "globalQueue.cu"
#include "nvector.cu"
#include "rcm.cu"

#include "gather.cu"
#include "scatter.cu"
#include "stdp.cu"
#include "applySTDP.cu"

#ifdef NEMO_BRIAN_ENABLED
#	include "compact.cu"
#endif
