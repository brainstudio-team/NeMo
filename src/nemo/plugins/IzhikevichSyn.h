#ifndef NEMO_PLUGINS_IZHIKEVICH2007_H
#define NEMO_PLUGINS_IZHIKEVICH2007_H

/* Common symbol names for Izhikevich kernel */

// Parameters
#define PARAM_A 0
#define PARAM_B 1
#define PARAM_C 2
#define PARAM_D 3
#define PARAM_VPEAK 4
#define PARAM_SIGMA 5 // for gaussian RNG
#define PARAM_TauMinusDtOverTau_Exc 6
#define PARAM_G_Exc 7
#define PARAM_E_Exc 8
#define PARAM_TauMinusDtOverTau_Inh 9
#define PARAM_G_Inh 10
#define PARAM_E_Inh 11

// State variables
#define STATE_U 0
#define STATE_V 1
#define STATE_Ge 2
#define STATE_Gi 3

#endif
