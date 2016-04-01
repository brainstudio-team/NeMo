#ifndef NEMO_PLUGINS_IZHIKEVICH2007_H
#define NEMO_PLUGINS_IZHIKEVICH2007_H

/* Common symbol names for Izhikevich kernel */

// Parameters
#define PARAM_2007A 0
#define PARAM_2007B 1
#define PARAM_2007C 2
#define PARAM_2007D 3
#define PARAM_2007VPEAK 4
#define PARAM_2007VR 5
#define PARAM_2007VT 6
#define PARAM_2007K 7
#define PARAM_2007CCAP 8
#define PARAM_2007SIGMA 9 // for gaussian RNG
#define PARAM_2007D1 10
#define PARAM_2007D2 11
#define PARAM_2007TauMinusDtOverTau_Exc 12
#define PARAM_2007G_Exc 13
#define PARAM_2007E_Exc 14
#define PARAM_2007TauMinusDtOverTau_Inh 15
#define PARAM_2007G_Inh 16
#define PARAM_2007E_Inh 17



// State variables
#define STATE_2007U 0
#define STATE_2007V 1
#define STATE_2007Ge 2
#define STATE_2007Gi 3

// Constants from Humphries etal (2009), Capturing dopaminergic...
#define MSN_BETA1 6.3
#define MSN_BETA2 0.215
#define MSN_K 0.0289
#define MSN_L 0.331
#define MSN_ALPHA 0.032

#endif
