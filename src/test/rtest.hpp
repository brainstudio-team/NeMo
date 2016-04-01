#ifndef NEMO_TEST_RTEST_HPP
#define NEMO_TEST_RTEST_HPP

/*! Run a fixed torus network for regression testing purposes
 *
 * \param creating
 * 		If true, create test data and (over)write this to four named files.
 * 		If false, verify that simulation results (firing only) exactly
 * 		matches the data found in the data files previously created by this
 * 		function in 'create' mode.
 */ 	
void runTorus(bool creating);
void runKuramoto(bool creating);

#endif
