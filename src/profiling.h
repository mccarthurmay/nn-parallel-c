#ifndef PROFILING_H
#define PROFILING_H


/*
Build normally with ./build.sh (no TAU, no overhead), or with tau_cc.sh and
-DTAU_ENABLED to collect profiles. See tau_run.sbatch.
*/

#ifdef TAU_ENABLED

#include <TAU.h>

#define PROF_INIT(argc, argv)   do { TAU_PROFILE_INIT(argc, argv); \
                                     TAU_PROFILE_SET_NODE(0); } while (0)
#define PROF_PHASE(var, name)   TAU_PHASE_CREATE_STATIC(var, name, "", TAU_USER)
#define PROF_PHASE_START(var)   TAU_PHASE_START(var)
#define PROF_PHASE_STOP(var)    TAU_PHASE_STOP(var)
#define PROF_TIMER(var, name)   TAU_PROFILE_TIMER(var, name, "", TAU_USER)
#define PROF_START(var)         TAU_PROFILE_START(var)
#define PROF_STOP(var)          TAU_PROFILE_STOP(var)

#else

#define PROF_INIT(argc, argv)   ((void)0)
#define PROF_PHASE(var, name)   ((void)0)
#define PROF_PHASE_START(var)   ((void)0)
#define PROF_PHASE_STOP(var)    ((void)0)
#define PROF_TIMER(var, name)   ((void)0)
#define PROF_START(var)         ((void)0)
#define PROF_STOP(var)          ((void)0)

#endif

#endif
