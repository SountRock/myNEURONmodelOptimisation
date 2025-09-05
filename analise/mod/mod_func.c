#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _currentGauss_reg();
extern void _izhi2003a_reg();
extern void _izhi2007b_reg();
extern void _izhi2007bmod_reg();
extern void _tmgsyn_reg();
extern void _vecevent_reg();

void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," currentGauss.mod");
fprintf(stderr," izhi2003a.mod");
fprintf(stderr," izhi2007b.mod");
fprintf(stderr," izhi2007bmod.mod");
fprintf(stderr," tmgsyn.mod");
fprintf(stderr," vecevent.mod");
fprintf(stderr, "\n");
    }
_currentGauss_reg();
_izhi2003a_reg();
_izhi2007b_reg();
_izhi2007bmod_reg();
_tmgsyn_reg();
_vecevent_reg();
}
