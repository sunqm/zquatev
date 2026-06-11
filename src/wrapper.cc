#include "zquatev.h"
#include "f77.h"

using namespace std;

extern "C" {
int eigh(complex<double>* D, double* eig, int n2, int ld2)
{
    return ts::zquatev(n2, D, ld2, eig);
}
}
