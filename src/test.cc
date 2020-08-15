//
// ZQUATEV: Diagonalization of quaternionic matrices
// File   : test.cc
// Copyright (c) 2013, Toru Shiozaki (shiozaki@northwestern.edu)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The views and conclusions contained in the software and documentation are those
// of the authors and should not be interpreted as representing official policies,
// either expressed or implied, of the FreeBSD Project.
//

#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include "zquatev.h"
#include "f77.h"

using namespace std;

int main(int argc, char * argv[]) {

  const int n = (argc>1) ? atoi(argv[1]) : 1000;
  const int n2 = n*2;
  const int nld = n*2;
  cout << "n=" << n << endl;

  // ZL@20200211: use smart pointer & array for matrix
  unique_ptr<complex<double>[]> A(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> B(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> C(new complex<double>[nld*n2]);
  unique_ptr<complex<double>[]> D(new complex<double>[nld*n2]);
  unique_ptr<complex<double>[]> E(new complex<double>[nld*n2]);

  unique_ptr<double[]> eig(new double[n2]);
  unique_ptr<double[]> eig2(new double[n2]);

  // some random matrices A (Hermite) and B (anti-Hermite)
  srand(32);
  for (int i = 0; i != n; ++i) {
    for (int j = 0; j <= i; ++j) {
      const array<double,4> t = {{rand()%10000 * 0.0001, rand()%10000 * 0.0001,
                                  rand()%10000 * 0.0001, rand()%10000 * 0.0001}};
      A[j+n*i] = j == i ? t[0] : complex<double>(t[0], t[1]); // ZL: A[i,j]
      A[i+n*j] = conj(A[j+n*i]);
      B[j+n*i] = j == i ? 0.0 : complex<double>(t[2], t[3]);
      B[i+n*j] = -B[j+n*i];
    }
  }

  //
  // [A -B*]
  // [B  A*] nld=2n
  //
  for (int i = 0; i != n; ++i) {
    for (int j = 0; j != n; ++j) {
      C[j+nld*i] = A[j+n*i];
      C[n+j+nld*(n+i)] = conj(A[j+n*i]);
      C[j+nld*(n+i)] = B[j+n*i];
      C[n+j+nld*(i)] = -conj(B[j+n*i]);
    }
  }
  copy_n(C.get(), nld*n2, D.get());
  copy_n(C.get(), nld*n2, E.get());

  cout << endl;
  cout << " **** using zheev **** " << endl;
  auto time0 = chrono::high_resolution_clock::now();
  {
    int lwork = -1;
    unique_ptr<double[]> rwork(new double[6*n-2]);
    complex<double> lworkopt;
    int info;
    // ZL: ZHEEV will modify the data
    // SUBROUTINE ZHEEV( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, RWORK, INFO )
    /* 
       LWORK   (input) INTEGER
          The length of the array WORK.  LWORK >= max(1,2*N-1).
          For optimal efficiency, LWORK >= (NB+1)*N,
          where NB is the blocksize for ZHETRD returned by ILAENV.
 
          If LWORK = -1, then a workspace query is assumed; the routine
          only calculates the optimal size of the WORK array, returns
          this value as the first entry of the WORK array, and no error
          message related to LWORK is issued by XERBLA.
    */
    zheev_("V", "U", &n2, C.get(), &nld, eig.get(), &lworkopt, &lwork, rwork.get(), &info);
    lwork = static_cast<int>(lworkopt.real());
    unique_ptr<complex<double>[]> work(new complex<double>[lwork]);
    zheev_("V", "U", &n2, C.get(), &nld, eig.get(), work.get(), &lwork, rwork.get(), &info);
    if (info) throw runtime_error("zheev failed");
  }

  cout << " **** using zquartev **** " << endl;
  auto time1 = chrono::high_resolution_clock::now();
  {
    // kernel	  
    const int info2 = ts::zquatev(n2, D.get(), nld, eig2.get());
    if (info2) throw runtime_error("zquatev failed");
  }
  auto time2 = chrono::high_resolution_clock::now();

  // Check: 
  double maxdev = 0.0;
  for (int i = 0; i != n; ++i)
    maxdev = max(maxdev, abs(eig[2*i]-eig2[i]));
  // E=D^+*E*D
  zgemm_("N", "N", n2, n2, n2, 1.0, E.get(), n2, D.get(), n2, 0.0, C.get(), n2);
  zgemm_("C", "N", n2, n2, n2, 1.0, D.get(), n2, C.get(), n2, 0.0, E.get(), n2);
  for (int i = 0; i != n2; ++i)
    E[i+i*n2] -= eig2[i%n];
  const double error = real(zdotc_(n2*n2, E.get(), 1, E.get(), 1));
  // print
  cout << setprecision(5) << scientific << endl;
  cout << " * Max deviation of the eigenvalues: " << maxdev << endl;
  cout << " * Errors in the eigenvectors      : " << error << endl;
  cout << endl;
  cout << " zheev   : " << setw(10) << fixed << setprecision(2) << chrono::duration_cast<chrono::milliseconds>(time1-time0).count()*0.001 << endl;
  cout << " zquartev: " << setw(10) << fixed << setprecision(2) << chrono::duration_cast<chrono::milliseconds>(time2-time1).count()*0.001 << endl;

  return 0;
}
