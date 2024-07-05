#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10000

#define MAXRAND (4294967296ULL)
#define FNORM   (2.3283064365e-10)
#define RANDOM  ((ira[ip++] = ira[ip1++] + ira[ip2++]) ^ ira[ip3++])
#define FRANDOM (FNORM * RANDOM)
#define pm1 ((FRANDOM > 0.5) ? 1 : -1)
#define min(a,b) (a < b ? a : b)
#define max(a,b) (a > b ? a : b)

double u[N], d[N], new_u[N], new_d[N];
unsigned poissonTable[1000];

unsigned myrand, ira[256];
unsigned char ip, ip1, ip2, ip3;

unsigned rand4init(void) {
  unsigned long long y;
  
  y = myrand * 16807LL;
  myrand = (y & 0x7fffffff) + (y >> 31);
  if (myrand & 0x80000000)
    myrand = (myrand & 0x7fffffff) + 1;
  return myrand;
}

void Init_Random(void) {
  int i;
  
  ip = 128;    
  ip1 = ip - 24;    
  ip2 = ip - 55;    
  ip3 = ip - 61;
  
  for (i = ip3; i < ip; i++)
    ira[i] = rand4init();
}

void initPoisson(unsigned *pTable, float alpha) {
  unsigned long long k, cut;
  double c, fact;

  c = exp(-alpha);
  cut = c * MAXRAND + 0.5;
  k = 0;
  fact = 1.0;
  while (cut < MAXRAND) {
    pTable[k++] = cut;
    fact *= alpha / k;
    cut += max((unsigned)(c * fact * MAXRAND + 0.5), 1);
  }
  pTable[k] = MAXRAND-1;
}

unsigned poissonRan(unsigned *pTable) {
  int k, ran = RANDOM;

  k = 0;
  while (ran > pTable[k])
    k++;
  return k;
}

void error(char* stringa) {
  
  fprintf(stderr, "ERROR: %s\n", stringa);
  exit(1);
}

inline double f(double x, double y) {
  return (y - x) * (y - x) / (1.- x) / (1.- x) / y;
}

void oneStep(int degree, double temperature, double field,
	     double *pAveVarU, double *pAvePert,
	     double *pAvePert2, double *pAvePert3) {
  int i, j, k, ran;
  double avePert, avePert2, avePert3, sum, pert, beta = 1./temperature;
  double newU, varU, tmp, tmp2, tanhBeta2 = tanh(beta) * tanh(beta);

  avePert = 0.0;
  varU = 0.0;
  k = degree - 1;
  for (i = 0; i < N; i++) {
    sum = field;
    pert = 0.0;
    for (j = 0; j < k; j++) {
      ran = (int)(FRANDOM * N);
      sum += u[ran];
      pert += d[ran];
    }
    newU = 0.5 * temperature *
      log(cosh(beta * (sum + 1.0)) / cosh(beta * (sum - 1.0)));
    new_u[i] = pm1 * newU;
    varU += newU * newU;
    new_d[i] = f(tanh(beta*newU)*tanh(beta*newU), tanhBeta2) * pert;
    avePert += new_d[i];
  }
  varU /= N;
  avePert /= N;
  tmp = 1./ avePert;
  avePert2 = 0.0;
  avePert3 = 0.0;
  for (i = 0; i < N; i++) {
    u[i] = new_u[i];
    d[i] = tmp * new_d[i];
    tmp2 = f(tanh(beta * new_u[i]) * tanh(beta * new_u[i]), tanhBeta2);
    avePert2 += tmp2;
    avePert3 += tmp2 * tmp2;
  }
  *pAveVarU += varU;
  *pAvePert += avePert;
  *pAvePert2 += (degree - 1) * avePert2 / N;
  *pAvePert3 += (degree - 1) * sqrt(avePert3 / N);
}

double overlap(int degree, double field, double beta) {
  int i, j, ran;
  double sum, res = 0.0;

  for (i = 0; i < N; i++) {
    sum = field;
    for (j = 0; j < degree; j++) {
      ran = (int)(FRANDOM * N);
      sum += u[ran];
    }
    res += tanh(beta * sum) * tanh(beta * sum);
  }
  return res / N;
}

void printPoints(void) {
  int i;

  for (i = 0; i < N; i++) {
    printf("%g %g\n", u[i], d[i]);
  }
  printf("\n");
}


int main(int argc, char *argv[]) {
  int i, iter, nIter, nTerm, degree;
  double field, temperature, lastTemp, dTemp;
  double aveVarU, avePert, avePert2, avePert3, q;
  
  if (argc != 9) {
    fprintf(stderr,
	    "usage: %s <fixed degree> <field> <Tmin> <Tmax> <dT> <nTerm> <nIter> <seed>\n",
	    argv[0]);
    exit(1);
  }
  degree = atoi(argv[1]);
  field = atof(argv[2]);
  temperature = atof(argv[3]);
  lastTemp = atof(argv[4]);
  dTemp = atof(argv[5]);
  nTerm = atoi(argv[6]);
  nIter = atoi(argv[7]);
  myrand = atoi(argv[8]);

  if (myrand == 2147483647) {
    fprintf(stderr, "Error: seed must be less than 2147483647\n");
    exit(1);
  }
  printf("# N = %u   fixed degree = %i   field = %.3f   nTerm = %i   nIter = %i   seed = %u\n", N, degree, field, nTerm, nIter, myrand);
  fflush(stdout);
  Init_Random();
  for (i = 0; i < N; i++) {
    u[i] = -1. + 2. * FRANDOM;
    d[i] = 1.0;
  }
  for (iter = 0; iter < 30; iter++) {
    oneStep(degree, temperature, field, &aveVarU,
	    &avePert, &avePert2, &avePert3);
  }
  while ((lastTemp - temperature) / dTemp > -0.5) {
    for (iter = 0; iter < nTerm; iter++) {
      oneStep(degree, temperature, field, &aveVarU,
	      &avePert, &avePert2, &avePert3);
    }
    aveVarU = 0.0;
    avePert = 0.0;
    avePert2 = 0.0;
    avePert3 = 0.0;
    q = 0.0;
    for (iter = 0; iter < nIter; iter++) {
      oneStep(degree, temperature, field, &aveVarU,
	      &avePert, &avePert2, &avePert3);
      q += overlap(degree, field, 1./temperature);
    }
    printf("%g %g %g %g %g %g\n", temperature, aveVarU / nIter,
	   avePert / nIter, avePert2 / nIter, avePert3 / nIter,
	   q / nIter);
    fflush(stdout);
    temperature += dTemp;
  }
  printf("\n");
  //printPoints();
  return 0;
}
