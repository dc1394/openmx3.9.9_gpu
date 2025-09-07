#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    double a,b;
} dlists;

typedef struct {
  double a;
  int b;
} dilists;


int dlists_cmp(const dlists *x, const dlists *y);
void qsort_double_int(long n, double *a, int *b);



int main(int argc, char *argv[]) 
{
  int po,i,j,k,num,knum,*flag;
  double tmp,**vals,**vals2;
  double *a;
  int *b;

  scanf("%ld",&num); 

  flag = (int*)malloc(sizeof(int)*num); 
  b = (int*)malloc(sizeof(int)*num); 
  
  vals = (double**)malloc(sizeof(double*)*num); 
  for (i=0; i<num; i++){
    vals[i] = (double*)malloc(sizeof(double)*12); 
  }

  vals2 = (double**)malloc(sizeof(double*)*num); 
  for (i=0; i<num; i++){
    vals2[i] = (double*)malloc(sizeof(double)*12); 
  }

  a = (double*)malloc(sizeof(double)*num); 

  /* read data */

  for (i=0; i<num; i++){
    flag[i] = 0;
    for (j=0; j<11; j++){
      scanf("%lf",&tmp); 
      vals[i][j] = tmp;
    }   
  }

  /* clasify the data */

  po = 0;
  knum = 0; 

  for (i=0; i<num; i++){

    po = 0;
    for (j=0; j<i; j++){
      if ( po!=1 && vals[i][1]==vals[j][1] && vals[i][2]==vals[j][2] && vals[i][3]==vals[j][3]
        && vals[i][5]==vals[j][5] && vals[i][6]==vals[j][6] ){
        po = 1;
        k = j; 
      }
    }

    if (po==0){
      flag[i] = knum;
      knum++;
    }
    else{
      flag[i] = flag[k];
    } 
  }

  /*
  for (i=0; i<num; i++){
    printf("i=%2d %10.5f %10.5f %10.5f %2.0f %2.0f  %2d\n",i,vals[i][1],vals[i][2],vals[i][3],vals[i][5],vals[i][6],flag[i]);
  }
  */

  /* sum up the data with the same index */

  for (i=0; i<knum; i++){
    for (j=0; j<11; j++){
      vals2[i][j] = 0.0;
    }
  }

  for (i=0; i<num; i++){

    j = flag[i];
    vals2[j][1] = vals[i][1];
    vals2[j][2] = vals[i][2];
    vals2[j][3] = vals[i][3];

    vals2[j][5] = vals[i][5];
    vals2[j][6] = vals[i][6];

    vals2[j][7] += vals[i][7];
    vals2[j][8] += vals[i][8];
    vals2[j][9] += vals[i][9];
    vals2[j][10] += vals[i][10];
  }

  /*
  for (i=0; i<knum; i++){
    printf("i=%2d %10.5f %10.5f %10.5f %2.0f %2.0f  %10.5f\n",i,vals2[i][1],vals2[i][2],vals2[i][3],vals2[i][5],vals2[i][6],vals2[i][7]);
  }
  */
  
  /* sort */

  for (i=0; i<knum; i++){
    a[i] = vals2[i][7];
    b[i] = i;
  }
  qsort_double_int(knum,a,b);


  for (i=0; i<knum; i++){
    j = b[i];

    printf("%2d %10.5f %10.5f %10.5f   %2.0f %2.0f  %10.5f\n",j,vals2[j][1],vals2[j][2],vals2[j][3],vals2[j][5],vals2[j][6],vals2[j][7]);
  }


  free(flag);
  free(b);
  
  for (i=0; i<num; i++){
    free(vals[i]);
  }
  free(vals);

  for (i=0; i<num; i++){
    free(vals2[i]);
  }
  free(vals2);
  free(a);

}


void qsort_double_int(long n, double *a, int *b)
{
  int i;
  dilists *AB;

  AB = (dilists*)malloc(sizeof(dilists)*n);

  for (i=0; i<n; i++){
    AB[i].a = a[i];     
    AB[i].b = b[i];
  }

  qsort(AB, n, sizeof(dilists), (int(*)(const void*, const void*))dlists_cmp);

  for (i=0; i<n; i++){
    a[i] = AB[i].a;
    b[i] = AB[i].b;
  }

  free(AB);
}

int dlists_cmp(const dlists *x, const dlists *y)
{
  return (x->a > y->a ? -1 :
          y->a > x->a ?  1 : 0);
}
