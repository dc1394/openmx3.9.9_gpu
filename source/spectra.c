/**********************************************************************
  spectra.c

   spectra.c is a program which calculates a smeared spectrum from 
   excited energies and intensities.

      Usage:

         ./spectra input output

  Log of spectra.c:

     16/Apr./2020  Released by T. Ozaki 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void XANES(int argc, char *argv[]);

int main(int argc, char *argv[]) 
{
  if (argc!=3){
    printf("Usage:\n");
    printf("  ./spectra input output\n");
    exit(0);
  }

  if (argv[1],".xanes") XANES(argc,argv);

}

void XANES(int argc, char *argv[]) 
{
  int num,num_mesh=2001;
  int i,j,jmin,jmax,Num_XANES_Out;
  double tmp0,tmp1,tmp2,tmp3,tmp4,tmp5;
  double **XANES_Out,**XANES_Spec;
  double f,x,E,E0,Emin,Emax,dE,gwidth;
  char ctmp[1000];
  FILE *fp1,*fp2;
  char buf[1000];

  /*******************************************
              read the first file 
  *******************************************/

  if ((fp1 = fopen(argv[1],"r")) != NULL){

    /* get the amount of information */

    fgets(ctmp,1000,fp1);
    fgets(ctmp,1000,fp1);

    num = 0;
    while (EOF!=fscanf(fp1,"%lf %lf %lf %lf %lf",&tmp0,&tmp1,&tmp2,&tmp3,&tmp4)){
      num++;
    }
    fclose(fp1);

    Num_XANES_Out = num;

    /* allocations of arrays */

    XANES_Out = (double**)malloc(sizeof(double*)*num);
    for (i=0; i<num; i++){
      XANES_Out[i] = (double*)malloc(sizeof(double)*10);
    }

    XANES_Spec = (double**)malloc(sizeof(double*)*num_mesh);
    for (i=0; i<num_mesh; i++){
      XANES_Spec[i] = (double*)malloc(sizeof(double)*5);
      for (j=0; j<5; j++){
        XANES_Spec[i][j] = 0.0;
      }
    }

    /* read data and store them */

    if ((fp1 = fopen(argv[1],"r")) != NULL){

      fgets(ctmp,1000,fp1);
      fgets(ctmp,1000,fp1);

      num = 0;
      while (EOF!=fscanf(fp1,"%lf %lf %lf %lf %lf",&tmp0,&tmp1,&tmp2,&tmp3,&tmp4)){

        XANES_Out[num][0] = tmp0;
        XANES_Out[num][1] = tmp1;
        XANES_Out[num][2] = tmp2;
        XANES_Out[num][3] = tmp3;
        XANES_Out[num][4] = tmp4;

	num++;
      }

      fclose(fp1);
    }   

    /* find the lowest and highest energies */

    Emin = 1.0e+10;
    Emax =-1.0e+10;

    for (i=0; i<Num_XANES_Out; i++){
      if (XANES_Out[i][0]<Emin) Emin = XANES_Out[i][0];
      if (XANES_Out[i][0]>Emax) Emax = XANES_Out[i][0];
    }

    /* interactively ask the Gaussian width */

    printf("Please input a value of gaussian width (eV)\n"); 
    fgets(buf,1000,stdin); sscanf(buf,"%lf",&gwidth); 

    /* update Emin and Emax */

    Emin -= 4.0*gwidth;
    Emax += 4.0*gwidth;
 
    /* set energies on mesh */

    dE = (Emax - Emin)/(double)(num_mesh-1);
    for (i=0; i<num_mesh; i++){
      XANES_Spec[i][0] = Emin + (double)i*dE;
    }

    /* calculate XANES_Spec */

    for (i=0; i<Num_XANES_Out; i++){
     
      E0 = XANES_Out[i][0];

      jmin = (int)((E0-4.0*gwidth-Emin)/dE);
      jmax = (int)((E0+4.0*gwidth-Emin)/dE);

      if (jmin<0)            jmin = 0;
      if ((num_mesh-1)<jmax) jmax = num_mesh - 1;

      for (j=jmin; j<=jmax; j++){

        x = (XANES_Spec[j][0]-E0)/gwidth;
        f = exp(-x*x);

        XANES_Spec[j][1] += f*XANES_Out[i][1]; 
        XANES_Spec[j][2] += f*XANES_Out[i][2]; 
        XANES_Spec[j][3] += f*XANES_Out[i][3]; 
        XANES_Spec[j][4] += f*XANES_Out[i][4]; 
      }
    }

    /* output of results */

    if ((fp2 = fopen(argv[2],"w")) != NULL){

      for (i=0; i<num_mesh; i++){
	fprintf(fp2,"%15.12f %15.12f %15.12f %15.12f %15.12f\n",
		XANES_Spec[i][0],
		XANES_Spec[i][1],
		XANES_Spec[i][2],
		XANES_Spec[i][3],
		XANES_Spec[i][4]);
      }

      /* fclose of fp2 */
      fclose(fp2);
    }

    /* freeing of arrays */

    for (i=0; i<num_mesh; i++){
      free(XANES_Spec[i]);
    }
    free(XANES_Spec);

    for (i=0; i<num; i++){
      free(XANES_Out[i]);
    }
    free(XANES_Out);

  }
  else{
    printf("error in scanfing %s\n",argv[1]);
  }
}


