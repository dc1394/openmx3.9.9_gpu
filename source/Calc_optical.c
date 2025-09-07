/**********************************************************************
  Calc_optical.c:

     Calc_optical.c is a subroutine to calculate electric conductivity
     and dielectric function, and related optical properties.

  Log of Calc_optical.c:

     3/September/2018  Released by YT Lee 
    16/December /2019  Updated  by YT Lee
    30/July     /2021  Updated  by YT Lee
      (1) Update the code of optical calculations in collinear case. [1,2]
      (2) Add a function to calculate optical properties of a material
      in the non-collinear case by [2].
      (3) Remove a redundant option of matieral's type (metal/insulator).

  References:
  [1] PRB 102 (7), 143251 (2020) for collinear case (numerical approach).
  [2] PRB 98 (11), 115115 (2018) for collinear and non-collinear cases.

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"

int i,j,k,l,d=3,total_spins;
int CDDF_freq_grid_total_number;
int tnos; /* total number of states */
unsigned int n0;
double conductivity_unit = 4599847.9907856912; /* S/m = Mho/m = (Ohm*m)^{-1} , = 4599847.99078569120128773568916 */
double dielectricless_unit = 366044.28814557; /* (F/(m*s) = Ohm*m) */
double eta, eta2; /* setup eta and eta^2 in Hartree */
double range_Ha; /* range in frequency domain in Hartree */
double step; /* difference omega at frequency domain for conductivity and dielectric function */
dcomplex**** cd_tensor; /* for omega = 1~N */
dcomplex**** df_tensor; /* for omega = 1~N */
dcomplex*** cd_tensor_omega0; /* for omega = 0 */
dcomplex*** df_tensor_omega0; /* for omega = 0 */

/* MPI */
int myid2=0;
int numprocs2=1; /* initial value = 1 when MPI_CommWD2 doesn't be activated or used for parallelization of bands. */

void Set_MPIworld_for_optical(int myid_for_MPIWD2,int numprocs_for_MPIWD2){
  myid2 = myid_for_MPIWD2;
  numprocs2 = numprocs_for_MPIWD2;
}

void Free_optical_1(){
  int Gc_AN,h_AN,tno0;
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    if (Gc_AN==0){ Gc_AN = 0; tno0 = 1; }else{ tno0 = Spe_Total_CNO[ WhatSpecies[Gc_AN] ]; }
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      for (k=0; k<tno0; k++){
        for (l=0; l< Spe_Total_CNO[ WhatSpecies[ natn[Gc_AN][h_AN] ] ] ; l++) free(MME_allorb[Gc_AN][h_AN][k][l]);
        free(MME_allorb[Gc_AN][h_AN][k]);
      }
      free(MME_allorb[Gc_AN][h_AN]);
    }
    free(MME_allorb[Gc_AN]);
  }
  free(MME_allorb);
}

void Free_optical_2(int n){
  for (l=0;l<total_spins;l++){
    for (i=0;i<d;i++){
      for (j=0;j<d;j++) free(cd_tensor[l][i][j]);
      free(cd_tensor[l][i]);
    }
    free(cd_tensor[l]);
  }
  free(cd_tensor);

  for (l=0;l<total_spins;l++){
    for (i=0;i<d;i++){
      for (j=0;j<d;j++) free(df_tensor[l][i][j]);
      free(df_tensor[l][i]);
    }
    free(df_tensor[l]);
  }
  free(df_tensor);

  for (l=0;l<total_spins;l++){
    for (i=0;i<d;i++) free(cd_tensor_omega0[l][i]);
    free(cd_tensor_omega0[l]);
  }
  free(cd_tensor_omega0);

  for (l=0;l<total_spins;l++){
    for (i=0;i<d;i++) free(df_tensor_omega0[l][i]);
    free(df_tensor_omega0[l]);
  }
  free(df_tensor_omega0);
}

void Initialize_optical(){
  eta = CDDF_FWHM/eV2Hartree;
  eta2=eta*eta; /* setup eta and eta^2 in Hartree */
  range_Ha = (CDDF_max_eV-CDDF_min_eV)/eV2Hartree; /* in Hartree */
  step     = range_Ha/CDDF_freq_grid_number;
  CDDF_freq_grid_total_number = CDDF_freq_grid_number + ( CDDF_FWHM/(step*eV2Hartree) );

  if (SpinP_switch==0){
    total_spins=1;
  }else if (SpinP_switch==1){
    total_spins=2;
  }else if (SpinP_switch==3){
    total_spins=4;
  }

  /* declare conductivity tensor [1 (SpinP_Switch=0 or 3) or 2 (SpinP_Switch=1)][3][3][freq_grid] */
  cd_tensor = (dcomplex****)malloc(sizeof(dcomplex***)*total_spins);
  for (l=0;l<total_spins;l++){
    cd_tensor[l] = (dcomplex***)malloc(sizeof(dcomplex**)*d);
    for (i=0;i<d;i++){
      cd_tensor[l][i] = (dcomplex**)malloc(sizeof(dcomplex*)*d);
      for (j=0;j<d;j++){
        cd_tensor[l][i][j] = (dcomplex*)malloc(sizeof(dcomplex)*CDDF_freq_grid_total_number);
        for (k=0;k<CDDF_freq_grid_total_number;k++) cd_tensor[l][i][j][k] = Complex(0.0,0.0);
      }
    }
  }

  df_tensor = (dcomplex****)malloc(sizeof(dcomplex***)*total_spins);
  for (l=0;l<total_spins;l++){
    df_tensor[l] = (dcomplex***)malloc(sizeof(dcomplex**)*d);
    for (i=0;i<d;i++){
      df_tensor[l][i] = (dcomplex**)malloc(sizeof(dcomplex*)*d);
      for (j=0;j<d;j++){
        df_tensor[l][i][j] = (dcomplex*)malloc(sizeof(dcomplex)*CDDF_freq_grid_total_number);
        for (k=0;k<CDDF_freq_grid_total_number;k++) df_tensor[l][i][j][k] = Complex(0.0,0.0);
      }
    }
  }

  cd_tensor_omega0 = (dcomplex***)malloc(sizeof(dcomplex**)*total_spins);
  for (l=0;l<total_spins;l++){
    cd_tensor_omega0[l] = (dcomplex**)malloc(sizeof(dcomplex*)*d);
    for (i=0;i<d;i++){
      cd_tensor_omega0[l][i] = (dcomplex*)malloc(sizeof(dcomplex)*d);
      for (j=0;j<d;j++) cd_tensor_omega0[l][i][j] = Complex(0.0,0.0);
    }
  }

  df_tensor_omega0 = (dcomplex***)malloc(sizeof(dcomplex**)*total_spins);
  for (l=0;l<total_spins;l++){
    df_tensor_omega0[l] = (dcomplex**)malloc(sizeof(dcomplex*)*d);
    for (i=0;i<d;i++){
      df_tensor_omega0[l][i] = (dcomplex*)malloc(sizeof(dcomplex)*d);
      for (j=0;j<d;j++) df_tensor_omega0[l][i][j] = Complex(0.0,0.0);
    }
  }
}

void Print_optical(int doesPrintCD,int doesPrintDF,int doesPrintOP){
  /* doesPrintCD = does it print out conductivity? */
  /* doesPrintDF = does it print out dielectric function? */
  /* doesPrintOP = does it print out optical properties? */

  /* start to save conductivity tensor, dielectric function, and other optical properties */
  double k1,k2,k3,k4,omega;

  char fname1[300];
  FILE *fp;

  /* print conductivity and dielectric function */
  if (SpinP_switch==0){

    if (doesPrintCD==1){

      sprintf(fname1,"%s%s.cd_re",filepath,filename);
      fp = fopen(fname1,"w");
      fprintf(fp,"# conductivity tensor (real part) , unit = Siemens/meter = Mho/meter = 1/(Ohm*meter)\n");
      fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
      fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz   (xx+yy+zz)/3\n");
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", 0.0,
          cd_tensor_omega0[0][0][0].r, cd_tensor_omega0[0][0][1].r, cd_tensor_omega0[0][0][2].r,
          cd_tensor_omega0[0][1][0].r, cd_tensor_omega0[0][1][1].r, cd_tensor_omega0[0][1][2].r,
          cd_tensor_omega0[0][2][0].r, cd_tensor_omega0[0][2][1].r, cd_tensor_omega0[0][2][2].r,
          (cd_tensor_omega0[0][0][0].r + cd_tensor_omega0[0][1][1].r + cd_tensor_omega0[0][2][2].r )/3.0 );
      for (k=0;k<CDDF_freq_grid_number;k++){
        omega=(k+1)*step*eV2Hartree;
        fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", omega,
            cd_tensor[0][0][0][k].r, cd_tensor[0][0][1][k].r, cd_tensor[0][0][2][k].r,
            cd_tensor[0][1][0][k].r, cd_tensor[0][1][1][k].r, cd_tensor[0][1][2][k].r,
            cd_tensor[0][2][0][k].r, cd_tensor[0][2][1][k].r, cd_tensor[0][2][2][k].r,
            (cd_tensor[0][0][0][k].r + cd_tensor[0][1][1][k].r + cd_tensor[0][2][2][k].r )/3.0 );
      }
      fclose(fp);

      sprintf(fname1,"%s%s.cd_im",filepath,filename);
      fp = fopen(fname1,"w");
      fprintf(fp,"# conductivity tensor (imaginery part) , unit = Siemens/meter = Mho/meter = 1/(Ohm*meter)\n");
      fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
      fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz   (xx+yy+zz)/3\n");
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", 0.0,
          cd_tensor_omega0[0][0][0].i, cd_tensor_omega0[0][0][1].i, cd_tensor_omega0[0][0][2].i,
          cd_tensor_omega0[0][1][0].i, cd_tensor_omega0[0][1][1].i, cd_tensor_omega0[0][1][2].i,
          cd_tensor_omega0[0][2][0].i, cd_tensor_omega0[0][2][1].i, cd_tensor_omega0[0][2][2].i,
          (cd_tensor_omega0[0][0][0].i + cd_tensor_omega0[0][1][1].i + cd_tensor_omega0[0][2][2].i )/3.0 );
      for (k=0;k<CDDF_freq_grid_number;k++){
        omega=(k+1)*step*eV2Hartree;
        fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", omega,
            cd_tensor[0][0][0][k].i, cd_tensor[0][0][1][k].i, cd_tensor[0][0][2][k].i,
            cd_tensor[0][1][0][k].i, cd_tensor[0][1][1][k].i, cd_tensor[0][1][2][k].i,
            cd_tensor[0][2][0][k].i, cd_tensor[0][2][1][k].i, cd_tensor[0][2][2][k].i,
            (cd_tensor[0][0][0][k].i + cd_tensor[0][1][1][k].i + cd_tensor[0][2][2][k].i )/3.0 );
      }
      fclose(fp);

    }

    if (doesPrintDF==1){

      sprintf(fname1,"%s%s.df_re",filepath,filename);
      fp = fopen(fname1,"w");
      fprintf(fp,"# dielectric function (real part)\n");
      fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
      fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz   (xx+yy+zz)/3\n");
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", 0.0,
          df_tensor_omega0[0][0][0].r+1, df_tensor_omega0[0][0][1].r, df_tensor_omega0[0][0][2].r,
          df_tensor_omega0[0][1][0].r, df_tensor_omega0[0][1][1].r+1, df_tensor_omega0[0][1][2].r,
          df_tensor_omega0[0][2][0].r, df_tensor_omega0[0][2][1].r, df_tensor_omega0[0][2][2].r+1,
          (df_tensor_omega0[0][0][0].r + df_tensor_omega0[0][1][1].r + df_tensor_omega0[0][2][2].r +3)/3.0 );
      for (k=0;k<CDDF_freq_grid_number;k++){
        omega=(k+1)*step*eV2Hartree;
        fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", omega,
          df_tensor[0][0][0][k].r+1, df_tensor[0][0][1][k].r, df_tensor[0][0][2][k].r,
          df_tensor[0][1][0][k].r, df_tensor[0][1][1][k].r+1, df_tensor[0][1][2][k].r,
          df_tensor[0][2][0][k].r, df_tensor[0][2][1][k].r, df_tensor[0][2][2][k].r+1,
          (df_tensor[0][0][0][k].r + df_tensor[0][1][1][k].r + df_tensor[0][2][2][k].r +3)/3.0 );
      }
      fclose(fp);

      sprintf(fname1,"%s%s.df_im",filepath,filename);
      fp = fopen(fname1,"w");
      fprintf(fp,"# dielectric function (imaginery part)\n");
      fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
      fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz   (xx+yy+zz)/3\n");
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", 0.0,
          df_tensor_omega0[0][0][0].i, df_tensor_omega0[0][0][1].i, df_tensor_omega0[0][0][2].i,
          df_tensor_omega0[0][1][0].i, df_tensor_omega0[0][1][1].i, df_tensor_omega0[0][1][2].i,
          df_tensor_omega0[0][2][0].i, df_tensor_omega0[0][2][1].i, df_tensor_omega0[0][2][2].i,
          (df_tensor_omega0[0][0][0].i + df_tensor_omega0[0][1][1].i + df_tensor_omega0[0][2][2].i )/3.0 );
      for (k=0;k<CDDF_freq_grid_number;k++){
        omega=(k+1)*step*eV2Hartree;
        fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", omega,
          df_tensor[0][0][0][k].i, df_tensor[0][0][1][k].i, df_tensor[0][0][2][k].i,
          df_tensor[0][1][0][k].i, df_tensor[0][1][1][k].i, df_tensor[0][1][2][k].i,
          df_tensor[0][2][0][k].i, df_tensor[0][2][1][k].i, df_tensor[0][2][2][k].i,
          (df_tensor[0][0][0][k].i + df_tensor[0][1][1][k].i + df_tensor[0][2][2][k].i )/3.0  );
      }
      fclose(fp);

    }
  }else if (SpinP_switch==1 || SpinP_switch==3){

    if (doesPrintCD==1){

      sprintf(fname1,"%s%s.cd_re",filepath,filename);
      fp = fopen(fname1,"w");
      fprintf(fp,"# conductivity tensor (real part) , unit = Siemens/meter = Mho/meter = 1/(Ohm*meter)\n");
      fprintf(fp,"# index: energy-grid=1, xx~zz=2~10, trace=11, up-xx ~ up-zz=12~20, down-xx ~ down-zz=21~29\n");
      fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz   (xx+yy+zz)/3");
      fprintf(fp,"        up-xx         up-xy         up-xz         up-yx         up-yy         up-yz         up-zx         up-zy         up-zz");
      fprintf(fp,"       down-xx       down-xy       down-xz       down-yx       down-yy       down-yz       down-zx       down-zy       down-zz\n");
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", 0.0,
          cd_tensor_omega0[0][0][0].r+cd_tensor_omega0[1][0][0].r, cd_tensor_omega0[0][0][1].r+cd_tensor_omega0[1][0][1].r, cd_tensor_omega0[0][0][2].r+cd_tensor_omega0[1][0][2].r,
          cd_tensor_omega0[0][1][0].r+cd_tensor_omega0[1][1][0].r, cd_tensor_omega0[0][1][1].r+cd_tensor_omega0[1][1][1].r, cd_tensor_omega0[0][1][2].r+cd_tensor_omega0[1][1][2].r,
          cd_tensor_omega0[0][2][0].r+cd_tensor_omega0[1][2][0].r, cd_tensor_omega0[0][2][1].r+cd_tensor_omega0[1][2][1].r, cd_tensor_omega0[0][2][2].r+cd_tensor_omega0[1][2][2].r,
          (cd_tensor_omega0[0][0][0].r + cd_tensor_omega0[0][1][1].r + cd_tensor_omega0[0][2][2].r + cd_tensor_omega0[1][0][0].r + cd_tensor_omega0[1][1][1].r + cd_tensor_omega0[1][2][2].r)/3.0,
          cd_tensor_omega0[0][0][0].r, cd_tensor_omega0[0][0][1].r, cd_tensor_omega0[0][0][2].r, /* for up-spin tensor */
          cd_tensor_omega0[0][1][0].r, cd_tensor_omega0[0][1][1].r, cd_tensor_omega0[0][1][2].r,
          cd_tensor_omega0[0][2][0].r, cd_tensor_omega0[0][2][1].r, cd_tensor_omega0[0][2][2].r,
          cd_tensor_omega0[1][0][0].r, cd_tensor_omega0[1][0][1].r, cd_tensor_omega0[1][0][2].r, /* for down-spin tensor */
          cd_tensor_omega0[1][1][0].r, cd_tensor_omega0[1][1][1].r, cd_tensor_omega0[1][1][2].r,
          cd_tensor_omega0[1][2][0].r, cd_tensor_omega0[1][2][1].r, cd_tensor_omega0[1][2][2].r);
      for (k=0;k<CDDF_freq_grid_number;k++){
        omega=(k+1)*step*eV2Hartree;
        fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", omega,
          cd_tensor[0][0][0][k].r+cd_tensor[1][0][0][k].r, cd_tensor[0][0][1][k].r+cd_tensor[1][0][1][k].r, cd_tensor[0][0][2][k].r+cd_tensor[1][0][2][k].r,
          cd_tensor[0][1][0][k].r+cd_tensor[1][1][0][k].r, cd_tensor[0][1][1][k].r+cd_tensor[1][1][1][k].r, cd_tensor[0][1][2][k].r+cd_tensor[1][1][2][k].r,
          cd_tensor[0][2][0][k].r+cd_tensor[1][2][0][k].r, cd_tensor[0][2][1][k].r+cd_tensor[1][2][1][k].r, cd_tensor[0][2][2][k].r+cd_tensor[1][2][2][k].r,
          (cd_tensor[0][0][0][k].r + cd_tensor[0][1][1][k].r + cd_tensor[0][2][2][k].r + cd_tensor[1][0][0][k].r + cd_tensor[1][1][1][k].r + cd_tensor[1][2][2][k].r)/3.0,
          cd_tensor[0][0][0][k].r, cd_tensor[0][0][1][k].r, cd_tensor[0][0][2][k].r, /* for up-spin tensor */
          cd_tensor[0][1][0][k].r, cd_tensor[0][1][1][k].r, cd_tensor[0][1][2][k].r,
          cd_tensor[0][2][0][k].r, cd_tensor[0][2][1][k].r, cd_tensor[0][2][2][k].r,
          cd_tensor[1][0][0][k].r, cd_tensor[1][0][1][k].r, cd_tensor[1][0][2][k].r, /* for down-spin tensor */
          cd_tensor[1][1][0][k].r, cd_tensor[1][1][1][k].r, cd_tensor[1][1][2][k].r,
          cd_tensor[1][2][0][k].r, cd_tensor[1][2][1][k].r, cd_tensor[1][2][2][k].r);
      }
      fclose(fp);

      sprintf(fname1,"%s%s.cd_im",filepath,filename);
      fp = fopen(fname1,"w");
      fprintf(fp,"# conductivity tensor (imaginery part) , unit = Siemens/meter = Mho/meter = 1/(Ohm*meter)\n");
      fprintf(fp,"# index: energy-grid=1, xx~zz=2~10, trace=11, up-xx ~ up-zz=12~20, down-xx ~ down-zz=21~29\n");
      fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz   (xx+yy+zz)/3");
      fprintf(fp,"        up-xx         up-xy         up-xz         up-yx         up-yy         up-yz         up-zx         up-zy         up-zz");
      fprintf(fp,"       down-xx       down-xy       down-xz       down-yx       down-yy       down-yz       down-zx       down-zy       down-zz\n");
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", 0.0,
          cd_tensor_omega0[0][0][0].i+cd_tensor_omega0[1][0][0].i, cd_tensor_omega0[0][0][1].i+cd_tensor_omega0[1][0][1].i, cd_tensor_omega0[0][0][2].i+cd_tensor_omega0[1][0][2].i,
          cd_tensor_omega0[0][1][0].i+cd_tensor_omega0[1][1][0].i, cd_tensor_omega0[0][1][1].i+cd_tensor_omega0[1][1][1].i, cd_tensor_omega0[0][1][2].i+cd_tensor_omega0[1][1][2].i,
          cd_tensor_omega0[0][2][0].i+cd_tensor_omega0[1][2][0].i, cd_tensor_omega0[0][2][1].i+cd_tensor_omega0[1][2][1].i, cd_tensor_omega0[0][2][2].i+cd_tensor_omega0[1][2][2].i,
          (cd_tensor_omega0[0][0][0].i + cd_tensor_omega0[0][1][1].i + cd_tensor_omega0[0][2][2].i + cd_tensor_omega0[1][0][0].i + cd_tensor_omega0[1][1][1].i + cd_tensor_omega0[1][2][2].i)/3.0,
          cd_tensor_omega0[0][0][0].i, cd_tensor_omega0[0][0][1].i, cd_tensor_omega0[0][0][2].i, /* for up-spin tensor */
          cd_tensor_omega0[0][1][0].i, cd_tensor_omega0[0][1][1].i, cd_tensor_omega0[0][1][2].i,
          cd_tensor_omega0[0][2][0].i, cd_tensor_omega0[0][2][1].i, cd_tensor_omega0[0][2][2].i,
          cd_tensor_omega0[1][0][0].i, cd_tensor_omega0[1][0][1].i, cd_tensor_omega0[1][0][2].i, /* for down-spin tensor */
          cd_tensor_omega0[1][1][0].i, cd_tensor_omega0[1][1][1].i, cd_tensor_omega0[1][1][2].i,
          cd_tensor_omega0[1][2][0].i, cd_tensor_omega0[1][2][1].i, cd_tensor_omega0[1][2][2].i);
      for (k=0;k<CDDF_freq_grid_number;k++){
        omega=(k+1)*step*eV2Hartree;
        fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", omega,
          cd_tensor[0][0][0][k].i+cd_tensor[1][0][0][k].i, cd_tensor[0][0][1][k].i+cd_tensor[1][0][1][k].i, cd_tensor[0][0][2][k].i+cd_tensor[1][0][2][k].i,
          cd_tensor[0][1][0][k].i+cd_tensor[1][1][0][k].i, cd_tensor[0][1][1][k].i+cd_tensor[1][1][1][k].i, cd_tensor[0][1][2][k].i+cd_tensor[1][1][2][k].i,
          cd_tensor[0][2][0][k].i+cd_tensor[1][2][0][k].i, cd_tensor[0][2][1][k].i+cd_tensor[1][2][1][k].i, cd_tensor[0][2][2][k].i+cd_tensor[1][2][2][k].i,
          (cd_tensor[0][0][0][k].i + cd_tensor[0][1][1][k].i + cd_tensor[0][2][2][k].i + cd_tensor[1][0][0][k].i + cd_tensor[1][1][1][k].i + cd_tensor[1][2][2][k].i )/3.0,
          cd_tensor[0][0][0][k].i, cd_tensor[0][0][1][k].i, cd_tensor[0][0][2][k].i, /* for up-spin tensor */
          cd_tensor[0][1][0][k].i, cd_tensor[0][1][1][k].i, cd_tensor[0][1][2][k].i,
          cd_tensor[0][2][0][k].i, cd_tensor[0][2][1][k].i, cd_tensor[0][2][2][k].i,
          cd_tensor[1][0][0][k].i, cd_tensor[1][0][1][k].i, cd_tensor[1][0][2][k].i, /* for down-spin tensor */
          cd_tensor[1][1][0][k].i, cd_tensor[1][1][1][k].i, cd_tensor[1][1][2][k].i,
          cd_tensor[1][2][0][k].i, cd_tensor[1][2][1][k].i, cd_tensor[1][2][2][k].i);
      }
      fclose(fp);

    }

    if (doesPrintDF==1){

      sprintf(fname1,"%s%s.df_re",filepath,filename);
      fp = fopen(fname1,"w");
      fprintf(fp,"# dielectric function (real part)\n");
      fprintf(fp,"# index: energy-grid=1, xx~zz=2~10, trace=11, up-xx ~ up-zz=12~20, down-xx ~ down-zz=21~29\n");
      fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz   (xx+yy+zz)/3");
      fprintf(fp,"        up-xx         up-xy         up-xz         up-yx         up-yy         up-yz         up-zx         up-zy         up-zz");
      fprintf(fp,"       down-xx       down-xy       down-xz       down-yx       down-yy       down-yz       down-zx       down-zy       down-zz\n");
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", 0.0,
          df_tensor_omega0[0][0][0].r+df_tensor_omega0[1][0][0].r+1, df_tensor_omega0[0][0][1].r+df_tensor_omega0[1][0][1].r, df_tensor_omega0[0][0][2].r+df_tensor_omega0[1][0][2].r,
          df_tensor_omega0[0][1][0].r+df_tensor_omega0[1][1][0].r, df_tensor_omega0[0][1][1].r+df_tensor_omega0[1][1][1].r+1, df_tensor_omega0[0][1][2].r+df_tensor_omega0[1][1][2].r,
          df_tensor_omega0[0][2][0].r+df_tensor_omega0[1][2][0].r, df_tensor_omega0[0][2][1].r+df_tensor_omega0[1][2][1].r, df_tensor_omega0[0][2][2].r+df_tensor_omega0[1][2][2].r+1,
          (3+df_tensor_omega0[0][0][0].r + df_tensor_omega0[0][1][1].r + df_tensor_omega0[0][2][2].r + df_tensor_omega0[1][0][0].r + df_tensor_omega0[1][1][1].r + df_tensor_omega0[1][2][2].r)/3.0,
          df_tensor_omega0[0][0][0].r+1, df_tensor_omega0[0][0][1].r, df_tensor_omega0[0][0][2].r, /* for up-spin tensor */
          df_tensor_omega0[0][1][0].r, df_tensor_omega0[0][1][1].r+1, df_tensor_omega0[0][1][2].r,
          df_tensor_omega0[0][2][0].r, df_tensor_omega0[0][2][1].r, df_tensor_omega0[0][2][2].r+1,
          df_tensor_omega0[1][0][0].r+1, df_tensor_omega0[1][0][1].r, df_tensor_omega0[1][0][2].r, /* for down-spin tensor */
          df_tensor_omega0[1][1][0].r, df_tensor_omega0[1][1][1].r+1, df_tensor_omega0[1][1][2].r,
          df_tensor_omega0[1][2][0].r, df_tensor_omega0[1][2][1].r, df_tensor_omega0[1][2][2].r+1);
      for (k=0;k<CDDF_freq_grid_number;k++){
        omega=(k+1)*step*eV2Hartree;
        fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", omega,
          df_tensor[0][0][0][k].r+df_tensor[1][0][0][k].r+1, df_tensor[0][0][1][k].r+df_tensor[1][0][1][k].r, df_tensor[0][0][2][k].r+df_tensor[1][0][2][k].r,
          df_tensor[0][1][0][k].r+df_tensor[1][1][0][k].r, df_tensor[0][1][1][k].r+df_tensor[1][1][1][k].r+1, df_tensor[0][1][2][k].r+df_tensor[1][1][2][k].r,
          df_tensor[0][2][0][k].r+df_tensor[1][2][0][k].r, df_tensor[0][2][1][k].r+df_tensor[1][2][1][k].r, df_tensor[0][2][2][k].r+df_tensor[1][2][2][k].r+1,
          (df_tensor[0][0][0][k].r+df_tensor[0][1][1][k].r+df_tensor[0][2][2][k].r+df_tensor[1][0][0][k].r+df_tensor[1][1][1][k].r+df_tensor[1][2][2][k].r+3)/3.0,
          df_tensor[0][0][0][k].r, df_tensor[0][0][1][k].r, df_tensor[0][0][2][k].r, /* for up-spin tensor */
          df_tensor[0][1][0][k].r, df_tensor[0][1][1][k].r, df_tensor[0][1][2][k].r,
          df_tensor[0][2][0][k].r, df_tensor[0][2][1][k].r, df_tensor[0][2][2][k].r,
          df_tensor[1][0][0][k].r, df_tensor[1][0][1][k].r, df_tensor[1][0][2][k].r, /* for down-spin tensor */
          df_tensor[1][1][0][k].r, df_tensor[1][1][1][k].r, df_tensor[1][1][2][k].r,
          df_tensor[1][2][0][k].r, df_tensor[1][2][1][k].r, df_tensor[1][2][2][k].r);
      }
      fclose(fp);

      sprintf(fname1,"%s%s.df_im",filepath,filename);
      fp = fopen(fname1,"w");
      fprintf(fp,"# dielectric function (imaginery part)\n");
      fprintf(fp,"# index: energy-grid=1, xx~zz=2~10, trace=11, up-xx ~ up-zz=12~20, down-xx ~ down-zz=21~29\n");
      fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz   (xx+yy+zz)/3");
      fprintf(fp,"        up-xx         up-xy         up-xz         up-yx         up-yy         up-yz         up-zx         up-zy         up-zz");
      fprintf(fp,"       down-xx       down-xy       down-xz       down-yx       down-yy       down-yz       down-zx       down-zy       down-zz\n");
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", 0.0,
          df_tensor_omega0[0][0][0].i+df_tensor_omega0[1][0][0].i, df_tensor_omega0[0][0][1].i+df_tensor_omega0[1][0][1].i, df_tensor_omega0[0][0][2].i+df_tensor_omega0[1][0][2].i,
          df_tensor_omega0[0][1][0].i+df_tensor_omega0[1][1][0].i, df_tensor_omega0[0][1][1].i+df_tensor_omega0[1][1][1].i, df_tensor_omega0[0][1][2].i+df_tensor_omega0[1][1][2].i,
          df_tensor_omega0[0][2][0].i+df_tensor_omega0[1][2][0].i, df_tensor_omega0[0][2][1].i+df_tensor_omega0[1][2][1].i, df_tensor_omega0[0][2][2].i+df_tensor_omega0[1][2][2].i,
          (df_tensor_omega0[0][0][0].i + df_tensor_omega0[0][1][1].i + df_tensor_omega0[0][2][2].i + df_tensor_omega0[1][0][0].i + df_tensor_omega0[1][1][1].i + df_tensor_omega0[1][2][2].i)/3.0,
          df_tensor_omega0[0][0][0].i, df_tensor_omega0[0][0][1].i, df_tensor_omega0[0][0][2].i, /* for up-spin tensor */
          df_tensor_omega0[0][1][0].i, df_tensor_omega0[0][1][1].i, df_tensor_omega0[0][1][2].i,
          df_tensor_omega0[0][2][0].i, df_tensor_omega0[0][2][1].i, df_tensor_omega0[0][2][2].i,
          df_tensor_omega0[1][0][0].i, df_tensor_omega0[1][0][1].i, df_tensor_omega0[1][0][2].i, /* for down-spin tensor */
          df_tensor_omega0[1][1][0].i, df_tensor_omega0[1][1][1].i, df_tensor_omega0[1][1][2].i,
          df_tensor_omega0[1][2][0].i, df_tensor_omega0[1][2][1].i, df_tensor_omega0[1][2][2].i);
      for (k=0;k<CDDF_freq_grid_number;k++){
        omega=(k+1)*step*eV2Hartree;
        fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", omega,
          df_tensor[0][0][0][k].i+df_tensor[1][0][0][k].i, df_tensor[0][0][1][k].i+df_tensor[1][0][1][k].i, df_tensor[0][0][2][k].i+df_tensor[1][0][2][k].i,
          df_tensor[0][1][0][k].i+df_tensor[1][1][0][k].i, df_tensor[0][1][1][k].i+df_tensor[1][1][1][k].i, df_tensor[0][1][2][k].i+df_tensor[1][1][2][k].i,
          df_tensor[0][2][0][k].i+df_tensor[1][2][0][k].i, df_tensor[0][2][1][k].i+df_tensor[1][2][1][k].i, df_tensor[0][2][2][k].i+df_tensor[1][2][2][k].i,
          (df_tensor[0][0][0][k].i+df_tensor[0][1][1][k].i+df_tensor[0][2][2][k].i+df_tensor[1][0][0][k].i+df_tensor[1][1][1][k].i+df_tensor[1][2][2][k].i)/3.0,
          df_tensor[0][0][0][k].i, df_tensor[0][0][1][k].i, df_tensor[0][0][2][k].i, /* for up-spin tensor */
          df_tensor[0][1][0][k].i, df_tensor[0][1][1][k].i, df_tensor[0][1][2][k].i,
          df_tensor[0][2][0][k].i, df_tensor[0][2][1][k].i, df_tensor[0][2][2][k].i,
          df_tensor[1][0][0][k].i, df_tensor[1][0][1][k].i, df_tensor[1][0][2][k].i, /* for down-spin tensor */
          df_tensor[1][1][0][k].i, df_tensor[1][1][1][k].i, df_tensor[1][1][2][k].i,
          df_tensor[1][2][0][k].i, df_tensor[1][2][1][k].i, df_tensor[1][2][2][k].i);
      }
      fclose(fp);

    }

  }


  if (doesPrintOP==1){

    /* start to write refractive index, extinction_coefficient, absorption coefficient, reflection coefficient , and transmission coefficient. */
    sprintf(fname1,"%s%s.refractive_index",filepath,filename); /* rft = refractive */
    fp = fopen(fname1,"w");
    fprintf(fp,"# refractive index\n");
    fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
    fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz\n");

    double refractive_index[9],extinction_coefficient[9],absorption_coefficient[9],reflection_coefficient[9],transmission_coefficient[9];
    /* at omega = 0 */
    for (i=0;i<d;i++){
      for (j=0;j<d;j++){
        l=i*d+j;
        if (SpinP_switch==0){
          k2 = df_tensor_omega0[0][i][j].r;
          if (i==j) k2+=1;
          k1 = sqrt ( k2*k2 + df_tensor_omega0[0][i][j].i*df_tensor_omega0[0][i][j].i ) ;
          refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
        }else if (SpinP_switch==1 || SpinP_switch==3){
          k2 = df_tensor_omega0[0][i][j].r+df_tensor_omega0[1][i][j].r;
          k3 = df_tensor_omega0[0][i][j].i+df_tensor_omega0[1][i][j].i;
          if (i==j) k2+=1;
          k1 = sqrt ( k2*k2 + k3*k3 ) ;
          refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
        }
      }
    }
    fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",0.0,refractive_index[0],refractive_index[1],refractive_index[2],refractive_index[3],refractive_index[4],refractive_index[5],refractive_index[6],refractive_index[7],refractive_index[8]);
    for (k=0;k<CDDF_freq_grid_number;k++){
      omega=(k+1)*step*eV2Hartree;
      for (i=0;i<d;i++){
        for (j=0;j<d;j++){ 
          l=i*d+j;
          if (SpinP_switch==0){
            k2 = df_tensor[0][i][j][k].r;
            if (i==j) k2+=1;
            k1 = sqrt ( k2*k2 + df_tensor[0][i][j][k].i*df_tensor[0][i][j][k].i ) ;
            refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
          }else if (SpinP_switch==1 || SpinP_switch==3){
            k2 = df_tensor[0][i][j][k].r+df_tensor[1][i][j][k].r;
            k3 = df_tensor[0][i][j][k].i+df_tensor[1][i][j][k].i;
            if (i==j) k2+=1;
            k1 = sqrt ( k2*k2 + k3*k3 ) ;
            refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
          }
        }
      }
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",omega,refractive_index[0],refractive_index[1],refractive_index[2],refractive_index[3],refractive_index[4],refractive_index[5],refractive_index[6],refractive_index[7],refractive_index[8]);
    }
    fclose(fp);

    sprintf(fname1,"%s%s.extinction",filepath,filename);
    fp = fopen(fname1,"w");
    fprintf(fp,"# extinction coefficient\n");
    fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
    fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz\n");
    /* at omega = 0 */
    for (i=0;i<d;i++){
      for (j=0;j<d;j++){ 
        l=i*d+j;
        if (SpinP_switch==0){
          k2 = df_tensor_omega0[0][i][j].r;
          if (i==j) k2+=1;
          k1 = sqrt ( k2*k2 + df_tensor_omega0[0][i][j].i*df_tensor_omega0[0][i][j].i ) ;
          extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
        }else if (SpinP_switch==1 || SpinP_switch==3){
          k2 = df_tensor_omega0[0][i][j].r + df_tensor_omega0[1][i][j].r;
          k3 = df_tensor_omega0[0][i][j].i + df_tensor_omega0[1][i][j].i;
          if (i==j) k2+=1;
          k1 = sqrt ( k2*k2 + k3*k3 ) ;
          extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
        }
      }
    }
    fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",0.0,extinction_coefficient[0],extinction_coefficient[1],extinction_coefficient[2],extinction_coefficient[3],extinction_coefficient[4],extinction_coefficient[5],extinction_coefficient[6],extinction_coefficient[7],extinction_coefficient[8]);
    for (k=0;k<CDDF_freq_grid_number;k++){
      omega=(k+1)*step*eV2Hartree;
      for (i=0;i<d;i++){
        for (j=0;j<d;j++){ 
          l=i*d+j;
          if (SpinP_switch==0){
            k2 = df_tensor[0][i][j][k].r;
            if (i==j) k2+=1;
            k1 = sqrt ( k2*k2 + df_tensor[0][i][j][k].i*df_tensor[0][i][j][k].i ) ;
            extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
          }else if (SpinP_switch==1 || SpinP_switch==3){
            k2 = df_tensor[0][i][j][k].r + df_tensor[1][i][j][k].r;
            k3 = df_tensor[0][i][j][k].i + df_tensor[1][i][j][k].i;
            if (i==j) k2+=1;
            k1 = sqrt ( k2*k2 + k3*k3 ) ;
            extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
          }
        }
      }
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",omega,extinction_coefficient[0],extinction_coefficient[1],extinction_coefficient[2],extinction_coefficient[3],extinction_coefficient[4],extinction_coefficient[5],extinction_coefficient[6],extinction_coefficient[7],extinction_coefficient[8]);
    }
    fclose(fp);

    sprintf(fname1,"%s%s.absorption",filepath,filename);
    fp = fopen(fname1,"w");
    fprintf(fp,"# absorption coefficient (unit = 10^6 1/m)\n");
    fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
    fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz\n");
    double hbar_ineV = pow(10,9)/0.65821195144 ; /* absorption coefficient (10^6 1/m) */
    double speed_of_light = 299792458 ;
    /* at omega = 0 */
    fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
    for (k=0;k<CDDF_freq_grid_number;k++){
      omega=(k+1)*step*eV2Hartree;
      for (i=0;i<d;i++){
        for (j=0;j<d;j++){ 
          l=i*d+j;
          if (SpinP_switch==0){
            k2 = df_tensor[0][i][j][k].r;
            if (i==j) k2+=1;
            k1 = sqrt ( k2*k2 + df_tensor[0][i][j][k].i*df_tensor[0][i][j][k].i ) ;
            extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
          }else if (SpinP_switch==1 || SpinP_switch==3){
            k2 = df_tensor[0][i][j][k].r+df_tensor[1][i][j][k].r;
            k3 = df_tensor[0][i][j][k].i+df_tensor[1][i][j][k].i;
            if (i==j) k2+=1;
            k1 = sqrt ( k2*k2 + k3*k3 ) ;
            extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
          }
          absorption_coefficient[l] = 2 * omega * extinction_coefficient[l] * hbar_ineV / speed_of_light ; 
        }
      }
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",omega,absorption_coefficient[0],absorption_coefficient[1],absorption_coefficient[2],absorption_coefficient[3],absorption_coefficient[4],absorption_coefficient[5],absorption_coefficient[6],absorption_coefficient[7],absorption_coefficient[8]);
    }
    fclose(fp);

    sprintf(fname1,"%s%s.reflection",filepath,filename);
    fp = fopen(fname1,"w");
    fprintf(fp,"# reflection coefficient (%%)\n");
    fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
    fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz\n");
    /* at omega = 0 */
    for (i=0;i<d;i++){
      for (j=0;j<d;j++){ 
        l=i*d+j;
        if (SpinP_switch==0){
          k2 = df_tensor_omega0[0][i][j].r;
          if (i==j) k2+=1;
          k1 = sqrt ( k2*k2 + df_tensor_omega0[0][i][j].i*df_tensor_omega0[0][i][j].i ) ;
          refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
          extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
        }else if (SpinP_switch==1 || SpinP_switch==3){
          k2 = df_tensor_omega0[0][i][j].r+df_tensor_omega0[1][i][j].r;
          if (i==j) k2+=1;
          k3 = df_tensor_omega0[0][i][j].i+df_tensor_omega0[1][i][j].i;
          k1 = sqrt ( k2*k2 + k3*k3 ) ; //// check
          refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
          extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
        }
        k1 = (refractive_index[l]-1)*(refractive_index[l]-1);
        k2 = (refractive_index[l]+1)*(refractive_index[l]+1);
        k3 = extinction_coefficient[l]*extinction_coefficient[l];
        reflection_coefficient[l] = 100 * ( k1 + k3 ) / ( k2 + k3 ) ; 
      }
    }
    fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",0.0,reflection_coefficient[0],reflection_coefficient[1],reflection_coefficient[2],reflection_coefficient[3],reflection_coefficient[4],reflection_coefficient[5],reflection_coefficient[6],reflection_coefficient[7],reflection_coefficient[8]);
    for (k=0;k<CDDF_freq_grid_number;k++){
      omega=(k+1)*step*eV2Hartree;
      for (i=0;i<d;i++){
        for (j=0;j<d;j++){ 
          l=i*d+j;
          if (SpinP_switch==0){
            k2 = df_tensor[0][i][j][k].r;
            if (i==j) k2+=1;
            k1 = sqrt ( k2*k2 + df_tensor[0][i][j][k].i*df_tensor[0][i][j][k].i ) ;
            refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
            extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
          }else if (SpinP_switch==1 || SpinP_switch==3){
            k2 = df_tensor[0][i][j][k].r+df_tensor[1][i][j][k].r;
            if (i==j) k2+=1;
            k3 = df_tensor[0][i][j][k].i+df_tensor[1][i][j][k].i;
            k1 = sqrt ( k2*k2 + k3*k3 ) ; //// check
            refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
            extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
          }
          k1 = (refractive_index[l]-1)*(refractive_index[l]-1);
          k2 = (refractive_index[l]+1)*(refractive_index[l]+1);
          k3 = extinction_coefficient[l]*extinction_coefficient[l];
          reflection_coefficient[l] = 100 * ( k1 + k3 ) / ( k2 + k3 ) ; 
        }
      }
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",omega,reflection_coefficient[0],reflection_coefficient[1],reflection_coefficient[2],reflection_coefficient[3],reflection_coefficient[4],reflection_coefficient[5],reflection_coefficient[6],reflection_coefficient[7],reflection_coefficient[8]);
    }
    fclose(fp);

    sprintf(fname1,"%s%s.transmission",filepath,filename);
    fp = fopen(fname1,"w");
    fprintf(fp,"# transmission coefficient (%%)\n");
    fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
    fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz\n");
    /* at omega = 0 */
    for (i=0;i<d;i++){
      for (j=0;j<d;j++){ 
        l=i*d+j;
        if (SpinP_switch==0){
          k2 = df_tensor_omega0[0][i][j].r;
          if (i==j) k2++;
          k1 = sqrt ( k2*k2 + df_tensor_omega0[0][i][j].i*df_tensor_omega0[0][i][j].i ) ;
          refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
          extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
        }else if (SpinP_switch==1 || SpinP_switch==3){
          k2 = df_tensor_omega0[0][i][j].r+df_tensor_omega0[1][i][j].r;
          k3 = df_tensor_omega0[0][i][j].i+df_tensor_omega0[1][i][j].i;
          if (i==j) k2++;
          k1 = sqrt ( k2*k2 + k3*k3 ) ;
          refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
          extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
        }
        k1 = (refractive_index[l]-1)*(refractive_index[l]-1);
        k2 = (refractive_index[l]+1)*(refractive_index[l]+1);
        k3 = extinction_coefficient[l]*extinction_coefficient[l];
        transmission_coefficient[l] = 100.0 - 100 * ( k1 + k3 ) / ( k2 + k3 ) ; 
      }
    }
    fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",0.0,transmission_coefficient[0],transmission_coefficient[1],transmission_coefficient[2],transmission_coefficient[3],transmission_coefficient[4],transmission_coefficient[5],transmission_coefficient[6],transmission_coefficient[7],transmission_coefficient[8]);
    for (k=0;k<CDDF_freq_grid_number;k++){
      omega=(k+1)*step*eV2Hartree;
      for (i=0;i<d;i++){
        for (j=0;j<d;j++){ 
          l=i*d+j;
          if (SpinP_switch==0){
            k2 = df_tensor[0][i][j][k].r;
            if (i==j) k2++;
            k1 = sqrt ( k2*k2 + df_tensor[0][i][j][k].i*df_tensor[0][i][j][k].i ) ;
            refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
            extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
          }else if (SpinP_switch==1 || SpinP_switch==3){
            k2 = df_tensor[0][i][j][k].r+df_tensor[1][i][j][k].r;
            k3 = df_tensor[0][i][j][k].i+df_tensor[1][i][j][k].i;
            if (i==j) k2++;
            k1 = sqrt ( k2*k2 + k3*k3 ) ;
            refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
            extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
          }
          k1 = (refractive_index[l]-1)*(refractive_index[l]-1);
          k2 = (refractive_index[l]+1)*(refractive_index[l]+1);
          k3 = extinction_coefficient[l]*extinction_coefficient[l];
          transmission_coefficient[l] = 100.0 - 100 * ( k1 + k3 ) / ( k2 + k3 ) ; 
        }
      }

      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",omega,transmission_coefficient[0],transmission_coefficient[1],transmission_coefficient[2],transmission_coefficient[3],transmission_coefficient[4],transmission_coefficient[5],transmission_coefficient[6],transmission_coefficient[7],transmission_coefficient[8]);
    }
    fclose(fp);
  }
}

void Calc_band_optical_col_1(double kx,double ky,double kz,int spin_index,int n,double* EIGEN, dcomplex** coeff, double* fd_dist,double ChemP){
  double k2,k3,omega,p1,p5,p2,p3,p4,kRn,sx;
  double dFD,dEuv,dEuv2,dEuv_p_omega,dFDdEuvdeno,dFDdEuvdeno2;
  double dE_limit = 1.0E-6; /* for intra-band and degenerate states */
  int Gc_AN,h_AN,Gh_AN,Rn,l1,l2,l3,o,state_m,state_n,tno0,tno1;

  tnos = CDDF_max_unoccupied_state;

  /* ### MPI for bands ### */
  n0 = tnos*tnos;
  unsigned int n1 = n0/numprocs2, n2 = n0%numprocs2, n3[numprocs2], n5[numprocs2+1], ui, ul, uk; /* for step 1 */

  if (numprocs2>1){

    /* initialization by T. Ozaki */
    for (ui=0; ui<=numprocs2; ui++) n5[ui] = 0; 

    for (ui=0; ui<numprocs2; ui++){
      n3[ui]=n1;
      if (ui<n2) n3[ui]++;
    }

    uk=0; ul=n3[0]; n5[0]=0; n5[1]=ul;
    for (ui=0; ui<n0; ui++){
      if (ui>=ul){
        uk++;
        ul += n3[uk];
        n5[uk+1] = ul;
      }
    }

  }
  else{
    n5[0] = 0; 
    n5[1] = n0;
    n3[0] = n0;
  }

  /* initial atomic orbital index */
  int atomic_orbital_index[atomnum];
  atomic_orbital_index[0]=0;
  i=0;
  for (Gc_AN=1;Gc_AN<atomnum;Gc_AN++){
    j = Spe_Total_CNO[ WhatSpecies[Gc_AN] ];
    i += j;
    atomic_orbital_index[Gc_AN]=i;
  }

  int numm=3;
  if (CDDF_approach==1) numm=9;

  for (ul=n5[myid2]; ul<n5[myid2+1]; ul++){

    state_m = ul/tnos; /* occupied state */
    state_n = ul%tnos; /* unoccupied state */

    double dfd_x_kw = fd_dist[state_n] - fd_dist[state_m];
    if (dfd_x_kw==0.0) continue; /* check */

    int m1=state_m+1,n1=state_n+1;

    /* initialize */
    double MME[6]={0.0,0.0,0.0,0.0,0.0,0.0}; /* for saving momentum matrix elements, i.e. (x_re,y_re,z_re,x_im,y_im,z_im) */
    double tMME[18]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){ /* the third loop  - index of the first atoms within primitive cell */
      tno0  = Spe_Total_CNO[ WhatSpecies[Gc_AN] ];
      int o1=atomic_orbital_index[Gc_AN-1]+1;

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){ /* the fourth loop - index of the second atoms within primtive cell */
        Gh_AN = natn[Gc_AN][h_AN];
        tno1  = Spe_Total_CNO[ WhatSpecies[Gh_AN] ];
        int o2=atomic_orbital_index[Gh_AN-1]+1;

        Rn=ncn[Gc_AN][h_AN]; l1 = atv_ijk[Rn][1]; l2 = atv_ijk[Rn][2]; l3 = atv_ijk[Rn][3];

        kRn = ( kx*(double)l1 + ky*(double)l2 + kz*(double)l3 )*PIx2;

        /* phase factor */
        //phase_factor = Complex( cos(PIx2*kRn) , sin(PIx2*kRn) );
        /* phase factor * i = ( cos(a) + i sin(a) ) * i = -sin(a) + i cos(a) */
        dcomplex phase_factor = Complex( -sin(kRn) , cos(kRn) );

        dcomplex phase_factorR[3];
        if (CDDF_approach==1){
          phase_factorR[0] = RCmul( l1*tv[1][1] + l2*tv[2][1] + l3*tv[3][1] , phase_factor );
          phase_factorR[1] = RCmul( l1*tv[1][2] + l2*tv[2][2] + l3*tv[3][2] , phase_factor );
          phase_factorR[2] = RCmul( l1*tv[1][3] + l2*tv[2][3] + l3*tv[3][3] , phase_factor );
        }

        dcomplex MME_x_cui_x_cuj_x_pf[numm];
        for (k=0; k<numm; k++) MME_x_cui_x_cuj_x_pf[k]=Complex(0.0,0.0);

        for (k=0; k< tno0 ; k++){ /* the fifth loop  - orbital index k within atom i */

          dcomplex MME_x_cuj_x_pf[numm];
          for (o=0; o<numm; o++) MME_x_cuj_x_pf[o]=Complex(0.0,0.0); 

          dcomplex cuj_x_pf = Complex(0.0,0.0),cuj_x_pfR[3];
          if (CDDF_approach==1){
            cuj_x_pfR[0] = Complex(0.0,0.0);
            cuj_x_pfR[1] = Complex(0.0,0.0);
            cuj_x_pfR[2] = Complex(0.0,0.0);
          }

          for (o=0; o< tno1 ; o++){ /* the fixth loop  - orbital index o within atom j */
            /* step (1) */
            dcomplex coefficient = coeff[o2+o][n1] ;
            cuj_x_pf = Cmul( coefficient, phase_factor ) ; /* LCAO[ atomic orbitals ][ states ] */
            if (CDDF_approach==1){
              cuj_x_pfR[0] = Cmul( coefficient , phase_factorR[0] ) ;
              cuj_x_pfR[1] = Cmul( coefficient , phase_factorR[1] ) ;
              cuj_x_pfR[2] = Cmul( coefficient , phase_factorR[2] ) ;
            }

            /* step (2) */
            if (CDDF_approach==0){
              MME_x_cuj_x_pf[0] = Cadd( MME_x_cuj_x_pf[0], RCmul( MME_allorb[Gc_AN][h_AN][k][o][0], cuj_x_pf ) ) ;
              MME_x_cuj_x_pf[1] = Cadd( MME_x_cuj_x_pf[1], RCmul( MME_allorb[Gc_AN][h_AN][k][o][1], cuj_x_pf ) ) ;
              MME_x_cuj_x_pf[2] = Cadd( MME_x_cuj_x_pf[2], RCmul( MME_allorb[Gc_AN][h_AN][k][o][2], cuj_x_pf ) ) ;
            }else if (CDDF_approach==1){
              double H_matrix_element =   H_all[spin_index][Gc_AN][h_AN][k][o];
              double S_matrix_element = OLP_all[Gc_AN][h_AN][k][o];
              MME_x_cuj_x_pf[0] = Cadd( MME_x_cuj_x_pf[0], RCmul( H_matrix_element, cuj_x_pfR[0]) ) ;
              MME_x_cuj_x_pf[1] = Cadd( MME_x_cuj_x_pf[1], RCmul( H_matrix_element, cuj_x_pfR[1]) ) ;
              MME_x_cuj_x_pf[2] = Cadd( MME_x_cuj_x_pf[2], RCmul( H_matrix_element, cuj_x_pfR[2]) ) ;
              MME_x_cuj_x_pf[3] = Cadd( MME_x_cuj_x_pf[3], RCmul( S_matrix_element, cuj_x_pfR[0]) ) ;
              MME_x_cuj_x_pf[4] = Cadd( MME_x_cuj_x_pf[4], RCmul( S_matrix_element, cuj_x_pfR[1]) ) ;
              MME_x_cuj_x_pf[5] = Cadd( MME_x_cuj_x_pf[5], RCmul( S_matrix_element, cuj_x_pfR[2]) ) ;
              MME_x_cuj_x_pf[6] = Cadd( MME_x_cuj_x_pf[6], RCmul( OLPpox_all[Gc_AN][h_AN][k][o]+Gxyz[Gc_AN][1]*S_matrix_element, cuj_x_pf ) ) ;
              MME_x_cuj_x_pf[7] = Cadd( MME_x_cuj_x_pf[7], RCmul( OLPpoy_all[Gc_AN][h_AN][k][o]+Gxyz[Gc_AN][2]*S_matrix_element, cuj_x_pf ) ) ;
              MME_x_cuj_x_pf[8] = Cadd( MME_x_cuj_x_pf[8], RCmul( OLPpoz_all[Gc_AN][h_AN][k][o]+Gxyz[Gc_AN][3]*S_matrix_element, cuj_x_pf ) ) ;
            }
          } /* 8th loop for orbital index o of atom j */

          /* step (3) */
          dcomplex conjg_lcao_coef = Conjg( coeff[o1+k][m1] ); /* LCAO[ atomic orbitals ][ states ] */

          if (CDDF_approach==0){
            MME_x_cui_x_cuj_x_pf[0] = Cadd( MME_x_cui_x_cuj_x_pf[0], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[0] ) );
            MME_x_cui_x_cuj_x_pf[1] = Cadd( MME_x_cui_x_cuj_x_pf[1], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[1] ) );
            MME_x_cui_x_cuj_x_pf[2] = Cadd( MME_x_cui_x_cuj_x_pf[2], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[2] ) );
          }else if (CDDF_approach==1){
            MME_x_cui_x_cuj_x_pf[0] = Cadd( MME_x_cui_x_cuj_x_pf[0], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[0] ) );
            MME_x_cui_x_cuj_x_pf[1] = Cadd( MME_x_cui_x_cuj_x_pf[1], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[1] ) );
            MME_x_cui_x_cuj_x_pf[2] = Cadd( MME_x_cui_x_cuj_x_pf[2], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[2] ) );
            MME_x_cui_x_cuj_x_pf[3] = Cadd( MME_x_cui_x_cuj_x_pf[3], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[3] ) );
            MME_x_cui_x_cuj_x_pf[4] = Cadd( MME_x_cui_x_cuj_x_pf[4], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[4] ) );
            MME_x_cui_x_cuj_x_pf[5] = Cadd( MME_x_cui_x_cuj_x_pf[5], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[5] ) );
            MME_x_cui_x_cuj_x_pf[6] = Cadd( MME_x_cui_x_cuj_x_pf[6], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[6] ) );
            MME_x_cui_x_cuj_x_pf[7] = Cadd( MME_x_cui_x_cuj_x_pf[7], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[7] ) );
            MME_x_cui_x_cuj_x_pf[8] = Cadd( MME_x_cui_x_cuj_x_pf[8], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[8] ) );
          }
        } /* 7th loop for orbital index k of atom i */

        /* step (4) : momentum matrix elements - MME[ state_m ][ state_n ][ i ] */
        if (CDDF_approach==0){
          MME[0] += MME_x_cui_x_cuj_x_pf[0].i ;
          MME[1] += MME_x_cui_x_cuj_x_pf[1].i ;
          MME[2] += MME_x_cui_x_cuj_x_pf[2].i ;
          MME[3] -= MME_x_cui_x_cuj_x_pf[0].r ;
          MME[4] -= MME_x_cui_x_cuj_x_pf[1].r ;
          MME[5] -= MME_x_cui_x_cuj_x_pf[2].r ; 
        }else if (CDDF_approach==1){
          tMME[ 0] += MME_x_cui_x_cuj_x_pf[0].r ; // Hx.r
          tMME[ 1] += MME_x_cui_x_cuj_x_pf[1].r ; // Hy.r
          tMME[ 2] += MME_x_cui_x_cuj_x_pf[2].r ; // Hz.r
          tMME[ 3] += MME_x_cui_x_cuj_x_pf[3].r ; // Sx.r
          tMME[ 4] += MME_x_cui_x_cuj_x_pf[4].r ; // Sy.r
          tMME[ 5] += MME_x_cui_x_cuj_x_pf[5].r ; // Sz.r
          tMME[ 6] += MME_x_cui_x_cuj_x_pf[6].r ; // x.r
          tMME[ 7] += MME_x_cui_x_cuj_x_pf[7].r ; // y.r
          tMME[ 8] += MME_x_cui_x_cuj_x_pf[8].r ; // z.r
          tMME[ 9] += MME_x_cui_x_cuj_x_pf[0].i ; // Hx.i
          tMME[10] += MME_x_cui_x_cuj_x_pf[1].i ; // Hy.i
          tMME[11] += MME_x_cui_x_cuj_x_pf[2].i ; // Hz.i
          tMME[12] += MME_x_cui_x_cuj_x_pf[3].i ; // Sx.i
          tMME[13] += MME_x_cui_x_cuj_x_pf[4].i ; // Sy.i
          tMME[14] += MME_x_cui_x_cuj_x_pf[5].i ; // Sz.i
          tMME[15] += MME_x_cui_x_cuj_x_pf[6].i ; // x.i
          tMME[16] += MME_x_cui_x_cuj_x_pf[7].i ; // y.i
          tMME[17] += MME_x_cui_x_cuj_x_pf[8].i ; // z.i
        }
      } /* 6th loop for atomic index j */
    } /* 5th loop for atomic index i */
    /* ### end of MME from state m to state n ### */

    dEuv  = EIGEN[m1] - EIGEN[n1] ;

    if (CDDF_approach==1){
      MME[0]=tMME[ 0]-(EIGEN[m1]*tMME[ 3])+(dEuv*tMME[ 6]); // momentum x.r
      MME[1]=tMME[ 1]-(EIGEN[m1]*tMME[ 4])+(dEuv*tMME[ 7]); // momentum y.r
      MME[2]=tMME[ 2]-(EIGEN[m1]*tMME[ 5])+(dEuv*tMME[ 8]); // momentum z.r
      MME[3]=tMME[ 9]-(EIGEN[m1]*tMME[12])+(dEuv*tMME[15]); // momentum x.i
      MME[4]=tMME[10]-(EIGEN[m1]*tMME[13])+(dEuv*tMME[16]); // momentum y.i
      MME[5]=tMME[11]-(EIGEN[m1]*tMME[14])+(dEuv*tMME[17]); // momentum z.i
    }

    dcomplex p6[9];
    p6[0] = Complex( MME[0]*MME[0] + MME[3]*MME[3] , 0.0 );
    p6[1] = Complex( MME[0]*MME[1] + MME[3]*MME[4] , MME[3]*MME[1] - MME[0]*MME[4] );
    p6[2] = Complex( MME[0]*MME[2] + MME[3]*MME[5] , MME[3]*MME[2] - MME[0]*MME[5] );
    p6[3] = Complex( p6[1].r                       , -p6[1].i );
    p6[4] = Complex( MME[1]*MME[1] + MME[4]*MME[4] , 0.0 );
    p6[5] = Complex( MME[1]*MME[2] + MME[4]*MME[5] , MME[4]*MME[2] - MME[1]*MME[5] );
    p6[6] = Complex( p6[2].r                       , -p6[2].i );
    p6[7] = Complex( p6[5].r                       , -p6[5].i );
    p6[8] = Complex( MME[2]*MME[2] + MME[5]*MME[5] , 0.0 );

    sx = (EIGEN[m1] - ChemP) * Beta;
    if (100<sx) sx = 100.0; 
    k2 = exp(sx);
    k3 = - k2 * Beta / ( (1+k2) * (1+k2) ) ;

    for (i=0;i<d;i++){ /* tensor index, i = alpha */
      for (j=0;j<d;j++){ /* tensor index, j = beta */

        int j1 = i*3+j;
        p1 = p6[j1].r*eta ;
        p5 = p6[j1].i*eta ;

        for (k=0; k<CDDF_freq_grid_total_number; k++){ /* scan all positive frequencies */

          omega=(k+1)*step;
          dEuv_p_omega = dEuv + omega ;
       
          /* At the same k-point */
          /* (1) intra-band : state_m == state_n ( dEuv == 0 ) and (2) degenerate states : state_m != state_n && dEuv -> 0 */
          if ( state_m == state_n ){
            dFDdEuvdeno  = - k3 / ( ( omega * omega ) + eta2 ) ;
            dFDdEuvdeno2 = dFDdEuvdeno / omega ;
          }else if ( fabs(dEuv) < dE_limit ){
            dFDdEuvdeno  = k3 / ( ( omega * omega ) + eta2 ) ;
            dFDdEuvdeno2 = dFDdEuvdeno / omega ;
          }
          /* (3) inter-band : state_m != state_n && , dEuv is not close to 0 */
          else{
            dFDdEuvdeno  = dfd_x_kw / ( dEuv * ( ( dEuv_p_omega * dEuv_p_omega ) + eta2 ) );
            dFDdEuvdeno2 = dFDdEuvdeno / dEuv ;
          }

          p3 = p1 - p6[j1].i*dEuv_p_omega ;
          p4 = p6[j1].r*dEuv_p_omega + p5 ;

          /* real part      of conductivity tensor */
          cd_tensor[spin_index][i][j][k].r += dFDdEuvdeno  * p3 ;
          /* imaginery part of conductivity tensor */
          cd_tensor[spin_index][i][j][k].i += dFDdEuvdeno  * p4 ;
          /* real part      of dielectric function */
          df_tensor[spin_index][i][j][k].r += dFDdEuvdeno2 * p4 ;
          /* imaginery part of dielectric function */
          df_tensor[spin_index][i][j][k].i -= dFDdEuvdeno2 * p3 ;
        } /* frequency */
      } /* tensor index, j = beta */
    } /* tensor index, i = alpha */
  } /* ul: state_m and state_n */

}

void Calc_band_optical_noncol_1(double kx,double ky,double kz,int n,double* EIGEN, dcomplex** LCAO, double* fd_dist,double ChemP){
  double k2,k3,omega,p1,p5,p2,p3,p4,kRn,sx;
  double dFD,dEuv,dEuv_p_omega,dFDdEuvdeno,dFDdEuvdeno2;
  double dE_limit = 1.0E-6; /* for intra-band and degenerate states */
  int Gc_AN,h_AN,Gh_AN,Rn,l1,l2,l3,o,sp,state_m,state_n;
  int total_number_of_orbitals = n*0.5; /* for spin-up and spin-down */

  tnos = CDDF_max_unoccupied_state;

  /* ### MPI for bands ### */
  n0 = tnos*tnos;
  unsigned int n1 = n0/numprocs2, n2 = n0%numprocs2, n3[numprocs2], n5[numprocs2+1], ui, ul, uk; /* for step 1 */

  if (numprocs2>1){

    /* initialization by T. Ozaki */
    for (ui=0; ui<=numprocs2; ui++) n5[ui] = 0;

    for (ui=0;ui<numprocs2;ui++){
      n3[ui]=n1;
      if (ui<n2) n3[ui]++;
    }

    uk=0; ul=n3[0]; n5[0]=0; n5[1]=ul;
    for (ui=0;ui<n0;ui++){
      if (ui>=ul){
        uk++;
        ul+=n3[uk];
        n5[uk+1]=ul;
      }
    }
  }else{
    n5[0]=0; n5[1]=n0;
    n3[0]=n0;
  }

  /* initial atomic orbital index */
  int atomic_orbital_index[2][atomnum];
  atomic_orbital_index[0][0]=0; /* for spin-up */
  atomic_orbital_index[1][0]=total_number_of_orbitals; /* for spin-down */
  i=0;
  for (Gc_AN=1;Gc_AN<atomnum;Gc_AN++){
    j = Spe_Total_CNO[ WhatSpecies[Gc_AN] ];
    i += j;
    atomic_orbital_index[0][Gc_AN]=i; /* for spin-up */
    atomic_orbital_index[1][Gc_AN]=i+total_number_of_orbitals; /* for spin-down */
  }


  dcomplex MME_x_cuj_x_pf[9],MME_x_cui_x_cuj_x_pf[9],cuj_x_pf,cuj_x_pfR[3],conjg_lcao_coef,p6[9],phase_factor,phase_factorR[3];

  for (sp=0;sp<total_spins;sp++){ /* for spin-up and spin-down */
    int sp1,sp2;
    if (sp==0){ sp1=0; sp2=0; } // up up
    else if (sp==1){ sp1=1; sp2=1; } // down down
    else if (sp==2){ sp1=0; sp2=1; } // up down
    else if (sp==3){ sp1=1; sp2=0; } // down up

    for (ul=n5[myid2];ul<n5[myid2+1];ul++){ /* occupied state u */
      state_m = ul/tnos; /* occupied state */
      state_n = ul%tnos; /* unoccupied state */

      double dfd_x_kw = fd_dist[state_n] - fd_dist[state_m];

      int m1=state_m+1;

      /* initialize */
      double MME[6]={0.0,0.0,0.0,0.0,0.0,0.0}; // for saving momentum matrix elements, i.e. (x_re,y_re,z_re,x_im,y_im,z_im) */
      double tMME[18]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){ // the third loop  - index of the first atoms within primitive cell */
        int o1=atomic_orbital_index[sp1][Gc_AN-1];

        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){ /* the fourth loop - index of the second atoms within primtive cell */
          Gh_AN = natn[Gc_AN][h_AN];
          int o2=atomic_orbital_index[sp2][Gh_AN-1];

          Rn=ncn[Gc_AN][h_AN]; l1 = atv_ijk[Rn][1]; l2 = atv_ijk[Rn][2]; l3 = atv_ijk[Rn][3];
          kRn = kx*(double)l1 + ky*(double)l2 + kz*(double)l3;

          /* phase factor */
          /* double si = sin(PIx2*kRn),co = cos(PIx2*kRn); */ /* for cluster si = 0 , phase_factor = Complex (1,0) */
          phase_factor = Complex( cos(PIx2*kRn) , sin(PIx2*kRn) );
          phase_factorR[0] = RCmul( l1*tv[1][1] + l2*tv[2][1] + l3*tv[3][1] , phase_factor );
          phase_factorR[1] = RCmul( l1*tv[1][2] + l2*tv[2][2] + l3*tv[3][2] , phase_factor );
          phase_factorR[2] = RCmul( l1*tv[1][3] + l2*tv[2][3] + l3*tv[3][3] , phase_factor );

          for (k=0; k<9; k++) MME_x_cui_x_cuj_x_pf[k] = Complex(0.0,0.0);

          for (k=0; k< Spe_Total_CNO[ WhatSpecies[Gc_AN] ] ; k++){ /* the fifth loop  - orbital index k within atom i */

            for (o=0; o<9; o++) MME_x_cuj_x_pf[o] = Complex(0.0,0.0);

            cuj_x_pf = Complex(0.0,0.0);
            for (o=0; o<3; o++) cuj_x_pfR[o] = Complex(0.0,0.0);

            /* step (3) */
            conjg_lcao_coef = Conjg( LCAO[state_m][o1+k] ); /* LCAO[ states ][ atomic orbitals ] */

            for (o=0; o< Spe_Total_CNO[ WhatSpecies[Gh_AN] ] ; o++){ /* the fixth loop  - orbital index o within atom j */
              /* step (1) */
              dcomplex coeff=LCAO[state_n][o2+o];
              cuj_x_pf = Cmul( coeff, phase_factor ) ; /* LCAO[ states ][ atomic orbitals ] */
              cuj_x_pfR[0] = Cmul( coeff, phase_factorR[0] ) ;
              cuj_x_pfR[1] = Cmul( coeff, phase_factorR[1] ) ;
              cuj_x_pfR[2] = Cmul( coeff, phase_factorR[2] ) ;

              /* step (2) */
              /* for n cpus after broadcasting H/S/position matrix elements */
              dcomplex H_matrix_element=Complex(0.0,0.0);
              if (sp==0){ // spin up up
                H_matrix_element = Complex( H_all[0][Gc_AN][h_AN][k][o] , iH_all[0][Gc_AN][h_AN][k][o] );
              }else if (sp==1){ // spin down down
                H_matrix_element = Complex( H_all[1][Gc_AN][h_AN][k][o] , iH_all[1][Gc_AN][h_AN][k][o] );
              }else if (sp==2){ // spin up down
                H_matrix_element = Complex( H_all[2][Gc_AN][h_AN][k][o] , H_all[3][Gc_AN][h_AN][k][o] + iH_all[2][Gc_AN][h_AN][k][o] );
              }else if (sp==3){ // spin down up
                H_matrix_element = Conjg( Complex( H_all[2][Gc_AN][h_AN][k][o] , H_all[3][Gc_AN][h_AN][k][o] + iH_all[2][Gc_AN][h_AN][k][o] ) );
              }

              double S_matrix_element = OLP_all[Gc_AN][h_AN][k][o];
              MME_x_cuj_x_pf[0] = Cadd( MME_x_cuj_x_pf[0],  Cmul( H_matrix_element, cuj_x_pfR[0]) ) ;
              MME_x_cuj_x_pf[1] = Cadd( MME_x_cuj_x_pf[1],  Cmul( H_matrix_element, cuj_x_pfR[1]) ) ;
              MME_x_cuj_x_pf[2] = Cadd( MME_x_cuj_x_pf[2],  Cmul( H_matrix_element, cuj_x_pfR[2]) ) ;
              MME_x_cuj_x_pf[3] = Cadd( MME_x_cuj_x_pf[3], RCmul( S_matrix_element, cuj_x_pfR[0]) ) ;
              MME_x_cuj_x_pf[4] = Cadd( MME_x_cuj_x_pf[4], RCmul( S_matrix_element, cuj_x_pfR[1]) ) ;
              MME_x_cuj_x_pf[5] = Cadd( MME_x_cuj_x_pf[5], RCmul( S_matrix_element, cuj_x_pfR[2]) ) ;
              MME_x_cuj_x_pf[6] = Cadd( MME_x_cuj_x_pf[6], RCmul( OLPpox_all[Gc_AN][h_AN][k][o]+Gxyz[Gc_AN][1]*S_matrix_element, cuj_x_pf ) ) ;
              MME_x_cuj_x_pf[7] = Cadd( MME_x_cuj_x_pf[7], RCmul( OLPpoy_all[Gc_AN][h_AN][k][o]+Gxyz[Gc_AN][2]*S_matrix_element, cuj_x_pf ) ) ;
              MME_x_cuj_x_pf[8] = Cadd( MME_x_cuj_x_pf[8], RCmul( OLPpoz_all[Gc_AN][h_AN][k][o]+Gxyz[Gc_AN][3]*S_matrix_element, cuj_x_pf ) ) ;
            } /* 8th loop for orbital index o of atom j */

            MME_x_cui_x_cuj_x_pf[0] = Cadd( MME_x_cui_x_cuj_x_pf[0], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[0] ) );
            MME_x_cui_x_cuj_x_pf[1] = Cadd( MME_x_cui_x_cuj_x_pf[1], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[1] ) );
            MME_x_cui_x_cuj_x_pf[2] = Cadd( MME_x_cui_x_cuj_x_pf[2], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[2] ) );
            MME_x_cui_x_cuj_x_pf[3] = Cadd( MME_x_cui_x_cuj_x_pf[3], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[3] ) );
            MME_x_cui_x_cuj_x_pf[4] = Cadd( MME_x_cui_x_cuj_x_pf[4], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[4] ) );
            MME_x_cui_x_cuj_x_pf[5] = Cadd( MME_x_cui_x_cuj_x_pf[5], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[5] ) );
            MME_x_cui_x_cuj_x_pf[6] = Cadd( MME_x_cui_x_cuj_x_pf[6], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[6] ) );
            MME_x_cui_x_cuj_x_pf[7] = Cadd( MME_x_cui_x_cuj_x_pf[7], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[7] ) );
            MME_x_cui_x_cuj_x_pf[8] = Cadd( MME_x_cui_x_cuj_x_pf[8], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[8] ) );
          } /* 7th loop for orbital index k of atom i */

          /* step (4) : momentum matrix elements - MME[ state_m ][ state_n ][ i ] */
          tMME[ 0] += MME_x_cui_x_cuj_x_pf[0].i ; // Hx.r
          tMME[ 1] += MME_x_cui_x_cuj_x_pf[1].i ; // Hy.r
          tMME[ 2] += MME_x_cui_x_cuj_x_pf[2].i ; // Hz.r
          tMME[ 3] += MME_x_cui_x_cuj_x_pf[3].i ; // Sx.r
          tMME[ 4] += MME_x_cui_x_cuj_x_pf[4].i ; // Sy.r
          tMME[ 5] += MME_x_cui_x_cuj_x_pf[5].i ; // Sz.r
          tMME[ 6] += MME_x_cui_x_cuj_x_pf[6].i ; // x.r
          tMME[ 7] += MME_x_cui_x_cuj_x_pf[7].i ; // y.r
          tMME[ 8] += MME_x_cui_x_cuj_x_pf[8].i ; // z.r
          tMME[ 9] -= MME_x_cui_x_cuj_x_pf[0].r ; // Hx.i
          tMME[10] -= MME_x_cui_x_cuj_x_pf[1].r ; // Hy.i
          tMME[11] -= MME_x_cui_x_cuj_x_pf[2].r ; // Hz.i
          tMME[12] -= MME_x_cui_x_cuj_x_pf[3].r ; // Sx.i
          tMME[13] -= MME_x_cui_x_cuj_x_pf[4].r ; // Sy.i
          tMME[14] -= MME_x_cui_x_cuj_x_pf[5].r ; // Sz.i
          tMME[15] -= MME_x_cui_x_cuj_x_pf[6].r ; // x.i
          tMME[16] -= MME_x_cui_x_cuj_x_pf[7].r ; // y.i
          tMME[17] -= MME_x_cui_x_cuj_x_pf[8].r ; // z.i
        } /* 6th loop for atomic index j */
      } /* 5th loop for atomic index i */
      /* ### end of MME from state m to state n ### */

      dEuv = EIGEN[m1] - EIGEN[state_n+1] ;

      MME[0]=tMME[ 0]-(EIGEN[m1]*tMME[ 3])+(dEuv*tMME[ 6]); // momentum x.r
      MME[1]=tMME[ 1]-(EIGEN[m1]*tMME[ 4])+(dEuv*tMME[ 7]); // momentum y.r
      MME[2]=tMME[ 2]-(EIGEN[m1]*tMME[ 5])+(dEuv*tMME[ 8]); // momentum z.r
      MME[3]=tMME[ 9]-(EIGEN[m1]*tMME[12])+(dEuv*tMME[15]); // momentum x.i
      MME[4]=tMME[10]-(EIGEN[m1]*tMME[13])+(dEuv*tMME[16]); // momentum y.i
      MME[5]=tMME[11]-(EIGEN[m1]*tMME[14])+(dEuv*tMME[17]); // momentum z.i


      p6[0] = Complex( MME[0]*MME[0] + MME[3]*MME[3] , 0.0 );
      p6[1] = Complex( MME[0]*MME[1] + MME[3]*MME[4] , MME[3]*MME[1] - MME[0]*MME[4] );
      p6[2] = Complex( MME[0]*MME[2] + MME[3]*MME[5] , MME[3]*MME[2] - MME[0]*MME[5] );
      p6[3] = Complex( p6[1].r                       , -p6[1].i );
      p6[4] = Complex( MME[1]*MME[1] + MME[4]*MME[4] , 0.0 );
      p6[5] = Complex( MME[1]*MME[2] + MME[4]*MME[5] , MME[4]*MME[2] - MME[1]*MME[5] );
      p6[6] = Complex( p6[2].r                       , -p6[2].i  );
      p6[7] = Complex( p6[5].r                       , -p6[5].i );
      p6[8] = Complex( MME[2]*MME[2] + MME[5]*MME[5] , 0.0 );

      sx = (EIGEN[m1] - ChemP) * Beta;
      if (100.0<sx) sx = 100.0; 
      k2 = exp(sx);
      k3 = - k2 * Beta / ( (1+k2) * (1+k2) ) ;

      for (i=0;i<d;i++){ /* tensor index, i = alpha */
        for (j=0;j<d;j++){ /* tensor index, j = beta */

          int j1 = i*3+j;
          p1 = p6[j1].r*eta ;
          p5 = p6[j1].i*eta ;

          for (k=0;k<CDDF_freq_grid_total_number;k++){ /* scan all positive frequencies */

            omega=(k+1)*step;
            dEuv_p_omega = dEuv + omega ;
         
            /* At the same k-point */
            /* (1) intra-band : state_m == state_n ( dEuv == 0 ) and (2) degenerate states : state_m != state_n && dEuv -> 0 */
            if ( state_m == state_n ){
              dFDdEuvdeno  = - k3 / ( ( omega * omega ) + eta2 ) ;
              dFDdEuvdeno2 = dFDdEuvdeno / omega ;
            }
            else if ( fabs(dEuv) < dE_limit ){
              dFDdEuvdeno  = k3 / ( ( omega * omega ) + eta2 ) ;
              dFDdEuvdeno2 = dFDdEuvdeno / omega ;
            }
            /* (3) inter-band : state_m != state_n && , dEuv is not close to 0 */
            else{
              dFDdEuvdeno  = dfd_x_kw / ( dEuv * ( ( dEuv_p_omega * dEuv_p_omega ) + eta2 ) );
              dFDdEuvdeno2 = dFDdEuvdeno / dEuv ;
            }

            p3 = p1 - p6[j1].i*dEuv_p_omega ;
            p4 = p6[j1].r*dEuv_p_omega + p5 ;

            /* real part of conductivity tensor */
            cd_tensor[sp][i][j][k].r += dFDdEuvdeno  * p3 ;
            /* imaginery part of conductivity tensor */
            cd_tensor[sp][i][j][k].i += dFDdEuvdeno  * p4 ;
            /* real part of dielectric function */
            df_tensor[sp][i][j][k].r += dFDdEuvdeno2 * p4 ;
            /* imaginery part of dielectric function */
            df_tensor[sp][i][j][k].i -= dFDdEuvdeno2 * p3 ;
          } /* frequency */
        } /* tensor index, j = beta */
      } /* tensor index, i = alpha */
    } /* state_m and state_n */
  } /* spin */
}


void Calc_optical_col_2(int n,double sum_weights){
  Free_optical_1(); /* free Nabra matrix elements, i.e. MME_allorb array. */

  int numprocs,myid,sp,o;
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  double q2=conductivity_unit/(Cell_Volume*sum_weights), q=q2/dielectricless_unit; /* unit */
  double *collected_temp_tensor = (double*)malloc(sizeof(double)*(CDDF_freq_grid_total_number*4) );
  double *temp_tensor = (double*)malloc(sizeof(double)*(CDDF_freq_grid_total_number*4) );

  l = 2*CDDF_freq_grid_total_number; 
  o = l + CDDF_freq_grid_total_number;

  for (sp=0;sp<total_spins;sp++){ /* tensor index, i = alpha */
    for (i=0;i<d;i++){ /* tensor index, i = alpha */
      for (j=0;j<d;j++){ /* tensor index, j = beta */
        for (k=0; k<CDDF_freq_grid_total_number; k++){ /* scan all frequencies */

          /* save conductivity tensor */
          cd_tensor[sp][i][j][k].r *= q2 ; /* conductivity unit = (Ohm m)^{-1} */
          cd_tensor[sp][i][j][k].i *= q2 ; /* conductivity unit = (Ohm m)^{-1} */

          /* calculate dielectric function */
          /* the second term */
          df_tensor[sp][i][j][k].r *= q ;
          df_tensor[sp][i][j][k].i *= q ;

          /* the first term */
          /*if (i==j) df_tensor[kp][i][j][k].r += 1; */

          /* store data at each cpus, except Host_ID */
          collected_temp_tensor[k                            ] = 0.0;
          collected_temp_tensor[CDDF_freq_grid_total_number+k] = 0.0;
          collected_temp_tensor[l+k                          ] = 0.0;
          collected_temp_tensor[o+k                          ] = 0.0;
          temp_tensor[k                            ] = cd_tensor[sp][i][j][k].r;
          temp_tensor[CDDF_freq_grid_total_number+k] = cd_tensor[sp][i][j][k].i;
          temp_tensor[l+k                          ] = df_tensor[sp][i][j][k].r;
          temp_tensor[o+k                          ] = df_tensor[sp][i][j][k].i;
        }

        MPI_Reduce(temp_tensor, collected_temp_tensor, CDDF_freq_grid_total_number*4, MPI_DOUBLE, MPI_SUM, 0, mpi_comm_level1);

        /*restore data of conductivity and dielectric function */
        for (k=0;k<CDDF_freq_grid_total_number;k++){
          cd_tensor[sp][i][j][k] = Complex(collected_temp_tensor[  k],collected_temp_tensor[CDDF_freq_grid_total_number+k]);
          df_tensor[sp][i][j][k] = Complex(collected_temp_tensor[l+k],collected_temp_tensor[o+k]);
        }

        /* linear extrapolation at omega = 0 for conductivity and dielectric function (and assume x1-x0 = x2-x1) */
        /* y(0) = y1 + (y2-y1)(0-x1)/(x2-x1) = y1 + (y1-y2)*x1/(x2-x1) = y1 + (y1-y2)*dx/dx = y1 + (y1-y2)*1 = y1 + (y1-y2) = 2*y1 - y2 */
        cd_tensor_omega0[sp][i][j].r = 2.0*cd_tensor[sp][i][j][0].r - cd_tensor[sp][i][j][1].r;
        cd_tensor_omega0[sp][i][j].i = 2.0*cd_tensor[sp][i][j][0].i - cd_tensor[sp][i][j][1].i;
        df_tensor_omega0[sp][i][j].r = 2.0*df_tensor[sp][i][j][0].r - df_tensor[sp][i][j][1].r;
        df_tensor_omega0[sp][i][j].i = 2.0*df_tensor[sp][i][j][0].i - df_tensor[sp][i][j][1].i;
      } /* tensor index, j = beta */
    } /* tensor index, i = alpha */
  } /* sp */

  free(collected_temp_tensor);
  free(temp_tensor);

  /* print out */
  if (myid==Host_ID) Print_optical(1,1,1);
  Free_optical_2(n);
}

void Calc_optical_noncol_2(int n,double sum_weights){
  Free_optical_1();

  int numprocs,myid,sp,o;
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  double q2=conductivity_unit/(Cell_Volume*sum_weights), q=q2/dielectricless_unit; /* unit */
  double *collected_temp_tensor = (double*)malloc(sizeof(double)*(CDDF_freq_grid_total_number*4) );
  double *temp_tensor = (double*)malloc(sizeof(double)*(CDDF_freq_grid_total_number*4) );

  l=2*CDDF_freq_grid_total_number; o=l+CDDF_freq_grid_total_number;
  for (sp=0;sp<total_spins;sp++){ /* tensor index, i = alpha */
    for (i=0;i<d;i++){ /* tensor index, i = alpha */
      for (j=0;j<d;j++){ /* tensor index, j = beta */
        for (k=0;k<CDDF_freq_grid_total_number;k++){ /* scan all frequencies */
          /* save conductivity tensor */
          cd_tensor[sp][i][j][k].r *= q2 ; /* conductivity unit = (Ohm m)^{-1} */
          cd_tensor[sp][i][j][k].i *= q2 ; /* conductivity unit = (Ohm m)^{-1} */

          /* calculate dielectric function */
          /* the second term */
          df_tensor[sp][i][j][k].r *= q ;
          df_tensor[sp][i][j][k].i *= q ;

          /* the first term */
          /*if (i==j) df_tensor[kp][i][j][k].r += 1; */

          /* store data at each cpus, except Host_ID */
          collected_temp_tensor[k]=0.0;
          collected_temp_tensor[CDDF_freq_grid_total_number+k]=0.0;
          collected_temp_tensor[l+k]=0.0;
          collected_temp_tensor[o+k]=0.0;
          temp_tensor[  k] = cd_tensor[sp][i][j][k].r;
          temp_tensor[CDDF_freq_grid_total_number+k] = cd_tensor[sp][i][j][k].i;
          temp_tensor[l+k] = df_tensor[sp][i][j][k].r;
          temp_tensor[o+k] = df_tensor[sp][i][j][k].i;
        }

        MPI_Reduce(temp_tensor, collected_temp_tensor, CDDF_freq_grid_total_number*4, MPI_DOUBLE, MPI_SUM, 0, mpi_comm_level1);

        /*restore data of conductivity and dielectric function */
        for (k=0;k<CDDF_freq_grid_total_number;k++){
          cd_tensor[sp][i][j][k] = Complex(collected_temp_tensor[  k],collected_temp_tensor[CDDF_freq_grid_total_number+k]);
          df_tensor[sp][i][j][k] = Complex(collected_temp_tensor[l+k],collected_temp_tensor[o+k]);
        }

        /* linear extrapolation at omega = 0 for conductivity and dielectric function (and assume x1-x0 = x2-x1) */
        /* y(0) = y1 + (y2-y1)(0-x1)/(x2-x1) = y1 + (y1-y2)*x1/(x2-x1) = y1 + (y1-y2)*dx/dx = y1 + (y1-y2)*1 = y1 + (y1-y2) = 2*y1 - y2 */
        cd_tensor_omega0[sp][i][j].r = 2.0*cd_tensor[sp][i][j][0].r - cd_tensor[sp][i][j][1].r;
        cd_tensor_omega0[sp][i][j].i = 2.0*cd_tensor[sp][i][j][0].i - cd_tensor[sp][i][j][1].i;
        df_tensor_omega0[sp][i][j].r = 2.0*df_tensor[sp][i][j][0].r - df_tensor[sp][i][j][1].r;
        df_tensor_omega0[sp][i][j].i = 2.0*df_tensor[sp][i][j][0].i - df_tensor[sp][i][j][1].i;
      } /* tensor index, j = beta */
    } /* tensor index, i = alpha */
  } /* sp */

  free(collected_temp_tensor);
  free(temp_tensor);
 
  if (myid==Host_ID) Print_optical(1,1,1); /* (print CD=1=yes, print DF=1=yes, print optical properties=1=yes) */
  Free_optical_2(n);
}
