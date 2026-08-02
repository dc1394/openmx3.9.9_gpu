#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "openmx_common.h"
#include "mpi.h"


static void Simple_Mixing_H(int MD_iter, int SCF_iter, int SCF_iter0 );
static void Pulay_Mixing_H(int MD_iter, int SCF_iter, int SCF_iter0 );
static void Pulay_Mixing_H_MultiSecant(int MD_iter, int SCF_iter, int SCF_iter0 );
static void Pulay_Mixing_H_with_One_Shot_Hessian(int MD_iter, int SCF_iter, int SCF_iter0 );
static void Inverse(int n, double **a, double **ia);


double Mixing_H( int MD_iter, int SCF_iter, int SCF_iter0 )
{
  double time0;
  double TStime,TEtime;
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /*******************************************************
    Simple mixing 
  *******************************************************/

  if ( SCF_iter<=(Pulay_SCF-1) ){
    Simple_Mixing_H( MD_iter, SCF_iter, SCF_iter0 );
  }

  /*******************************************************
    Pulay's method:
    Residual Minimazation Method (RMM) using
    Direct Inversion in the Iterative Subspace (DIIS)
  *******************************************************/

  else{

    Pulay_Mixing_H( MD_iter, SCF_iter, SCF_iter0 );

    /*
    Pulay_Mixing_H_with_One_Shot_Hessian( MD_iter, SCF_iter, SCF_iter0 );
    */

    /*
    Pulay_Mixing_H_MultiSecant( MD_iter, SCF_iter, SCF_iter0 );
    */
  }

  /* if SCF_iter0==1, then NormRD[0]=1 */
  if (SCF_iter0==1) NormRD[0] = 1.0;

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
} 





void Pulay_Mixing_H_MultiSecant(int MD_iter, int SCF_iter, int SCF_iter0 )
{
  int Mc_AN,Gc_AN,Cwan,Hwan,h_AN,Gh_AN,i,j,spin;
  int dim,m,n,flag_nan;
  double sum,my_sum,tmp1,tmp2,alpha;
  double r,r10,r11,r12,r13,r20,r21,r22;
  double h,h10,h11,h12,h13,h20,h21,h22;
  double my_sy,my_yy,sy,yy,norm,s,y,or,al,be;
  double **A,**IA,*coes,*coes2,*ror;
  char nanchar[300];

  /****************************************************
       determination of dimension of the subspace
  ****************************************************/

  if (SCF_iter<=Num_Mixing_pDM) dim = SCF_iter-1;
  else                          dim = Num_Mixing_pDM;

  /****************************************************
                allocation of arrays 
  ****************************************************/

  coes = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  coes2 = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  ror = (double*)malloc(sizeof(double)*List_YOUSO[39]);

  A = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    A[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  IA = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    IA[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  /****************************************************
                 shift the residual H
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      ResidualH1[m][0][Mc_AN][h_AN][i][j] = ResidualH1[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH1[m][1][Mc_AN][h_AN][i][j] = ResidualH1[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH1[m][2][Mc_AN][h_AN][i][j] = ResidualH1[m-1][2][Mc_AN][h_AN][i][j];
	      ResidualH1[m][3][Mc_AN][h_AN][i][j] = ResidualH1[m-1][3][Mc_AN][h_AN][i][j];

	      ResidualH2[m][0][Mc_AN][h_AN][i][j] = ResidualH2[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH2[m][1][Mc_AN][h_AN][i][j] = ResidualH2[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH2[m][2][Mc_AN][h_AN][i][j] = ResidualH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    ResidualH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j] - HisH1[0][0][Mc_AN][h_AN][i][j];
	    ResidualH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j] - HisH1[0][1][Mc_AN][h_AN][i][j];
	    ResidualH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j] - HisH1[0][2][Mc_AN][h_AN][i][j];
	    ResidualH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j] - HisH1[0][3][Mc_AN][h_AN][i][j];

	    ResidualH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j] - HisH2[0][0][Mc_AN][h_AN][i][j];
	    ResidualH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j] - HisH2[0][1][Mc_AN][h_AN][i][j];
	    ResidualH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j] - HisH2[0][2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else{

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		ResidualH1[m][spin][Mc_AN][h_AN][i][j] = ResidualH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
  	      ResidualH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j] - HisH1[0][spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /****************************************************
          calculation of the residual matrix
  ****************************************************/

  for (m=0; m<dim; m++){
    for (n=0; n<dim; n++){

      my_sum = 0.0;

      if (SpinP_switch==3){

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

		tmp1 = ResidualH1[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][0][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][1][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][2][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][3][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][3][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][0][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][1][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][2][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 
	      }
	    }
	  }
	}

      } /* if (SpinP_switch==3 */

      else{

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	    Gc_AN = M2G[Mc_AN];    
	    Cwan = WhatSpecies[Gc_AN];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){
		  tmp1 = ResidualH1[m][spin][Mc_AN][h_AN][i][j];
		  tmp2 = ResidualH1[n][spin][Mc_AN][h_AN][i][j];
                  my_sum += tmp1*tmp2; 
		}
	      }
	    }
	  }
	}

      } /* else */

      MPI_Allreduce(&my_sum, &A[m][n], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
      A[n][m] = A[m][n];

    } /* n */
  } /* m */

  NormRD[0] = A[0][0]/(double)atomnum;

  for (m=1; m<=dim; m++){
    A[m-1][dim] = -1.0;
    A[dim][m-1] = -1.0;
  }
  A[dim][dim] = 0.0;

  Inverse(dim,A,IA);

  for (m=1; m<=dim; m++){
    coes[m] = -IA[m-1][dim];
  }

  /****************************************************
            check "nan", "NaN", "inf" or "Inf"
  ****************************************************/

  flag_nan = 0;
  for (m=1; m<=dim; m++){

    sprintf(nanchar,"%8.4f",coes[m]);
    if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	|| strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

      flag_nan = 1;
    }
  }

  if (flag_nan==1){
    for (m=1; m<=dim; m++){
      coes[m] = 0.0;
    }
    coes[1] = 0.05;
    coes[2] = 0.95;
  }

  /****************************************************
      calculation of optimum residual Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            r10 = 0.0; 
            r11 = 0.0;
            r12 = 0.0;
            r13 = 0.0;

            r20 = 0.0; 
            r21 = 0.0;
            r22 = 0.0;

	    for (m=0; m<dim; m++){

	      r10 += ResidualH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r11 += ResidualH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r12 += ResidualH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      r13 += ResidualH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      r20 += ResidualH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r21 += ResidualH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r22 += ResidualH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	    }

            /* optimum Residual H is stored in ResidualH1[dim+1] and ResidualH2[dim+1] */

	    ResidualH1[dim+1][0][Mc_AN][h_AN][i][j] = r10;
	    ResidualH1[dim+1][1][Mc_AN][h_AN][i][j] = r11;
	    ResidualH1[dim+1][2][Mc_AN][h_AN][i][j] = r12;
	    ResidualH1[dim+1][3][Mc_AN][h_AN][i][j] = r13;

	    ResidualH2[dim+1][0][Mc_AN][h_AN][i][j] = r20;
	    ResidualH2[dim+1][1][Mc_AN][h_AN][i][j] = r21;
	    ResidualH2[dim+1][2][Mc_AN][h_AN][i][j] = r22;

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      r = 0.0; 
	      for (m=0; m<dim; m++){
		r += ResidualH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
	      }

              /* optimum Residual H is stored in ResidualH1[dim+1] */

              ResidualH1[dim+1][spin][Mc_AN][h_AN][i][j] = r;

	    }
	  }
	}
      }
    }
  }

  /******************************************************
   calculations of inner products of <s0|y0> and <y0|y0>
   in order to estimate the parameter "al".
  ******************************************************/

  my_sy = 0.0;
  my_yy = 0.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    tmp1 = HisH1[0][0][Mc_AN][h_AN][i][j] - HisH1[1][0][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][0][Mc_AN][h_AN][i][j] - ResidualH1[1][0][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][1][Mc_AN][h_AN][i][j] - HisH1[1][1][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][1][Mc_AN][h_AN][i][j] - ResidualH1[1][1][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][2][Mc_AN][h_AN][i][j] - HisH1[1][2][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][2][Mc_AN][h_AN][i][j] - ResidualH1[1][2][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][3][Mc_AN][h_AN][i][j] - HisH1[1][3][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][3][Mc_AN][h_AN][i][j] - ResidualH1[1][3][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][0][Mc_AN][h_AN][i][j] - HisH2[1][0][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][0][Mc_AN][h_AN][i][j] - ResidualH2[1][0][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][1][Mc_AN][h_AN][i][j] - HisH2[1][1][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][1][Mc_AN][h_AN][i][j] - ResidualH2[1][1][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][2][Mc_AN][h_AN][i][j] - HisH2[1][2][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][2][Mc_AN][h_AN][i][j] - ResidualH2[1][2][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 
	  }
	}
      }
    }

  } /* if (SpinP_switch==3 */

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      tmp1 = HisH1[0][spin][Mc_AN][h_AN][i][j] - HisH1[1][spin][Mc_AN][h_AN][i][j];           /* s */
	      tmp2 = ResidualH1[0][spin][Mc_AN][h_AN][i][j] - ResidualH1[1][spin][Mc_AN][h_AN][i][j]; /* y */
	      my_sy += tmp1*tmp2; 
	      my_yy += tmp2*tmp2; 
	    }
	  }
	}
      }
    }

  } /* else */

  MPI_Allreduce(&my_sy, &sy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&my_yy, &yy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* al < sy/yy */

  al = sy/yy - 0.2;

  /****************************************************
         calculations of inner products of <r|y> 
  ****************************************************/

  for (m=0; m<dim; m++){
    for (n=0; n<dim; n++){

      my_sum = 0.0;

      if (SpinP_switch==3){

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

                /* m */
		s = HisH1[m][0][Mc_AN][h_AN][i][j] - HisH1[m+1][0][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH1[m][0][Mc_AN][h_AN][i][j] - ResidualH1[m+1][0][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH1[n][0][Mc_AN][h_AN][i][j] - ResidualH1[n+1][0][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH1[m][1][Mc_AN][h_AN][i][j] - HisH1[m+1][1][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH1[m][1][Mc_AN][h_AN][i][j] - ResidualH1[m+1][1][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH1[n][1][Mc_AN][h_AN][i][j] - ResidualH1[n+1][1][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH1[m][2][Mc_AN][h_AN][i][j] - HisH1[m+1][2][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH1[m][2][Mc_AN][h_AN][i][j] - ResidualH1[m+1][2][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH1[n][2][Mc_AN][h_AN][i][j] - ResidualH1[n+1][2][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH1[m][3][Mc_AN][h_AN][i][j] - HisH1[m+1][3][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH1[m][3][Mc_AN][h_AN][i][j] - ResidualH1[m+1][3][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH1[n][3][Mc_AN][h_AN][i][j] - ResidualH1[n+1][3][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH2[m][0][Mc_AN][h_AN][i][j] - HisH2[m+1][0][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH2[m][0][Mc_AN][h_AN][i][j] - ResidualH2[m+1][0][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH2[n][0][Mc_AN][h_AN][i][j] - ResidualH2[n+1][0][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH2[m][1][Mc_AN][h_AN][i][j] - HisH2[m+1][1][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH2[m][1][Mc_AN][h_AN][i][j] - ResidualH2[m+1][1][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH2[n][1][Mc_AN][h_AN][i][j] - ResidualH2[n+1][1][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH2[m][2][Mc_AN][h_AN][i][j] - HisH2[m+1][2][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH2[m][2][Mc_AN][h_AN][i][j] - ResidualH2[m+1][2][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH2[n][2][Mc_AN][h_AN][i][j] - ResidualH2[n+1][2][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

	      }
	    }
	  }
	}

      } /* if (SpinP_switch==3 */

      else{

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	    Gc_AN = M2G[Mc_AN];    
	    Cwan = WhatSpecies[Gc_AN];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){

		  /* m */
		  s = HisH1[m][spin][Mc_AN][h_AN][i][j] - HisH1[m+1][spin][Mc_AN][h_AN][i][j];           /* s */
		  y = ResidualH1[m][spin][Mc_AN][h_AN][i][j] - ResidualH1[m+1][spin][Mc_AN][h_AN][i][j]; /* y */
		  r = s - al*y;                                                                          /* r */
		  /* n */
		  y = ResidualH1[n][spin][Mc_AN][h_AN][i][j] - ResidualH1[n+1][spin][Mc_AN][h_AN][i][j]; /* y */
		  my_sum += r*y; 

		}
	      }
	    }
	  }
	}

      } /* else */

      MPI_Allreduce(&my_sum, &A[m][n], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    } /* n */
  } /* m */    

  Inverse(dim-1,A,IA);

  /****************************************************
    calculations of inner products of <r|OptResidualH> 
  ****************************************************/

  for (m=0; m<dim; m++){

    my_sum = 0.0;

    if (SpinP_switch==3){

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      s = HisH1[m][0][Mc_AN][h_AN][i][j] - HisH1[m+1][0][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][0][Mc_AN][h_AN][i][j] - ResidualH1[m+1][0][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH1[dim+1][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH1[m][1][Mc_AN][h_AN][i][j] - HisH1[m+1][1][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][1][Mc_AN][h_AN][i][j] - ResidualH1[m+1][1][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH1[dim+1][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH1[m][2][Mc_AN][h_AN][i][j] - HisH1[m+1][2][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][2][Mc_AN][h_AN][i][j] - ResidualH1[m+1][2][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH1[dim+1][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH1[m][3][Mc_AN][h_AN][i][j] - HisH1[m+1][3][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][3][Mc_AN][h_AN][i][j] - ResidualH1[m+1][3][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH1[dim+1][3][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH2[m][0][Mc_AN][h_AN][i][j] - HisH2[m+1][0][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][0][Mc_AN][h_AN][i][j] - ResidualH2[m+1][0][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH2[dim+1][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH2[m][1][Mc_AN][h_AN][i][j] - HisH2[m+1][1][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][1][Mc_AN][h_AN][i][j] - ResidualH2[m+1][1][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH2[dim+1][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH2[m][2][Mc_AN][h_AN][i][j] - HisH2[m+1][2][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][2][Mc_AN][h_AN][i][j] - ResidualH2[m+1][2][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH2[dim+1][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	    }
	  }
	}
      }

    } /* if (SpinP_switch==3 */

    else{

      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

		s = HisH1[m][spin][Mc_AN][h_AN][i][j] - HisH1[m+1][spin][Mc_AN][h_AN][i][j];           
		y = ResidualH1[m][spin][Mc_AN][h_AN][i][j] - ResidualH1[m+1][spin][Mc_AN][h_AN][i][j]; 
		r = s - al*y;                                                                          
		or = ResidualH1[dim+1][spin][Mc_AN][h_AN][i][j];                                   
		my_sum += r*or; 
	      }
	    }
	  }
	}
      }

    } /* else */

    MPI_Allreduce(&my_sum, &ror[m], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  } /* m */    

  /****************************************************
     calculation of \sum_j b_{ij} * <r_j|OptResidualH> 
  ****************************************************/
    
  for (m=0; m<dim; m++){
    sum = 0.0;  
    for (n=0; n<dim; n++){
      sum += IA[m][n]*ror[n];  
    }

    coes2[m] = sum;
  }

  /****************************************************
                 mixing of Hamiltonian
  ****************************************************/

  if (1.0e-1<=NormRD[0])
    alpha = 0.5;
  else if (1.0e-2<=NormRD[0] && NormRD[0]<1.0e-1)
    alpha = 0.6;
  else if (1.0e-3<=NormRD[0] && NormRD[0]<1.0e-2)
    alpha = 0.7;
  else if (1.0e-4<=NormRD[0] && NormRD[0]<1.0e-3)
    alpha = 0.8;
  else
    alpha = 1.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            h10 = 0.0; 
            h11 = 0.0;
            h12 = 0.0;
            h13 = 0.0;

            h20 = 0.0; 
            h21 = 0.0;
            h22 = 0.0;

	    for (m=0; m<dim; m++){

	      h10 += HisH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h11 += HisH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h12 += HisH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      h13 += HisH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      h20 += HisH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h21 += HisH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h22 += HisH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];

	      s = HisH1[m][0][Mc_AN][h_AN][i][j] - HisH1[m+1][0][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][0][Mc_AN][h_AN][i][j] - ResidualH1[m+1][0][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h10 -= r*coes2[m];

	      s = HisH1[m][1][Mc_AN][h_AN][i][j] - HisH1[m+1][1][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][1][Mc_AN][h_AN][i][j] - ResidualH1[m+1][1][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h11 -= r*coes2[m];

	      s = HisH1[m][2][Mc_AN][h_AN][i][j] - HisH1[m+1][2][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][2][Mc_AN][h_AN][i][j] - ResidualH1[m+1][2][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h12 -= r*coes2[m];

	      s = HisH1[m][3][Mc_AN][h_AN][i][j] - HisH1[m+1][3][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][3][Mc_AN][h_AN][i][j] - ResidualH1[m+1][3][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h13 -= r*coes2[m];

	      s = HisH2[m][0][Mc_AN][h_AN][i][j] - HisH2[m+1][0][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][0][Mc_AN][h_AN][i][j] - ResidualH2[m+1][0][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h20 -= r*coes2[m];

	      s = HisH2[m][1][Mc_AN][h_AN][i][j] - HisH2[m+1][1][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][1][Mc_AN][h_AN][i][j] - ResidualH2[m+1][1][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h21 -= r*coes2[m];

	      s = HisH2[m][2][Mc_AN][h_AN][i][j] - HisH2[m+1][2][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][2][Mc_AN][h_AN][i][j] - ResidualH2[m+1][2][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h22 -= r*coes2[m];
	    }

	    H[0][Mc_AN][h_AN][i][j]    = h10 - al*ResidualH1[dim+1][0][Mc_AN][h_AN][i][j];
	    H[1][Mc_AN][h_AN][i][j]    = h11 - al*ResidualH1[dim+1][1][Mc_AN][h_AN][i][j];
	    H[2][Mc_AN][h_AN][i][j]    = h12 - al*ResidualH1[dim+1][2][Mc_AN][h_AN][i][j];
	    H[3][Mc_AN][h_AN][i][j]    = h13 - al*ResidualH1[dim+1][3][Mc_AN][h_AN][i][j];

            iHNL[0][Mc_AN][h_AN][i][j] = h20 - al*ResidualH2[dim+1][0][Mc_AN][h_AN][i][j];
            iHNL[1][Mc_AN][h_AN][i][j] = h21 - al*ResidualH2[dim+1][1][Mc_AN][h_AN][i][j];
            iHNL[2][Mc_AN][h_AN][i][j] = h22 - al*ResidualH2[dim+1][2][Mc_AN][h_AN][i][j];

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      h = 0.0;
	      for (m=0; m<dim; m++){

		h += HisH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
		s = HisH1[m][spin][Mc_AN][h_AN][i][j] - HisH1[m+1][spin][Mc_AN][h_AN][i][j];           
		y = ResidualH1[m][spin][Mc_AN][h_AN][i][j] - ResidualH1[m+1][spin][Mc_AN][h_AN][i][j]; 
		r = s - al*y;                                                                          
		h -= r*coes2[m];
	      }

	      H[spin][Mc_AN][h_AN][i][j] = h - al*ResidualH1[dim+1][spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }
  }

  /****************************************************
                  shifting of Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      HisH1[m][0][Mc_AN][h_AN][i][j] = HisH1[m-1][0][Mc_AN][h_AN][i][j];
	      HisH1[m][1][Mc_AN][h_AN][i][j] = HisH1[m-1][1][Mc_AN][h_AN][i][j];
	      HisH1[m][2][Mc_AN][h_AN][i][j] = HisH1[m-1][2][Mc_AN][h_AN][i][j];
	      HisH1[m][3][Mc_AN][h_AN][i][j] = HisH1[m-1][3][Mc_AN][h_AN][i][j];

	      HisH2[m][0][Mc_AN][h_AN][i][j] = HisH2[m-1][0][Mc_AN][h_AN][i][j];
	      HisH2[m][1][Mc_AN][h_AN][i][j] = HisH2[m-1][1][Mc_AN][h_AN][i][j];
	      HisH2[m][2][Mc_AN][h_AN][i][j] = HisH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    HisH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j];
	    HisH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j];
	    HisH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j];
	    HisH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j];

	    HisH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j];
	    HisH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j];
	    HisH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else {

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		HisH1[m][spin][Mc_AN][h_AN][i][j] = HisH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
	      HisH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /****************************************************
                   freeing of arrays 
  ****************************************************/

  free(coes);
  free(coes2);
  free(ror);

  for (i=0; i<List_YOUSO[39]; i++){
    free(A[i]);
  }
  free(A);

  for (i=0; i<List_YOUSO[39]; i++){
    free(IA[i]);
  }
  free(IA);
}










void Pulay_Mixing_H_with_One_Shot_Hessian(int MD_iter, int SCF_iter, int SCF_iter0 )
{
  int Mc_AN,Gc_AN,Cwan,Hwan,h_AN,Gh_AN,i,j,spin;
  int dim,m,n,flag_nan;
  double my_sum,tmp1,tmp2,alpha;
  double r,r10,r11,r12,r13,r20,r21,r22;
  double h,h10,h11,h12,h13,h20,h21,h22;
  double my_sy,my_yy,sy,yy,norm,s,y,or,al,be;
  double **A,**IA,*coes;
  char nanchar[300];

  /****************************************************
       determination of dimension of the subspace
  ****************************************************/

  if (SCF_iter<=Num_Mixing_pDM) dim = SCF_iter-1;
  else                          dim = Num_Mixing_pDM;

  /****************************************************
                allocation of arrays 
  ****************************************************/

  coes = (double*)malloc(sizeof(double)*List_YOUSO[39]);

  A = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    A[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  IA = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    IA[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  /****************************************************
                 shift the residual H
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      ResidualH1[m][0][Mc_AN][h_AN][i][j] = ResidualH1[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH1[m][1][Mc_AN][h_AN][i][j] = ResidualH1[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH1[m][2][Mc_AN][h_AN][i][j] = ResidualH1[m-1][2][Mc_AN][h_AN][i][j];
	      ResidualH1[m][3][Mc_AN][h_AN][i][j] = ResidualH1[m-1][3][Mc_AN][h_AN][i][j];

	      ResidualH2[m][0][Mc_AN][h_AN][i][j] = ResidualH2[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH2[m][1][Mc_AN][h_AN][i][j] = ResidualH2[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH2[m][2][Mc_AN][h_AN][i][j] = ResidualH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    ResidualH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j] - HisH1[0][0][Mc_AN][h_AN][i][j];
	    ResidualH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j] - HisH1[0][1][Mc_AN][h_AN][i][j];
	    ResidualH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j] - HisH1[0][2][Mc_AN][h_AN][i][j];
	    ResidualH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j] - HisH1[0][3][Mc_AN][h_AN][i][j];

	    ResidualH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j] - HisH2[0][0][Mc_AN][h_AN][i][j];
	    ResidualH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j] - HisH2[0][1][Mc_AN][h_AN][i][j];
	    ResidualH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j] - HisH2[0][2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else{

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		ResidualH1[m][spin][Mc_AN][h_AN][i][j] = ResidualH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
  	      ResidualH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j] - HisH1[0][spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /****************************************************
          calculation of the residual matrix
  ****************************************************/

  for (m=0; m<dim; m++){
    for (n=0; n<dim; n++){

      my_sum = 0.0;

      if (SpinP_switch==3){

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

		tmp1 = ResidualH1[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][0][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][1][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][2][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][3][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][3][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][0][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][1][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][2][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 
	      }
	    }
	  }
	}

      } /* if (SpinP_switch==3 */

      else{

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	    Gc_AN = M2G[Mc_AN];    
	    Cwan = WhatSpecies[Gc_AN];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){
		  tmp1 = ResidualH1[m][spin][Mc_AN][h_AN][i][j];
		  tmp2 = ResidualH1[n][spin][Mc_AN][h_AN][i][j];
                  my_sum += tmp1*tmp2; 
		}
	      }
	    }
	  }
	}

      } /* else */

      MPI_Allreduce(&my_sum, &A[m][n], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
      A[n][m] = A[m][n];

    } /* n */
  } /* m */

  NormRD[0] = A[0][0]/(double)atomnum;

  for (m=1; m<=dim; m++){
    A[m-1][dim] = -1.0;
    A[dim][m-1] = -1.0;
  }
  A[dim][dim] = 0.0;

  Inverse(dim,A,IA);

  for (m=1; m<=dim; m++){
    coes[m] = -IA[m-1][dim];
  }

  /****************************************************
            check "nan", "NaN", "inf" or "Inf"
  ****************************************************/

  flag_nan = 0;
  for (m=1; m<=dim; m++){

    sprintf(nanchar,"%8.4f",coes[m]);
    if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	|| strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

      flag_nan = 1;
    }
  }

  if (flag_nan==1){
    for (m=1; m<=dim; m++){
      coes[m] = 0.0;
    }
    coes[1] = 0.05;
    coes[2] = 0.95;
  }

  /****************************************************
      calculation of optimum residual Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            r10 = 0.0; 
            r11 = 0.0;
            r12 = 0.0;
            r13 = 0.0;

            r20 = 0.0; 
            r21 = 0.0;
            r22 = 0.0;

	    for (m=0; m<dim; m++){

	      r10 += ResidualH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r11 += ResidualH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r12 += ResidualH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      r13 += ResidualH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      r20 += ResidualH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r21 += ResidualH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r22 += ResidualH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	    }

            /* optimum Residual H is stored in ResidualH1[dim] and ResidualH2[dim] */

	    ResidualH1[dim][0][Mc_AN][h_AN][i][j] = r10;
	    ResidualH1[dim][1][Mc_AN][h_AN][i][j] = r11;
	    ResidualH1[dim][2][Mc_AN][h_AN][i][j] = r12;
	    ResidualH1[dim][3][Mc_AN][h_AN][i][j] = r13;

	    ResidualH2[dim][0][Mc_AN][h_AN][i][j] = r20;
	    ResidualH2[dim][1][Mc_AN][h_AN][i][j] = r21;
	    ResidualH2[dim][2][Mc_AN][h_AN][i][j] = r22;

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      r = 0.0; 
	      for (m=0; m<dim; m++){
		r += ResidualH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
	      }

              /* optimum Residual H is stored in ResidualH1[dim] */

              ResidualH1[dim][spin][Mc_AN][h_AN][i][j] = r;

	    }
	  }
	}
      }
    }
  }

  /****************************************************
           innner products of <s|y> and <y|y>
  ****************************************************/

  my_sy = 0.0;
  my_yy = 0.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    tmp1 = HisH1[0][0][Mc_AN][h_AN][i][j] - HisH1[1][0][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][0][Mc_AN][h_AN][i][j] - ResidualH1[1][0][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][1][Mc_AN][h_AN][i][j] - HisH1[1][1][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][1][Mc_AN][h_AN][i][j] - ResidualH1[1][1][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][2][Mc_AN][h_AN][i][j] - HisH1[1][2][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][2][Mc_AN][h_AN][i][j] - ResidualH1[1][2][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][3][Mc_AN][h_AN][i][j] - HisH1[1][3][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][3][Mc_AN][h_AN][i][j] - ResidualH1[1][3][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][0][Mc_AN][h_AN][i][j] - HisH2[1][0][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][0][Mc_AN][h_AN][i][j] - ResidualH2[1][0][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][1][Mc_AN][h_AN][i][j] - HisH2[1][1][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][1][Mc_AN][h_AN][i][j] - ResidualH2[1][1][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][2][Mc_AN][h_AN][i][j] - HisH2[1][2][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][2][Mc_AN][h_AN][i][j] - ResidualH2[1][2][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 
	  }
	}
      }
    }

  } /* if (SpinP_switch==3 */

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      tmp1 = HisH1[0][spin][Mc_AN][h_AN][i][j] - HisH1[1][spin][Mc_AN][h_AN][i][j];           /* s */
	      tmp2 = ResidualH1[0][spin][Mc_AN][h_AN][i][j] - ResidualH1[1][spin][Mc_AN][h_AN][i][j]; /* y */
	      my_sy += tmp1*tmp2; 
	      my_yy += tmp2*tmp2; 
	    }
	  }
	}
      }
    }

  } /* else */

  MPI_Allreduce(&my_sy, &sy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&my_yy, &yy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* al < sy/yy */

  al = sy/yy - 0.2;

  /* be = 1/(<s|y>-al*<y|y>) */

  be = 1.0/(sy-al*yy);

  /****************************************************
      inner product of (<s|-al<y|)|OptResidualH>
  ****************************************************/
 
  my_sum = 0.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    s = HisH1[0][0][Mc_AN][h_AN][i][j] - HisH1[1][0][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][0][Mc_AN][h_AN][i][j] - ResidualH1[1][0][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH1[0][1][Mc_AN][h_AN][i][j] - HisH1[1][1][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][1][Mc_AN][h_AN][i][j] - ResidualH1[1][1][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH1[0][2][Mc_AN][h_AN][i][j] - HisH1[1][2][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][2][Mc_AN][h_AN][i][j] - ResidualH1[1][2][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH1[0][3][Mc_AN][h_AN][i][j] - HisH1[1][3][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][3][Mc_AN][h_AN][i][j] - ResidualH1[1][3][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][3][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH2[0][0][Mc_AN][h_AN][i][j] - HisH2[1][0][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][0][Mc_AN][h_AN][i][j] - ResidualH2[1][0][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH2[0][1][Mc_AN][h_AN][i][j] - HisH2[1][1][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][1][Mc_AN][h_AN][i][j] - ResidualH2[1][1][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH2[0][2][Mc_AN][h_AN][i][j] - HisH2[1][2][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][2][Mc_AN][h_AN][i][j] - ResidualH2[1][2][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;
	  }
	}
      }
    }

  } /* if (SpinP_switch==3 */

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
	      s = HisH1[0][spin][Mc_AN][h_AN][i][j] - HisH1[1][spin][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[0][spin][Mc_AN][h_AN][i][j] - ResidualH1[1][spin][Mc_AN][h_AN][i][j]; /* y */
	      or = ResidualH1[dim][spin][Mc_AN][h_AN][i][j];                                       /* OptResidualH */
	      my_sum += (s-al*y)*or;
	    }
	  }
	}
      }
    }

  } /* else */

  MPI_Allreduce(&my_sum, &norm, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  be = norm*be;

  /****************************************************
                 mixing of Hamiltonian
  ****************************************************/

  if (1.0e-1<=NormRD[0])
    alpha = 0.5;
  else if (1.0e-2<=NormRD[0] && NormRD[0]<1.0e-1)
    alpha = 0.6;
  else if (1.0e-3<=NormRD[0] && NormRD[0]<1.0e-2)
    alpha = 0.7;
  else if (1.0e-4<=NormRD[0] && NormRD[0]<1.0e-3)
    alpha = 0.8;
  else
    alpha = 1.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            h10 = 0.0; 
            h11 = 0.0;
            h12 = 0.0;
            h13 = 0.0;

            h20 = 0.0; 
            h21 = 0.0;
            h22 = 0.0;

	    for (m=0; m<dim; m++){

	      h10 += HisH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h11 += HisH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h12 += HisH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      h13 += HisH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      h20 += HisH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h21 += HisH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h22 += HisH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	    }

	    s = HisH1[0][0][Mc_AN][h_AN][i][j] - HisH1[1][0][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][0][Mc_AN][h_AN][i][j] - ResidualH1[1][0][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    H[0][Mc_AN][h_AN][i][j] = h10 - alpha*(al*or + (s-al*y)*be);

	    s = HisH1[0][1][Mc_AN][h_AN][i][j] - HisH1[1][1][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][1][Mc_AN][h_AN][i][j] - ResidualH1[1][1][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    H[1][Mc_AN][h_AN][i][j] = h11 - alpha*(al*or + (s-al*y)*be);

	    s = HisH1[0][2][Mc_AN][h_AN][i][j] - HisH1[1][2][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][2][Mc_AN][h_AN][i][j] - ResidualH1[1][2][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    H[2][Mc_AN][h_AN][i][j] = h12 - alpha*(al*or + (s-al*y)*be);

	    s = HisH1[0][3][Mc_AN][h_AN][i][j] - HisH1[1][3][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][3][Mc_AN][h_AN][i][j] - ResidualH1[1][3][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][3][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    H[3][Mc_AN][h_AN][i][j] = h13 - alpha*(al*or + (s-al*y)*be);

	    s = HisH2[0][0][Mc_AN][h_AN][i][j] - HisH2[1][0][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][0][Mc_AN][h_AN][i][j] - ResidualH2[1][0][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    iHNL[0][Mc_AN][h_AN][i][j] = h20 - alpha*(al*or + (s-al*y)*be);

	    s = HisH2[0][1][Mc_AN][h_AN][i][j] - HisH2[1][1][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][1][Mc_AN][h_AN][i][j] - ResidualH2[1][1][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    iHNL[1][Mc_AN][h_AN][i][j] = h21 - alpha*(al*or + (s-al*y)*be);

	    s = HisH2[0][2][Mc_AN][h_AN][i][j] - HisH2[1][2][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][2][Mc_AN][h_AN][i][j] - ResidualH2[1][2][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    iHNL[2][Mc_AN][h_AN][i][j] = h22 - alpha*(al*or + (s-al*y)*be);
	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      h = 0.0;
	      for (m=0; m<dim; m++){
		h += HisH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
	      }

	      s = HisH1[0][spin][Mc_AN][h_AN][i][j] - HisH1[1][spin][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[0][spin][Mc_AN][h_AN][i][j] - ResidualH1[1][spin][Mc_AN][h_AN][i][j]; /* y */
              or = ResidualH1[dim][spin][Mc_AN][h_AN][i][j];                                       /* OptResidualH */
	      H[spin][Mc_AN][h_AN][i][j] = h - alpha*(al*or + (s-al*y)*be);

	    }
	  }
	}
      }
    }
  }

  /****************************************************
                  shifting of Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      HisH1[m][0][Mc_AN][h_AN][i][j] = HisH1[m-1][0][Mc_AN][h_AN][i][j];
	      HisH1[m][1][Mc_AN][h_AN][i][j] = HisH1[m-1][1][Mc_AN][h_AN][i][j];
	      HisH1[m][2][Mc_AN][h_AN][i][j] = HisH1[m-1][2][Mc_AN][h_AN][i][j];
	      HisH1[m][3][Mc_AN][h_AN][i][j] = HisH1[m-1][3][Mc_AN][h_AN][i][j];

	      HisH2[m][0][Mc_AN][h_AN][i][j] = HisH2[m-1][0][Mc_AN][h_AN][i][j];
	      HisH2[m][1][Mc_AN][h_AN][i][j] = HisH2[m-1][1][Mc_AN][h_AN][i][j];
	      HisH2[m][2][Mc_AN][h_AN][i][j] = HisH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    HisH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j];
	    HisH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j];
	    HisH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j];
	    HisH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j];

	    HisH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j];
	    HisH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j];
	    HisH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else {

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		HisH1[m][spin][Mc_AN][h_AN][i][j] = HisH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
	      HisH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /****************************************************
                   freeing of arrays 
  ****************************************************/

  free(coes);

  for (i=0; i<List_YOUSO[39]; i++){
    free(A[i]);
  }
  free(A);

  for (i=0; i<List_YOUSO[39]; i++){
    free(IA[i]);
  }
  free(IA);
}





/*******************************************************************************
  GPU acceleration of Pulay_Mixing_H (RMM-DIISH).

  The CPU implementation sweeps the whole sparse Hamiltonian structure
  dim*dim times per SCF iteration for the residual Gram matrix (with one
  MPI_Allreduce per matrix element) and another ~4*dim times for the
  shift/mix/save loops.  With scf.Mixing.History around 50 this dominates
  the SCF step.  The GPU path keeps the residual history device-resident:

    - ONE cudaMalloc arena per rank holds a ring of Num_Mixing_pDM flat
      residual slots plus the metric-weight vector, the optimum-residual
      vector, and the Gram pair buffer.  The per-iteration "shift" is a
      rotation of a host-side logical-to-physical slot map (zero copies).
    - The dim*(dim+1)/2 weighted inner products run in one OpenACC kernel;
      the whole triangle is reduced with a single MPI_Allreduce.
    - The optimum residual (sum_m coes[m+1]*R[m]) is one device kernel;
      only that L-vector is downloaded per iteration.
    - HisH1/HisH2 stay host-resident (they are touched O(dim*L) once per
      iteration); their shift becomes a rotation of the m-level pointers,
      which yields bitwise-identical logical contents.
    - The host jagged ResidualH1/ResidualH2 arrays are ALSO kept logically
      current (pointer rotation + fresh slot 0 each iteration), so the
      E_Temp-controller SCF rewind (DFT.c sets SCF_iter_shift mid-cycle,
      Simple_Mixing_H briefly takes over, then Pulay re-engages with a
      large dim) can always re-seed the device ring from the host arrays.

  The preflight is MPI-collective over mpi_comm_level1 and accounts for the
  SUM of the arena sizes of all ranks sharing one device.  On a negative
  verdict the original CPU code below runs unchanged.  Env knobs:
  OPENMX_MIXH_GPU (=0 never / =1 force), OPENMX_MIXH_GPU_RESERVE_MB
  (default max(total/10, 1536 MiB)), OPENMX_MIXH_GPU_VERBOSE.
*******************************************************************************/

#include <openacc.h>
#include <stdint.h>

typedef struct {
  int      verdict;          /* -1 unknown, 0 CPU fallback, 1 GPU; sticky per fingerprint */
  int      valid;            /* device ring mirrors the host residual history */
  int      announced;
  int      fp_md_iter;       /* fingerprint of the sticky verdict/layout */
  int      fp_matomnum;
  int      fp_spinp;
  int      fp_nslot;
  size_t   fp_L;
  int      last_scf_iter;
  int      nslot;            /* ring slots == Num_Mixing_pDM */
  size_t   L;                /* flat doubles per history slot on this rank */
  size_t   L_plane;          /* collinear: doubles per spin plane; NC: == L */
  unsigned char *base;       /* single cudaMalloc arena */
  size_t   total_bytes;
  size_t   o_ring, o_w, o_r, o_pair;
  int     *perm;             /* logical slot -> physical ring slot */
  size_t  *atom_off;         /* [Matomnum+1] flat offset of each atom block within a plane */
  double  *h_flat;           /* host staging buffer (L doubles) */
  double  *h_r;              /* downloaded optimum residual (L doubles) */
} MixH_GpuState;

static MixH_GpuState MixH_gpu = {
  -1, 0, 0, -1, -1, -1, -1, 0U, -1, 0, 0U, 0U,
  NULL, 0U, 0U, 0U, 0U, 0U, NULL, NULL, NULL, NULL
};

static void MixH_AbortWithMessage(const char *message)
{
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  MPI_Abort(mpi_comm_level1, 1);
}

/* Bounded device allocation (same policy as the cluster/band GPU paths):
   absorb short allocation races with a few retries, report persistent
   failure to the caller instead of retrying forever. */
static cudaError_t MixH_TryDeviceMalloc(void **ptr, size_t bytes)
{
  cudaError_t err = cudaErrorMemoryAllocation;

  for (int attempt = 0; attempt < 8; attempt++) {
    err = cudaMalloc(ptr, bytes);
    if (err == cudaSuccess) return cudaSuccess;
    (void)cudaGetLastError();

    double wait_time = drand48() * WAITTIME;
    double start_time = MPI_Wtime();
    double current_time = start_time;
    while ((current_time - start_time) < wait_time) {
      current_time = MPI_Wtime();
    }
  }

  *ptr = NULL;
  return err;
}

static size_t MixH_Align512(size_t bytes)
{
  return (bytes + (size_t)511) & ~((size_t)511);
}

/* Release everything and force a fresh preflight on the next Pulay call.
   Called before the force/energy phase of every SCF cycle (DFT.c) so the
   history ring never competes with the force-path device transients. */
static void MixH_IncFree(void);
static void MixH_IncBuild(int nslot);

void Mixing_H_Release_GPU(void)
{
  MixH_IncFree();
  if (MixH_gpu.base != NULL) cudaFree(MixH_gpu.base);
  free(MixH_gpu.perm);
  free(MixH_gpu.atom_off);
  free(MixH_gpu.h_flat);
  free(MixH_gpu.h_r);
  MixH_gpu.base = NULL;
  MixH_gpu.perm = NULL;
  MixH_gpu.atom_off = NULL;
  MixH_gpu.h_flat = NULL;
  MixH_gpu.h_r = NULL;
  MixH_gpu.total_bytes = 0U;
  MixH_gpu.L = 0U;
  MixH_gpu.L_plane = 0U;
  MixH_gpu.nslot = 0;
  MixH_gpu.valid = 0;
  MixH_gpu.verdict = -1;
  MixH_gpu.announced = 0;
  MixH_gpu.fp_md_iter = -1;
  MixH_gpu.fp_matomnum = -1;
  MixH_gpu.fp_spinp = -1;
  MixH_gpu.fp_nslot = -1;
  MixH_gpu.fp_L = 0U;
  MixH_gpu.last_scf_iter = -1;
}

/* Flat layout of one history slot: collinear packs the planes
   spin-major with the CPU loop order (Mc_AN, h_AN, i, j) inside each
   plane; the noncollinear case packs the seven components (four of
   H/HisH1/ResidualH1, three of iHNL/HisH2/ResidualH2) consecutively per
   (Mc_AN,h_AN,i,j) element. */
static void MixH_ComputeLayout(void)
{
  int Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan;
  int ncomp = (SpinP_switch==3) ? 7 : 1;
  size_t acc = 0U;

  free(MixH_gpu.atom_off);
  MixH_gpu.atom_off = (size_t*)malloc(sizeof(size_t)*(size_t)(Matomnum+1));
  if (MixH_gpu.atom_off == NULL) {
    MixH_AbortWithMessage("Mixing_H.c: failed to allocate the GPU atom offset table.");
  }
  MixH_gpu.atom_off[0] = 0U;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    size_t blk = 0U;
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      blk += (size_t)Spe_Total_NO[Cwan]*(size_t)Spe_Total_NO[Hwan];
    }
    MixH_gpu.atom_off[Mc_AN] = acc;
    acc += blk*(size_t)ncomp;
  }

  MixH_gpu.L_plane = acc;
  MixH_gpu.L = (SpinP_switch==3) ? acc : acc*(size_t)(SpinP_switch+1);
}

/* rotate the top-level (history) pointers of a six-star array over
   m = 0..hi; logically identical to the CPU content shift
   arr[m] = arr[m-1] (m = hi..1) followed by an overwrite of arr[0]. */
static void MixH_RotatePtr6(double ******arr, int hi)
{
  double *****tmp = arr[hi];
  int m;
  for (m=hi; 0<m; m--) arr[m] = arr[m-1];
  arr[0] = tmp;
}

/* Compute the fresh residual into the (already rotated) host slot 0 and,
   when dst is non-NULL, the packed flat image of that slot. */
static void MixH_Residual0Fused(double *dst)
{
  int Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,spin;

  if (SpinP_switch==3){

#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j) schedule(static)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      size_t k = MixH_gpu.atom_off[Mc_AN];
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          for (j=0; j<Spe_Total_NO[Hwan]; j++){

            double r0 = H[0][Mc_AN][h_AN][i][j]    - HisH1[0][0][Mc_AN][h_AN][i][j];
            double r1 = H[1][Mc_AN][h_AN][i][j]    - HisH1[0][1][Mc_AN][h_AN][i][j];
            double r2 = H[2][Mc_AN][h_AN][i][j]    - HisH1[0][2][Mc_AN][h_AN][i][j];
            double r3 = H[3][Mc_AN][h_AN][i][j]    - HisH1[0][3][Mc_AN][h_AN][i][j];
            double s0 = iHNL[0][Mc_AN][h_AN][i][j] - HisH2[0][0][Mc_AN][h_AN][i][j];
            double s1 = iHNL[1][Mc_AN][h_AN][i][j] - HisH2[0][1][Mc_AN][h_AN][i][j];
            double s2 = iHNL[2][Mc_AN][h_AN][i][j] - HisH2[0][2][Mc_AN][h_AN][i][j];

            ResidualH1[0][0][Mc_AN][h_AN][i][j] = r0;
            ResidualH1[0][1][Mc_AN][h_AN][i][j] = r1;
            ResidualH1[0][2][Mc_AN][h_AN][i][j] = r2;
            ResidualH1[0][3][Mc_AN][h_AN][i][j] = r3;
            ResidualH2[0][0][Mc_AN][h_AN][i][j] = s0;
            ResidualH2[0][1][Mc_AN][h_AN][i][j] = s1;
            ResidualH2[0][2][Mc_AN][h_AN][i][j] = s2;

            if (dst != NULL){
              dst[k  ] = r0;
              dst[k+1] = r1;
              dst[k+2] = r2;
              dst[k+3] = r3;
              dst[k+4] = s0;
              dst[k+5] = s1;
              dst[k+6] = s2;
            }
            k += 7;
          }
        }
      }
    }

  }
  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      size_t plane = (size_t)spin*MixH_gpu.L_plane;
#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j) schedule(static)
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        size_t k = plane + MixH_gpu.atom_off[Mc_AN];
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          for (i=0; i<Spe_Total_NO[Cwan]; i++){
            for (j=0; j<Spe_Total_NO[Hwan]; j++){
              double r = H[spin][Mc_AN][h_AN][i][j] - HisH1[0][spin][Mc_AN][h_AN][i][j];
              ResidualH1[0][spin][Mc_AN][h_AN][i][j] = r;
              if (dst != NULL) dst[k] = r;
              k++;
            }
          }
        }
      }
    }
  }
}

/* pack the host jagged residual slot m into a flat buffer (seeding) */
static void MixH_PackResidualSlot(int m, double *dst)
{
  int Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,spin;

  if (SpinP_switch==3){

#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j) schedule(static)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      size_t k = MixH_gpu.atom_off[Mc_AN];
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          for (j=0; j<Spe_Total_NO[Hwan]; j++){
            dst[k  ] = ResidualH1[m][0][Mc_AN][h_AN][i][j];
            dst[k+1] = ResidualH1[m][1][Mc_AN][h_AN][i][j];
            dst[k+2] = ResidualH1[m][2][Mc_AN][h_AN][i][j];
            dst[k+3] = ResidualH1[m][3][Mc_AN][h_AN][i][j];
            dst[k+4] = ResidualH2[m][0][Mc_AN][h_AN][i][j];
            dst[k+5] = ResidualH2[m][1][Mc_AN][h_AN][i][j];
            dst[k+6] = ResidualH2[m][2][Mc_AN][h_AN][i][j];
            k += 7;
          }
        }
      }
    }

  }
  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      size_t plane = (size_t)spin*MixH_gpu.L_plane;
#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j) schedule(static)
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        size_t k = plane + MixH_gpu.atom_off[Mc_AN];
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          for (i=0; i<Spe_Total_NO[Cwan]; i++){
            for (j=0; j<Spe_Total_NO[Hwan]; j++){
              dst[k] = ResidualH1[m][spin][Mc_AN][h_AN][i][j];
              k++;
            }
          }
        }
      }
    }
  }
}

/* pack the metric weights (metric[Mc_AN][i], identical for every spin
   plane / component) into the flat layout */
static void MixH_PackMetricW(double **metric, double *dst)
{
  int Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,spin;
  int ncomp = (SpinP_switch==3) ? 7 : 1;
  int nsp = (SpinP_switch==3) ? 1 : SpinP_switch+1;

  for (spin=0; spin<nsp; spin++){
    size_t plane = (size_t)spin*MixH_gpu.L_plane;
#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j) schedule(static)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      size_t k = plane + MixH_gpu.atom_off[Mc_AN];
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          double w = metric[Mc_AN][i];
          for (j=0; j<Spe_Total_NO[Hwan]; j++){
            int c;
            for (c=0; c<ncomp; c++) dst[k+c] = w;
            k += ncomp;
          }
        }
      }
    }
  }
}

/* H = sum_m coes[m+1]*HisH1[m] + alpha*r  (and iHNL analogously for NC) */
static void MixH_MixFromHistory(int dim, double alpha, const double *coes,
                                const double *r_flat)
{
  int Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,spin,m;

  if (SpinP_switch==3){

#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,m) schedule(static)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      size_t k = MixH_gpu.atom_off[Mc_AN];
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          for (j=0; j<Spe_Total_NO[Hwan]; j++){

            double h0=0.0,h1=0.0,h2=0.0,h3=0.0,g0=0.0,g1=0.0,g2=0.0;
            for (m=0; m<dim; m++){
              double c = coes[m+1];
              h0 += HisH1[m][0][Mc_AN][h_AN][i][j]*c;
              h1 += HisH1[m][1][Mc_AN][h_AN][i][j]*c;
              h2 += HisH1[m][2][Mc_AN][h_AN][i][j]*c;
              h3 += HisH1[m][3][Mc_AN][h_AN][i][j]*c;
              g0 += HisH2[m][0][Mc_AN][h_AN][i][j]*c;
              g1 += HisH2[m][1][Mc_AN][h_AN][i][j]*c;
              g2 += HisH2[m][2][Mc_AN][h_AN][i][j]*c;
            }

            H[0][Mc_AN][h_AN][i][j] = h0 + alpha*r_flat[k  ];
            H[1][Mc_AN][h_AN][i][j] = h1 + alpha*r_flat[k+1];
            H[2][Mc_AN][h_AN][i][j] = h2 + alpha*r_flat[k+2];
            H[3][Mc_AN][h_AN][i][j] = h3 + alpha*r_flat[k+3];
            iHNL[0][Mc_AN][h_AN][i][j] = g0 + alpha*r_flat[k+4];
            iHNL[1][Mc_AN][h_AN][i][j] = g1 + alpha*r_flat[k+5];
            iHNL[2][Mc_AN][h_AN][i][j] = g2 + alpha*r_flat[k+6];
            k += 7;
          }
        }
      }
    }

  }
  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      size_t plane = (size_t)spin*MixH_gpu.L_plane;
#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,m) schedule(static)
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        size_t k = plane + MixH_gpu.atom_off[Mc_AN];
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          for (i=0; i<Spe_Total_NO[Cwan]; i++){
            for (j=0; j<Spe_Total_NO[Hwan]; j++){
              double h = 0.0;
              for (m=0; m<dim; m++){
                h += HisH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
              }
              H[spin][Mc_AN][h_AN][i][j] = h + alpha*r_flat[k];
              k++;
            }
          }
        }
      }
    }
  }
}

/* save the mixed Hamiltonian into the (already rotated) history slot 0 */
static void MixH_SaveHistory0(void)
{
  int Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,spin;

  if (SpinP_switch==3){

#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j) schedule(static)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          for (j=0; j<Spe_Total_NO[Hwan]; j++){
            HisH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j];
            HisH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j];
            HisH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j];
            HisH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j];
            HisH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j];
            HisH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j];
            HisH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j];
          }
        }
      }
    }

  }
  else{

    for (spin=0; spin<=SpinP_switch; spin++){
#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j) schedule(static)
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          for (i=0; i<Spe_Total_NO[Cwan]; i++){
            for (j=0; j<Spe_Total_NO[Hwan]; j++){
              HisH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j];
            }
          }
        }
      }
    }
  }
}

/* Collective preflight: rebuild the layout and the device arena when the
   fingerprint changed, account for every rank sharing this device, and
   agree on one verdict over mpi_comm_level1. */
static int MixH_GpuPreflight(int MD_iter, int myid)
{
  const char *force_env = getenv("OPENMX_MIXH_GPU");
  int forced_on = (force_env != NULL && atoi(force_env) == 1);
  int forced_off = (force_env != NULL && atoi(force_env) == 0);
  int forced_inc = (force_env != NULL && atoi(force_env) == 2);
  int nslot = Num_Mixing_pDM;
  int fit, my_fit;
  size_t free_b = 0U, total_b = 0U;

  if (MixH_gpu.verdict != -1 &&
      MixH_gpu.fp_md_iter == MD_iter &&
      MixH_gpu.fp_matomnum == Matomnum &&
      MixH_gpu.fp_spinp == SpinP_switch &&
      MixH_gpu.fp_nslot == nslot){
    return MixH_gpu.verdict;
  }

  Mixing_H_Release_GPU();
  MixH_ComputeLayout();
  MixH_gpu.nslot = nslot;
  MixH_gpu.fp_md_iter = MD_iter;
  MixH_gpu.fp_matomnum = Matomnum;
  MixH_gpu.fp_spinp = SpinP_switch;
  MixH_gpu.fp_nslot = nslot;
  MixH_gpu.fp_L = MixH_gpu.L;

  if (forced_off){
    MixH_gpu.verdict = 0;
    return 0;
  }

  if (forced_inc){
    MixH_IncBuild(nslot);
    if (getenv("OPENMX_MIXH_GPU_VERBOSE") != NULL && myid == Host_ID){
      printf("<Mixing_H> incremental CPU RMM-DIISH mixing forced by OPENMX_MIXH_GPU=2.\n");
      fflush(stdout);
    }
    MixH_gpu.verdict = 2;
    return 2;
  }

  {
    size_t Lb = MixH_gpu.L*sizeof(double);
    size_t npair_cap = (size_t)nslot*((size_t)nslot+1)/2;
    MixH_gpu.o_ring = 0U;
    MixH_gpu.o_w    = MixH_Align512((size_t)nslot*Lb);
    MixH_gpu.o_r    = MixH_Align512(MixH_gpu.o_w + Lb);
    MixH_gpu.o_pair = MixH_Align512(MixH_gpu.o_r + Lb);
    MixH_gpu.total_bytes = MixH_Align512(MixH_gpu.o_pair + npair_cap*sizeof(double));
  }

  /* Quiesce and measure, then account per shared device.  Deliberately NO
     acc_clear_freelists() here: this preflight runs mid-cycle (first Pulay
     iteration) and draining the OpenACC pool at that point changes the
     allocation dynamics of the other resident GPU modules on a nearly full
     device.  Pool blocks therefore show up as "used", which only makes the
     estimate conservative. */
  acc_wait_all();
  cudaDeviceSynchronize();
  MPI_Barrier(mpi_comm_level1);
  cudaMemGetInfo(&free_b, &total_b);

  {
    MPI_Comm node_comm = MPI_COMM_NULL, device_comm = MPI_COMM_NULL;
    int cuda_device = -1;
    unsigned long long my_bytes = (unsigned long long)MixH_gpu.total_bytes;
    unsigned long long group_bytes = my_bytes;
    unsigned long long my_free = (unsigned long long)free_b;
    unsigned long long group_free = my_free;
    unsigned long long reserve;

    {
      const char *rsv = getenv("OPENMX_MIXH_GPU_RESERVE_MB");
      if (rsv != NULL && 0 < atoi(rsv)) {
        reserve = (unsigned long long)atoi(rsv)*1024ULL*1024ULL;
      }
      else {
        reserve = (unsigned long long)(total_b/10U);
        if (reserve < 1536ULL*1024ULL*1024ULL) reserve = 1536ULL*1024ULL*1024ULL;
      }
    }

    MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &node_comm);
    if (cudaGetDevice(&cuda_device) != cudaSuccess) cuda_device = -1;
    MPI_Comm_split(node_comm, cuda_device, 0, &device_comm);
    MPI_Comm_free(&node_comm);
    if (device_comm != MPI_COMM_NULL){
      MPI_Allreduce(&my_bytes, &group_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, device_comm);
      MPI_Allreduce(&my_free, &group_free, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, device_comm);
      MPI_Comm_free(&device_comm);
    }

    my_fit = (cuda_device >= 0 &&
              group_bytes + reserve <= group_free);
    if (forced_on) my_fit = (cuda_device >= 0);

    MPI_Allreduce(&my_fit, &fit, 1, MPI_INT, MPI_MIN, mpi_comm_level1);

    if (fit){
      int ok_local, ok;
      cudaError_t err = MixH_TryDeviceMalloc((void**)&MixH_gpu.base, MixH_gpu.total_bytes);
      ok_local = (err == cudaSuccess);
      MPI_Allreduce(&ok_local, &ok, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
      if (!ok){
        if (forced_on){
          char msg[512];
          snprintf(msg, sizeof(msg),
                   "Mixing_H.c: OPENMX_MIXH_GPU=1 was set but the %.1f MiB mixing-history arena could not be allocated on the GPU.",
                   (double)MixH_gpu.total_bytes/(1024.0*1024.0));
          MixH_AbortWithMessage(msg);
        }
        if (MixH_gpu.base != NULL) cudaFree(MixH_gpu.base);
        MixH_gpu.base = NULL;
        fit = 0;
      }
    }

    if (!fit && !MixH_gpu.announced){
      unsigned long long diag_bytes = group_bytes, diag_free = group_free;
      MPI_Allreduce(&group_bytes, &diag_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, mpi_comm_level1);
      MPI_Allreduce(&group_free, &diag_free, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, mpi_comm_level1);
      if (myid == Host_ID){
        printf("<Mixing_H> The RMM-DIISH residual history (%.1f MiB per device) does not fit on the GPU (%.1f MiB free); using the incremental CPU Pulay mixing instead. OPENMX_MIXH_GPU=0 restores the legacy CPU mixing, OPENMX_MIXH_GPU=1 forces the GPU path.\n",
               (double)diag_bytes/(1024.0*1024.0),
               (double)diag_free/(1024.0*1024.0));
        fflush(stdout);
      }
      MixH_gpu.announced = 1;
    }
  }

  if (fit){
    MixH_gpu.perm = (int*)malloc(sizeof(int)*(size_t)nslot);
    MixH_gpu.h_flat = (double*)malloc(sizeof(double)*((MixH_gpu.L==0U)?1U:MixH_gpu.L));
    MixH_gpu.h_r = (double*)malloc(sizeof(double)*((MixH_gpu.L==0U)?1U:MixH_gpu.L));
    if (MixH_gpu.perm == NULL || MixH_gpu.h_flat == NULL || MixH_gpu.h_r == NULL){
      MixH_AbortWithMessage("Mixing_H.c: failed to allocate the GPU staging buffers.");
    }
    if (getenv("OPENMX_MIXH_GPU_VERBOSE") != NULL && myid == Host_ID){
      printf("<Mixing_H> GPU RMM-DIISH mixing engaged (%d slots x %.1f MiB per rank).\n",
             nslot, (double)MixH_gpu.L*sizeof(double)/(1024.0*1024.0));
      fflush(stdout);
    }
    MixH_gpu.verdict = 1;
    return 1;
  }

  MixH_IncBuild(nslot);
  MixH_gpu.verdict = 2;
  return 2;
}

/* weighted Gram triangle of the logical residual slots 0..dim-1 */
static void MixH_GramDevice(int dim, int npair, const int *pm, const int *pn,
                            const int *pslot, double *tri_out)
{
  double *ring = (double*)(MixH_gpu.base + MixH_gpu.o_ring);
  double *w    = (double*)(MixH_gpu.base + MixH_gpu.o_w);
  double *ap   = (double*)(MixH_gpu.base + MixH_gpu.o_pair);
  const long Lk = (long)MixH_gpu.L;
  const size_t Ls = MixH_gpu.L;
  int p;

  if (0 < Lk){
#pragma acc parallel loop gang deviceptr(ring,w,ap) copyin(pm[0:npair],pn[0:npair],pslot[0:dim])
    for (p=0; p<npair; p++){
      const double *Rm = ring + (size_t)pslot[pm[p]]*Ls;
      const double *Rn = ring + (size_t)pslot[pn[p]]*Ls;
      double s = 0.0;
      long k;
#pragma acc loop vector reduction(+:s)
      for (k=0; k<Lk; k++) s += w[k]*Rm[k]*Rn[k];
      ap[p] = s;
    }
    acc_memcpy_from_device(tri_out, ap, sizeof(double)*(size_t)npair);
  }
  else{
    for (p=0; p<npair; p++) tri_out[p] = 0.0;
  }
}

/* optimum residual r = sum_m coes[m+1]*R[m] on the device */
static void MixH_OptResidualDevice(int dim, const int *pslot, const double *cshift)
{
  double *ring = (double*)(MixH_gpu.base + MixH_gpu.o_ring);
  double *r    = (double*)(MixH_gpu.base + MixH_gpu.o_r);
  const long Lk = (long)MixH_gpu.L;
  const size_t Ls = MixH_gpu.L;
  long k;

  if (Lk <= 0) return;

#pragma acc parallel loop gang vector deviceptr(ring,r) copyin(pslot[0:dim],cshift[0:dim])
  for (k=0; k<Lk; k++){
    double s = 0.0;
    int m;
    for (m=0; m<dim; m++) s += cshift[m]*ring[(size_t)pslot[m]*Ls + (size_t)k];
    r[k] = s;
  }
  acc_memcpy_from_device(MixH_gpu.h_r, r, sizeof(double)*(size_t)Lk);
}

/*******************************************************************************
  Tier-B fallback: incremental block-Gram Pulay mixing on the CPU.

  When the device-resident history does not fit (or OPENMX_MIXH_GPU=2), the
  residual Gram matrix is decomposed exactly as

      A[m][n] = sum_blk  w_blk * S_blk[m][n],     blk = (Mc_AN, i)

  where S_blk[m][n] is the metric-independent partial Gram of the (Mc_AN,i)
  row block.  Because the history "shift" is a pointer rotation, the old
  S_blk entries stay bitwise valid across iterations and only the new row
  S_blk[0][n] (n = 0..dim-1) has to be computed: O(dim*L) work instead of
  the legacy O(dim^2*L), while the changing metric is re-applied exactly
  every iteration.  A broken continuity (SCF rewind, new cycle) triggers a
  full O(dim^2*L) re-seed of S.
*******************************************************************************/

typedef struct {
  int      valid;
  int      last_scf_iter;
  int      nslot;
  int      nblk;         /* sum over local atoms of Spe_Total_NO */
  int     *blk_base;     /* [Matomnum+1]: first block id of each atom */
  int     *perm;         /* logical slot -> physical slot */
  double  *S;            /* [nslot][nslot][nblk], physical slot indexing */
  double  *wblk;         /* [nblk] metric weights of this iteration */
} MixH_IncState;

static MixH_IncState MixH_inc = { 0, -1, 0, 0, NULL, NULL, NULL, NULL };

static void MixH_IncFree(void)
{
  free(MixH_inc.blk_base);
  free(MixH_inc.perm);
  free(MixH_inc.S);
  free(MixH_inc.wblk);
  MixH_inc.blk_base = NULL;
  MixH_inc.perm = NULL;
  MixH_inc.S = NULL;
  MixH_inc.wblk = NULL;
  MixH_inc.valid = 0;
  MixH_inc.last_scf_iter = -1;
  MixH_inc.nslot = 0;
  MixH_inc.nblk = 0;
}

static void MixH_IncBuild(int nslot)
{
  int Mc_AN,Gc_AN,Cwan,nblk;

  MixH_IncFree();

  nblk = 0;
  MixH_inc.blk_base = (int*)malloc(sizeof(int)*(size_t)(Matomnum+1));
  if (MixH_inc.blk_base == NULL){
    MixH_AbortWithMessage("Mixing_H.c: failed to allocate the block base table.");
  }
  MixH_inc.blk_base[0] = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    MixH_inc.blk_base[Mc_AN] = nblk;
    nblk += Spe_Total_NO[Cwan];
  }
  MixH_inc.nblk = nblk;
  MixH_inc.nslot = nslot;

  MixH_inc.perm = (int*)malloc(sizeof(int)*(size_t)nslot);
  MixH_inc.S = (double*)malloc(sizeof(double)*
               ((size_t)nslot*(size_t)nslot*(size_t)((nblk==0)?1:nblk)));
  MixH_inc.wblk = (double*)malloc(sizeof(double)*(size_t)((nblk==0)?1:nblk));
  if (MixH_inc.perm == NULL || MixH_inc.S == NULL || MixH_inc.wblk == NULL){
    MixH_AbortWithMessage("Mixing_H.c: failed to allocate the block Gram cache.");
  }
}

/* compute the block Gram row of logical slot mlog against logical slots
   0..nmax and store it (symmetrically) under physical indexing */
static void MixH_IncComputeRow(int mlog, int nmax)
{
  int Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,n,spin;
  const int nslot = MixH_inc.nslot;
  const int nblk = MixH_inc.nblk;
  const int pm = MixH_inc.perm[mlog];

#pragma omp parallel private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,n,spin)
  {
    double *acc = (double*)malloc(sizeof(double)*(size_t)(nmax+1));

#pragma omp for schedule(static)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      int blk0 = MixH_inc.blk_base[Mc_AN];
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      for (i=0; i<Spe_Total_NO[Cwan]; i++){
        int blk = blk0 + i;

        for (n=0; n<=nmax; n++) acc[n] = 0.0;

        if (SpinP_switch==3){

          for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
            int jan;
            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            jan = Spe_Total_NO[Hwan];

            {
              const double *p10 = ResidualH1[mlog][0][Mc_AN][h_AN][i];
              const double *p11 = ResidualH1[mlog][1][Mc_AN][h_AN][i];
              const double *p12 = ResidualH1[mlog][2][Mc_AN][h_AN][i];
              const double *p13 = ResidualH1[mlog][3][Mc_AN][h_AN][i];
              const double *p20 = ResidualH2[mlog][0][Mc_AN][h_AN][i];
              const double *p21 = ResidualH2[mlog][1][Mc_AN][h_AN][i];
              const double *p22 = ResidualH2[mlog][2][Mc_AN][h_AN][i];

              for (n=0; n<=nmax; n++){
                const double *q10 = ResidualH1[n][0][Mc_AN][h_AN][i];
                const double *q11 = ResidualH1[n][1][Mc_AN][h_AN][i];
                const double *q12 = ResidualH1[n][2][Mc_AN][h_AN][i];
                const double *q13 = ResidualH1[n][3][Mc_AN][h_AN][i];
                const double *q20 = ResidualH2[n][0][Mc_AN][h_AN][i];
                const double *q21 = ResidualH2[n][1][Mc_AN][h_AN][i];
                const double *q22 = ResidualH2[n][2][Mc_AN][h_AN][i];
                double s = 0.0;
                for (j=0; j<jan; j++){
                  s += p10[j]*q10[j] + p11[j]*q11[j] + p12[j]*q12[j] + p13[j]*q13[j]
                     + p20[j]*q20[j] + p21[j]*q21[j] + p22[j]*q22[j];
                }
                acc[n] += s;
              }
            }
          }

        }
        else{

          for (spin=0; spin<=SpinP_switch; spin++){
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              int jan;
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              jan = Spe_Total_NO[Hwan];

              {
                const double *p = ResidualH1[mlog][spin][Mc_AN][h_AN][i];
                for (n=0; n<=nmax; n++){
                  const double *q = ResidualH1[n][spin][Mc_AN][h_AN][i];
                  double s = 0.0;
                  for (j=0; j<jan; j++) s += p[j]*q[j];
                  acc[n] += s;
                }
              }
            }
          }
        }

        for (n=0; n<=nmax; n++){
          int pn = MixH_inc.perm[n];
          MixH_inc.S[((size_t)pm*(size_t)nslot + (size_t)pn)*(size_t)nblk + (size_t)blk] = acc[n];
          MixH_inc.S[((size_t)pn*(size_t)nslot + (size_t)pm)*(size_t)nblk + (size_t)blk] = acc[n];
        }
      }
    }

    free(acc);
  }
}

/* fused optimum-residual + Hamiltonian mixing sweep:
   H = sum_m coes[m+1]*HisH1[m] + alpha * sum_m coes[m+1]*ResidualH1[m] */
static void MixH_IncMixSweep(int dim, double alpha, const double *coes)
{
  int Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,m,spin;

  if (SpinP_switch==3){

#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,m) schedule(static)
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          for (j=0; j<Spe_Total_NO[Hwan]; j++){

            double r10=0.0,r11=0.0,r12=0.0,r13=0.0,r20=0.0,r21=0.0,r22=0.0;
            double h10=0.0,h11=0.0,h12=0.0,h13=0.0,h20=0.0,h21=0.0,h22=0.0;

            for (m=0; m<dim; m++){
              double c = coes[m+1];
              r10 += ResidualH1[m][0][Mc_AN][h_AN][i][j]*c;
              r11 += ResidualH1[m][1][Mc_AN][h_AN][i][j]*c;
              r12 += ResidualH1[m][2][Mc_AN][h_AN][i][j]*c;
              r13 += ResidualH1[m][3][Mc_AN][h_AN][i][j]*c;
              r20 += ResidualH2[m][0][Mc_AN][h_AN][i][j]*c;
              r21 += ResidualH2[m][1][Mc_AN][h_AN][i][j]*c;
              r22 += ResidualH2[m][2][Mc_AN][h_AN][i][j]*c;
              h10 += HisH1[m][0][Mc_AN][h_AN][i][j]*c;
              h11 += HisH1[m][1][Mc_AN][h_AN][i][j]*c;
              h12 += HisH1[m][2][Mc_AN][h_AN][i][j]*c;
              h13 += HisH1[m][3][Mc_AN][h_AN][i][j]*c;
              h20 += HisH2[m][0][Mc_AN][h_AN][i][j]*c;
              h21 += HisH2[m][1][Mc_AN][h_AN][i][j]*c;
              h22 += HisH2[m][2][Mc_AN][h_AN][i][j]*c;
            }

            H[0][Mc_AN][h_AN][i][j] = h10 + alpha*r10;
            H[1][Mc_AN][h_AN][i][j] = h11 + alpha*r11;
            H[2][Mc_AN][h_AN][i][j] = h12 + alpha*r12;
            H[3][Mc_AN][h_AN][i][j] = h13 + alpha*r13;
            iHNL[0][Mc_AN][h_AN][i][j] = h20 + alpha*r20;
            iHNL[1][Mc_AN][h_AN][i][j] = h21 + alpha*r21;
            iHNL[2][Mc_AN][h_AN][i][j] = h22 + alpha*r22;
          }
        }
      }
    }

  }
  else{

    for (spin=0; spin<=SpinP_switch; spin++){
#pragma omp parallel for private(Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,m) schedule(static)
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          for (i=0; i<Spe_Total_NO[Cwan]; i++){
            for (j=0; j<Spe_Total_NO[Hwan]; j++){
              double r = 0.0;
              double h = 0.0;
              for (m=0; m<dim; m++){
                double c = coes[m+1];
                r += ResidualH1[m][spin][Mc_AN][h_AN][i][j]*c;
                h += HisH1[m][spin][Mc_AN][h_AN][i][j]*c;
              }
              H[spin][Mc_AN][h_AN][i][j] = h + alpha*r;
            }
          }
        }
      }
    }
  }
}

static void Pulay_Mixing_H_Inc(int MD_iter, int SCF_iter, int SCF_iter0, int dim)
{
  int Mc_AN,Gc_AN,Cwan,i,m,n,flag_nan;
  int seed;
  double alpha,d;
  double **A,**IA,*coes,**metric;
  char nanchar[300];

  /* small host work arrays (same shapes as the legacy path) */

  coes = (double*)malloc(sizeof(double)*List_YOUSO[39]);

  A = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    A[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  IA = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    IA[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  metric = (double**)malloc(sizeof(double*)*(Matomnum+1));
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    int tno;
    if (Mc_AN==0){
      tno = 1;
    }
    else{
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno = Spe_Total_NO[Cwan];
    }
    metric[Mc_AN] = (double*)malloc(sizeof(double)*tno);
  }

  /* metric used for the norm calculations (identical to the legacy path) */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    for (i=0; i<Spe_Total_NO[Cwan]; i++){
      d = fabs(HisH1[0][0][Mc_AN][0][i][i]-ChemP);
      metric[Mc_AN][i] = 5.0/(d*d+5.0);
      MixH_inc.wblk[MixH_inc.blk_base[Mc_AN]+i] = metric[Mc_AN][i];
    }
  }

  /* residual shift (pointer rotation) + fresh slot 0, then the block-Gram
     ring update: only the new row on continuity, everything on a seed */

  seed = !(MixH_inc.valid && MixH_inc.last_scf_iter+1 == SCF_iter);

  MixH_RotatePtr6(ResidualH1, dim);
  if (SpinP_switch==3) MixH_RotatePtr6(ResidualH2, dim);
  MixH_Residual0Fused(NULL);

  if (seed){
    for (m=0; m<MixH_inc.nslot; m++) MixH_inc.perm[m] = m;
    for (m=0; m<dim; m++) MixH_IncComputeRow(m, m);
    MixH_inc.valid = 1;
  }
  else{
    int last = MixH_inc.perm[MixH_inc.nslot-1];
    for (m=MixH_inc.nslot-1; 0<m; m--) MixH_inc.perm[m] = MixH_inc.perm[m-1];
    MixH_inc.perm[0] = last;
    MixH_IncComputeRow(0, dim-1);
  }
  MixH_inc.last_scf_iter = SCF_iter;

  /* assemble the metric-weighted Gram triangle and reduce it once */

  {
    int npair = dim*(dim+1)/2;
    int *pmv = (int*)malloc(sizeof(int)*(size_t)npair);
    int *pnv = (int*)malloc(sizeof(int)*(size_t)npair);
    double *tri_my = (double*)malloc(sizeof(double)*(size_t)npair);
    double *tri = (double*)malloc(sizeof(double)*(size_t)npair);
    int p = 0;

    if (pmv==NULL || pnv==NULL || tri_my==NULL || tri==NULL){
      MixH_AbortWithMessage("Mixing_H.c: failed to allocate the Gram pair tables.");
    }

    for (m=0; m<dim; m++){
      for (n=0; n<=m; n++){
        pmv[p] = m;
        pnv[p] = n;
        p++;
      }
    }

#pragma omp parallel for schedule(static)
    for (p=0; p<npair; p++){
      const int nblk = MixH_inc.nblk;
      const double *Sv = MixH_inc.S
        + ((size_t)MixH_inc.perm[pmv[p]]*(size_t)MixH_inc.nslot
           + (size_t)MixH_inc.perm[pnv[p]])*(size_t)nblk;
      double s = 0.0;
      int blk;
      for (blk=0; blk<nblk; blk++) s += MixH_inc.wblk[blk]*Sv[blk];
      tri_my[p] = s;
    }

    MPI_Allreduce(tri_my, tri, npair, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    for (p=0; p<npair; p++){
      A[pmv[p]][pnv[p]] = tri[p];
      A[pnv[p]][pmv[p]] = tri[p];
    }

    free(tri);
    free(tri_my);
    free(pnv);
    free(pmv);
  }

  NormRD[0] = A[0][0]/(double)atomnum;

  for (m=1; m<=dim; m++){
    A[m-1][dim] = -1.0;
    A[dim][m-1] = -1.0;
  }
  A[dim][dim] = 0.0;

  Inverse(dim,A,IA);

  for (m=1; m<=dim; m++){
    coes[m] = -IA[m-1][dim];
  }

  /* check "nan", "NaN", "inf" or "Inf" (identical to the legacy path) */

  flag_nan = 0;
  for (m=1; m<=dim; m++){

    sprintf(nanchar,"%8.4f",coes[m]);
    if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL
	|| strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

      flag_nan = 1;
    }
  }

  if (flag_nan==1){
    for (m=1; m<=dim; m++){
      coes[m] = 0.0;
    }
    coes[1] = 0.05;
    coes[2] = 0.95;
  }

  if (1.0e-1<=NormRD[0])
    alpha = 0.5;
  else if (1.0e-2<=NormRD[0] && NormRD[0]<1.0e-1)
    alpha = 0.6;
  else if (1.0e-3<=NormRD[0] && NormRD[0]<1.0e-2)
    alpha = 0.7;
  else if (1.0e-4<=NormRD[0] && NormRD[0]<1.0e-3)
    alpha = 0.8;
  else
    alpha = 1.0;

  MixH_IncMixSweep(dim, alpha, coes);

  /* Hamiltonian history shift (pointer rotation) and save of the mixed H */

  MixH_RotatePtr6(HisH1, dim);
  if (SpinP_switch==3) MixH_RotatePtr6(HisH2, dim);
  MixH_SaveHistory0();

  /* freeing of the host work arrays */

  free(coes);

  for (i=0; i<List_YOUSO[39]; i++){
    free(A[i]);
  }
  free(A);

  for (i=0; i<List_YOUSO[39]; i++){
    free(IA[i]);
  }
  free(IA);

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    free(metric[Mc_AN]);
  }
  free(metric);

  (void)MD_iter;
  (void)SCF_iter0;
}

static void Pulay_Mixing_H_GPU(int MD_iter, int SCF_iter, int SCF_iter0, int dim)
{
  int Mc_AN,Gc_AN,Cwan,i,m,n,flag_nan;
  int seed;
  double alpha,d;
  double **A,**IA,*coes,**metric;
  char nanchar[300];

  /* allocation of the small host work arrays (same shapes as the CPU path) */

  coes = (double*)malloc(sizeof(double)*List_YOUSO[39]);

  A = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    A[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  IA = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    IA[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  metric = (double**)malloc(sizeof(double*)*(Matomnum+1));
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    int tno;
    if (Mc_AN==0){
      tno = 1;
    }
    else{
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno = Spe_Total_NO[Cwan];
    }
    metric[Mc_AN] = (double*)malloc(sizeof(double)*tno);
  }

  /* metric used for the norm calculations (identical to the CPU path) */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    for (i=0; i<Spe_Total_NO[Cwan]; i++){
      d = fabs(HisH1[0][0][Mc_AN][0][i][i]-ChemP);
      metric[Mc_AN][i] = 5.0/(d*d+5.0);
    }
  }

  /* shift the residual history (pointer rotation == the CPU content shift)
     and compute the fresh residual slot 0 on the host */

  seed = !(MixH_gpu.valid && MixH_gpu.last_scf_iter+1 == SCF_iter);

  MixH_RotatePtr6(ResidualH1, dim);
  if (SpinP_switch==3) MixH_RotatePtr6(ResidualH2, dim);
  MixH_Residual0Fused(MixH_gpu.h_flat);

  /* bring the device ring up to date */

  if (seed){
    for (m=0; m<MixH_gpu.nslot; m++) MixH_gpu.perm[m] = m;
    if (0U < MixH_gpu.L){
      acc_memcpy_to_device(MixH_gpu.base + MixH_gpu.o_ring,
                           MixH_gpu.h_flat, MixH_gpu.L*sizeof(double));
      for (m=1; m<dim; m++){
        MixH_PackResidualSlot(m, MixH_gpu.h_flat);
        acc_memcpy_to_device(MixH_gpu.base + MixH_gpu.o_ring
                             + (size_t)m*MixH_gpu.L*sizeof(double),
                             MixH_gpu.h_flat, MixH_gpu.L*sizeof(double));
      }
    }
    MixH_gpu.valid = 1;
  }
  else{
    int last = MixH_gpu.perm[MixH_gpu.nslot-1];
    for (m=MixH_gpu.nslot-1; 0<m; m--) MixH_gpu.perm[m] = MixH_gpu.perm[m-1];
    MixH_gpu.perm[0] = last;
    if (0U < MixH_gpu.L){
      acc_memcpy_to_device(MixH_gpu.base + MixH_gpu.o_ring
                           + (size_t)MixH_gpu.perm[0]*MixH_gpu.L*sizeof(double),
                           MixH_gpu.h_flat, MixH_gpu.L*sizeof(double));
    }
  }
  MixH_gpu.last_scf_iter = SCF_iter;

  /* metric weights */

  if (0U < MixH_gpu.L){
    MixH_PackMetricW(metric, MixH_gpu.h_flat);
    acc_memcpy_to_device(MixH_gpu.base + MixH_gpu.o_w,
                         MixH_gpu.h_flat, MixH_gpu.L*sizeof(double));
  }

  /* residual Gram matrix: one device kernel + one Allreduce */

  {
    int npair = dim*(dim+1)/2;
    int *pm = (int*)malloc(sizeof(int)*(size_t)npair);
    int *pn = (int*)malloc(sizeof(int)*(size_t)npair);
    double *tri_my = (double*)malloc(sizeof(double)*(size_t)npair);
    double *tri = (double*)malloc(sizeof(double)*(size_t)npair);
    int p = 0;

    if (pm==NULL || pn==NULL || tri_my==NULL || tri==NULL){
      MixH_AbortWithMessage("Mixing_H.c: failed to allocate the Gram pair tables.");
    }

    for (m=0; m<dim; m++){
      for (n=0; n<=m; n++){
        pm[p] = m;
        pn[p] = n;
        p++;
      }
    }

    MixH_GramDevice(dim, npair, pm, pn, MixH_gpu.perm, tri_my);
    MPI_Allreduce(tri_my, tri, npair, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    for (p=0; p<npair; p++){
      A[pm[p]][pn[p]] = tri[p];
      A[pn[p]][pm[p]] = tri[p];
    }

    free(tri);
    free(tri_my);
    free(pn);
    free(pm);
  }

  NormRD[0] = A[0][0]/(double)atomnum;

  for (m=1; m<=dim; m++){
    A[m-1][dim] = -1.0;
    A[dim][m-1] = -1.0;
  }
  A[dim][dim] = 0.0;

  Inverse(dim,A,IA);

  for (m=1; m<=dim; m++){
    coes[m] = -IA[m-1][dim];
  }

  /* check "nan", "NaN", "inf" or "Inf" (identical to the CPU path) */

  flag_nan = 0;
  for (m=1; m<=dim; m++){

    sprintf(nanchar,"%8.4f",coes[m]);
    if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL
	|| strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

      flag_nan = 1;
    }
  }

  if (flag_nan==1){
    for (m=1; m<=dim; m++){
      coes[m] = 0.0;
    }
    coes[1] = 0.05;
    coes[2] = 0.95;
  }

  /* optimum residual on the device, then the mixing on the host */

  {
    double *cshift = (double*)malloc(sizeof(double)*(size_t)dim);
    if (cshift==NULL){
      MixH_AbortWithMessage("Mixing_H.c: failed to allocate the coefficient buffer.");
    }
    for (m=0; m<dim; m++) cshift[m] = coes[m+1];
    MixH_OptResidualDevice(dim, MixH_gpu.perm, cshift);
    free(cshift);
  }

  if (1.0e-1<=NormRD[0])
    alpha = 0.5;
  else if (1.0e-2<=NormRD[0] && NormRD[0]<1.0e-1)
    alpha = 0.6;
  else if (1.0e-3<=NormRD[0] && NormRD[0]<1.0e-2)
    alpha = 0.7;
  else if (1.0e-4<=NormRD[0] && NormRD[0]<1.0e-3)
    alpha = 0.8;
  else
    alpha = 1.0;

  MixH_MixFromHistory(dim, alpha, coes, MixH_gpu.h_r);

  /* shift the Hamiltonian history (pointer rotation) and save the mixed H */

  MixH_RotatePtr6(HisH1, dim);
  if (SpinP_switch==3) MixH_RotatePtr6(HisH2, dim);
  MixH_SaveHistory0();

  /* freeing of the host work arrays */

  free(coes);

  for (i=0; i<List_YOUSO[39]; i++){
    free(A[i]);
  }
  free(A);

  for (i=0; i<List_YOUSO[39]; i++){
    free(IA[i]);
  }
  free(IA);

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    free(metric[Mc_AN]);
  }
  free(metric);

  (void)MD_iter;
  (void)SCF_iter0;
}

void Pulay_Mixing_H(int MD_iter, int SCF_iter, int SCF_iter0 )
{
  int Mc_AN,Gc_AN,Cwan,Hwan,h_AN,Gh_AN,i,j,spin;
  int dim,m,n,flag_nan,tno;
  double my_sum,tmp1,tmp2,alpha,max_diff,d;
  double r,r10,r11,r12,r13,r20,r21,r22;
  double h,h10,h11,h12,h13,h20,h21,h22;
  double **A,**IA,*coes,**metric;
  char nanchar[300];

  /****************************************************
       determination of dimension of the subspace
  ****************************************************/

  if (SCF_iter<=Num_Mixing_pDM) dim = SCF_iter-1;
  else                          dim = Num_Mixing_pDM;

  /* GPU fast path; MPI-collective on every rank.  On a negative device
     preflight the original CPU code below runs unchanged. */

  if (scf_eigen_lib_flag == GPUSOLVER && 1<=dim){
    int myid_mixh, tier;
    MPI_Comm_rank(mpi_comm_level1,&myid_mixh);
    tier = MixH_GpuPreflight(MD_iter, myid_mixh);
    if (tier == 1){
      Pulay_Mixing_H_GPU(MD_iter, SCF_iter, SCF_iter0, dim);
      return;
    }
    if (tier == 2){
      Pulay_Mixing_H_Inc(MD_iter, SCF_iter, SCF_iter0, dim);
      return;
    }
  }

  /****************************************************
                allocation of arrays 
  ****************************************************/

  coes = (double*)malloc(sizeof(double)*List_YOUSO[39]);

  A = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    A[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  IA = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    IA[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  metric = (double**)malloc(sizeof(double*)*(Matomnum+1));
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    if (Mc_AN==0){
      tno = 1; 
    }
    else{
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      tno = Spe_Total_NO[Cwan];
    }
    metric[Mc_AN] = (double*)malloc(sizeof(double)*tno);
  }  

  /****************************************************
     determine metric used for calculations of norm
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    Cwan = WhatSpecies[Gc_AN];
    for (i=0; i<Spe_Total_NO[Cwan]; i++){
      d = fabs(HisH1[0][0][Mc_AN][0][i][i]-ChemP);
      metric[Mc_AN][i] = 5.0/(d*d+5.0);       
    }
  }

  /****************************************************
                 shift the residual H
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      ResidualH1[m][0][Mc_AN][h_AN][i][j] = ResidualH1[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH1[m][1][Mc_AN][h_AN][i][j] = ResidualH1[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH1[m][2][Mc_AN][h_AN][i][j] = ResidualH1[m-1][2][Mc_AN][h_AN][i][j];
	      ResidualH1[m][3][Mc_AN][h_AN][i][j] = ResidualH1[m-1][3][Mc_AN][h_AN][i][j];

	      ResidualH2[m][0][Mc_AN][h_AN][i][j] = ResidualH2[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH2[m][1][Mc_AN][h_AN][i][j] = ResidualH2[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH2[m][2][Mc_AN][h_AN][i][j] = ResidualH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    ResidualH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j] - HisH1[0][0][Mc_AN][h_AN][i][j];
	    ResidualH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j] - HisH1[0][1][Mc_AN][h_AN][i][j];
	    ResidualH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j] - HisH1[0][2][Mc_AN][h_AN][i][j];
	    ResidualH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j] - HisH1[0][3][Mc_AN][h_AN][i][j];

	    ResidualH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j] - HisH2[0][0][Mc_AN][h_AN][i][j];
	    ResidualH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j] - HisH2[0][1][Mc_AN][h_AN][i][j];
	    ResidualH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j] - HisH2[0][2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else{

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		ResidualH1[m][spin][Mc_AN][h_AN][i][j] = ResidualH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
  	      ResidualH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j] - HisH1[0][spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /****************************************************
          calculation of the residual matrix
  ****************************************************/

  for (m=0; m<dim; m++){
    for (n=0; n<dim; n++){

      my_sum = 0.0;

      if (SpinP_switch==3){

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

		tmp1 = ResidualH1[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][0][Mc_AN][h_AN][i][j];
                my_sum += metric[Mc_AN][i]*tmp1*tmp2; 

		tmp1 = ResidualH1[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][1][Mc_AN][h_AN][i][j];
                my_sum += metric[Mc_AN][i]*tmp1*tmp2; 

		tmp1 = ResidualH1[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][2][Mc_AN][h_AN][i][j];
                my_sum += metric[Mc_AN][i]*tmp1*tmp2; 

		tmp1 = ResidualH1[m][3][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][3][Mc_AN][h_AN][i][j];
                my_sum += metric[Mc_AN][i]*tmp1*tmp2; 

		tmp1 = ResidualH2[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][0][Mc_AN][h_AN][i][j];
                my_sum += metric[Mc_AN][i]*tmp1*tmp2; 

		tmp1 = ResidualH2[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][1][Mc_AN][h_AN][i][j];
                my_sum += metric[Mc_AN][i]*tmp1*tmp2; 

		tmp1 = ResidualH2[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][2][Mc_AN][h_AN][i][j];
                my_sum += metric[Mc_AN][i]*tmp1*tmp2; 
	      }
	    }
	  }
	}

      } /* if (SpinP_switch==3 */

      else{

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	    Gc_AN = M2G[Mc_AN];    
	    Cwan = WhatSpecies[Gc_AN];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){
		  tmp1 = ResidualH1[m][spin][Mc_AN][h_AN][i][j];
		  tmp2 = ResidualH1[n][spin][Mc_AN][h_AN][i][j];
                  my_sum += metric[Mc_AN][i]*tmp1*tmp2; 
		}
	      }
	    }
	  }
	}

      } /* else */

      MPI_Allreduce(&my_sum, &A[m][n], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
      A[n][m] = A[m][n];

    } /* n */
  } /* m */

  NormRD[0] = A[0][0]/(double)atomnum;

  for (m=1; m<=dim; m++){
    A[m-1][dim] = -1.0;
    A[dim][m-1] = -1.0;
  }
  A[dim][dim] = 0.0;

  Inverse(dim,A,IA);

  for (m=1; m<=dim; m++){
    coes[m] = -IA[m-1][dim];
  }

  /****************************************************
            check "nan", "NaN", "inf" or "Inf"
  ****************************************************/

  flag_nan = 0;
  for (m=1; m<=dim; m++){

    sprintf(nanchar,"%8.4f",coes[m]);
    if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	|| strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

      flag_nan = 1;
    }
  }

  if (flag_nan==1){
    for (m=1; m<=dim; m++){
      coes[m] = 0.0;
    }
    coes[1] = 0.05;
    coes[2] = 0.95;
  }

  /****************************************************
      calculation of optimum residual Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            r10 = 0.0; 
            r11 = 0.0;
            r12 = 0.0;
            r13 = 0.0;

            r20 = 0.0; 
            r21 = 0.0;
            r22 = 0.0;

	    for (m=0; m<dim; m++){

	      r10 += ResidualH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r11 += ResidualH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r12 += ResidualH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      r13 += ResidualH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      r20 += ResidualH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r21 += ResidualH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r22 += ResidualH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	    }

            /* optimum Residual H is stored in ResidualH1[dim] and ResidualH2[dim] */

	    ResidualH1[dim][0][Mc_AN][h_AN][i][j] = r10;
	    ResidualH1[dim][1][Mc_AN][h_AN][i][j] = r11;
	    ResidualH1[dim][2][Mc_AN][h_AN][i][j] = r12;
	    ResidualH1[dim][3][Mc_AN][h_AN][i][j] = r13;

	    ResidualH2[dim][0][Mc_AN][h_AN][i][j] = r20;
	    ResidualH2[dim][1][Mc_AN][h_AN][i][j] = r21;
	    ResidualH2[dim][2][Mc_AN][h_AN][i][j] = r22;

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      r = 0.0; 
	      for (m=0; m<dim; m++){
		r += ResidualH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
	      }

              /* optimum Residual H is stored in ResidualH1[dim] */

              ResidualH1[dim][spin][Mc_AN][h_AN][i][j] = r;

	    }
	  }
	}
      }
    }
  }

  /****************************************************
                   mixing of Hamiltonian
  ****************************************************/

  if (1.0e-1<=NormRD[0])
    alpha = 0.5;
  else if (1.0e-2<=NormRD[0] && NormRD[0]<1.0e-1)
    alpha = 0.6;
  else if (1.0e-3<=NormRD[0] && NormRD[0]<1.0e-2)
    alpha = 0.7;
  else if (1.0e-4<=NormRD[0] && NormRD[0]<1.0e-3)
    alpha = 0.8;
  else
    alpha = 1.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            h10 = 0.0; 
            h11 = 0.0;
            h12 = 0.0;
            h13 = 0.0;

            h20 = 0.0; 
            h21 = 0.0;
            h22 = 0.0;

	    for (m=0; m<dim; m++){

	      h10 += HisH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h11 += HisH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h12 += HisH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      h13 += HisH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      h20 += HisH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h21 += HisH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h22 += HisH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	    }

	    H[0][Mc_AN][h_AN][i][j] = h10 + alpha*ResidualH1[dim][0][Mc_AN][h_AN][i][j];
	    H[1][Mc_AN][h_AN][i][j] = h11 + alpha*ResidualH1[dim][1][Mc_AN][h_AN][i][j];
	    H[2][Mc_AN][h_AN][i][j] = h12 + alpha*ResidualH1[dim][2][Mc_AN][h_AN][i][j];
	    H[3][Mc_AN][h_AN][i][j] = h13 + alpha*ResidualH1[dim][3][Mc_AN][h_AN][i][j];
             
	    iHNL[0][Mc_AN][h_AN][i][j] = h20 + alpha*ResidualH2[dim][0][Mc_AN][h_AN][i][j];
	    iHNL[1][Mc_AN][h_AN][i][j] = h21 + alpha*ResidualH2[dim][1][Mc_AN][h_AN][i][j];
	    iHNL[2][Mc_AN][h_AN][i][j] = h22 + alpha*ResidualH2[dim][2][Mc_AN][h_AN][i][j];

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      r = 0.0; 
	      h = 0.0;

	      for (m=0; m<dim; m++){
		r += ResidualH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
		h += HisH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
	      }

	      H[spin][Mc_AN][h_AN][i][j] = h + alpha*r;
	    }
	  }
	}
      }
    }
  }

  /****************************************************
                  shifting of Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      HisH1[m][0][Mc_AN][h_AN][i][j] = HisH1[m-1][0][Mc_AN][h_AN][i][j];
	      HisH1[m][1][Mc_AN][h_AN][i][j] = HisH1[m-1][1][Mc_AN][h_AN][i][j];
	      HisH1[m][2][Mc_AN][h_AN][i][j] = HisH1[m-1][2][Mc_AN][h_AN][i][j];
	      HisH1[m][3][Mc_AN][h_AN][i][j] = HisH1[m-1][3][Mc_AN][h_AN][i][j];

	      HisH2[m][0][Mc_AN][h_AN][i][j] = HisH2[m-1][0][Mc_AN][h_AN][i][j];
	      HisH2[m][1][Mc_AN][h_AN][i][j] = HisH2[m-1][1][Mc_AN][h_AN][i][j];
	      HisH2[m][2][Mc_AN][h_AN][i][j] = HisH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    HisH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j];
	    HisH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j];
	    HisH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j];
	    HisH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j];

	    HisH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j];
	    HisH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j];
	    HisH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else {

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		HisH1[m][spin][Mc_AN][h_AN][i][j] = HisH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
	      HisH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /****************************************************
                   freeing of arrays 
  ****************************************************/

  free(coes);

  for (i=0; i<List_YOUSO[39]; i++){
    free(A[i]);
  }
  free(A);

  for (i=0; i<List_YOUSO[39]; i++){
    free(IA[i]);
  }
  free(IA);

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    if (Mc_AN==0){
      tno = 1; 
    }
    else{
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      tno = Spe_Total_NO[Cwan];
    }
    free(metric[Mc_AN]);
  }  
  free(metric);

}



void Simple_Mixing_H(int MD_iter, int SCF_iter, int SCF_iter0 )
{
  int m,Mc_AN,Gc_AN,Cwan,spin,dim;
  int i,j,h_AN,Gh_AN,Hwan,ian,jan,n;
  double w1,w2,My_Norm,Norm;
  double d0,d1,d2,d3,d,tmp0;
  double Mix_wgt,Min_Weight,Max_Weight;
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* start... */

  Min_Weight = Min_Mixing_weight;
  if (SCF_RENZOKU==-1){
    Max_Weight = Max_Mixing_weight;
    Max_Mixing_weight2 = Max_Mixing_weight;
  }
  else if (SCF_RENZOKU==1000){  /* past 3 */
    Max_Mixing_weight2 = 2.0*Max_Mixing_weight2;
    if (0.7<Max_Mixing_weight2) Max_Mixing_weight2 = 0.7;
    Max_Weight = Max_Mixing_weight2;
    SCF_RENZOKU = 0;
  }
  else{
    Max_Weight = Max_Mixing_weight2;
  }

  /* determination of dim */

  if (SCF_iter<Num_Mixing_pDM) dim = SCF_iter;
  else                         dim = Num_Mixing_pDM;

  /****************************************************
                  shift the residual H
  ****************************************************/

  if (Mixing_switch!=1 && Mixing_switch!=6){

    if (SpinP_switch==3){

      /* shift the residual Hamiltonian */

      for (m=(dim-1); 0<m; m--){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

		ResidualH1[m][0][Mc_AN][h_AN][i][j] = ResidualH1[m-1][0][Mc_AN][h_AN][i][j];
		ResidualH1[m][1][Mc_AN][h_AN][i][j] = ResidualH1[m-1][1][Mc_AN][h_AN][i][j];
		ResidualH1[m][2][Mc_AN][h_AN][i][j] = ResidualH1[m-1][2][Mc_AN][h_AN][i][j];
		ResidualH1[m][3][Mc_AN][h_AN][i][j] = ResidualH1[m-1][3][Mc_AN][h_AN][i][j];

		ResidualH2[m][0][Mc_AN][h_AN][i][j] = ResidualH2[m-1][0][Mc_AN][h_AN][i][j];
		ResidualH2[m][1][Mc_AN][h_AN][i][j] = ResidualH2[m-1][1][Mc_AN][h_AN][i][j];
		ResidualH2[m][2][Mc_AN][h_AN][i][j] = ResidualH2[m-1][2][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }

      /* calculate the current residual Hamiltonian */

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      ResidualH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j] - HisH1[0][0][Mc_AN][h_AN][i][j];
	      ResidualH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j] - HisH1[0][1][Mc_AN][h_AN][i][j];
	      ResidualH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j] - HisH1[0][2][Mc_AN][h_AN][i][j];
	      ResidualH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j] - HisH1[0][3][Mc_AN][h_AN][i][j];

	      ResidualH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j] - HisH2[0][0][Mc_AN][h_AN][i][j];
	      ResidualH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j] - HisH2[0][1][Mc_AN][h_AN][i][j];
	      ResidualH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j] - HisH2[0][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }

    }

    else{

      /* shift the residual Hamiltonian */

      for (m=(dim-1); 0<m; m--){
	for (spin=0; spin<=SpinP_switch; spin++){
	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	    Gc_AN = M2G[Mc_AN];    
	    Cwan = WhatSpecies[Gc_AN];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){
		  ResidualH1[m][spin][Mc_AN][h_AN][i][j] = ResidualH1[m-1][spin][Mc_AN][h_AN][i][j];
		}
	      }
	    }
	  }
	}
      }

      /* calculate the current residual Hamiltonian */

      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		ResidualH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j] - HisH1[0][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }

    } /* else */

  } /* if (Mixing_switch!=1 && Mixing_switch!=6) */

  /****************************************************
              calculation of NormRH
  ****************************************************/

  My_Norm = 0.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    d0 = HisH1[0][0][Mc_AN][h_AN][i][j] - H[0][Mc_AN][h_AN][i][j];  
	    d1 = HisH1[0][1][Mc_AN][h_AN][i][j] - H[1][Mc_AN][h_AN][i][j];  
	    d2 = HisH1[0][2][Mc_AN][h_AN][i][j] - H[2][Mc_AN][h_AN][i][j];  
	    d3 = HisH1[0][3][Mc_AN][h_AN][i][j] - H[3][Mc_AN][h_AN][i][j];  
            My_Norm += d0*d0 + d1*d1 + d2*d2 + d3*d3;

            d0 = HisH2[0][0][Mc_AN][h_AN][i][j] - iHNL[0][Mc_AN][h_AN][i][j];
            d1 = HisH2[0][1][Mc_AN][h_AN][i][j] - iHNL[1][Mc_AN][h_AN][i][j];
            d2 = HisH2[0][2][Mc_AN][h_AN][i][j] - iHNL[2][Mc_AN][h_AN][i][j];
            My_Norm += d0*d0 + d1*d1 + d2*d2;

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
  	      d = HisH1[0][spin][Mc_AN][h_AN][i][j] - H[spin][Mc_AN][h_AN][i][j];  
              My_Norm += d*d;
	    }
	  }
	}
      }
    }
  }

  /****************************************************
    MPI: 

    My_Norm
  ****************************************************/

  MPI_Allreduce(&My_Norm, &Norm, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /****************************************************
    find an optimum mixing weight
  ****************************************************/

  for (i=4; 1<=i; i--){
    NormRD[i] = NormRD[i-1];
    History_Uele[i] = History_Uele[i-1];
  }
  NormRD[0] = Norm/(double)atomnum;;
  History_Uele[0] = Uele;

  if (1<SCF_iter){

    if ( (int)sgn(History_Uele[0]-History_Uele[1])
	 ==(int)sgn(History_Uele[1]-History_Uele[2])
	 && NormRD[0]<NormRD[1]){

      /* tmp0 = 1.6*Mixing_weight; */

      tmp0 = NormRD[1]/(largest(NormRD[1]-NormRD[0],10e-10))*Mixing_weight;

      if (tmp0<Max_Weight){
	if (Min_Weight<tmp0){
	  Mixing_weight = tmp0;
	}
	else{ 
	  Mixing_weight = Min_Weight;
	}
      }
      else{ 
	Mixing_weight = Max_Weight;
	SCF_RENZOKU++;  
      }
    }
   
    else if ( (int)sgn(History_Uele[0]-History_Uele[1])
	      ==(int)sgn(History_Uele[1]-History_Uele[2])
	      && NormRD[1]<NormRD[0]){

      tmp0 = NormRD[1]/(largest(NormRD[1]+NormRD[0],10e-10))*Mixing_weight;

      /* tmp0 = Mixing_weight/1.6; */

      if (tmp0<Max_Weight){
	if (Min_Weight<tmp0)
	  Mixing_weight = tmp0;
	else 
	  Mixing_weight = Min_Weight;
      }
      else{
	Mixing_weight = Max_Weight;
      }

      SCF_RENZOKU = -1;  
    }

    else if ( (int)sgn(History_Uele[0]-History_Uele[1])
	      !=(int)sgn(History_Uele[1]-History_Uele[2])
	      && NormRD[0]<NormRD[1]){

      /* tmp0 = Mixing_weight*1.2; */

      tmp0 = NormRD[1]/(largest(NormRD[1]-NormRD[0],10e-10))*Mixing_weight;

      if (tmp0<Max_Weight){
	if (Min_Weight<tmp0)
	  Mixing_weight = tmp0;
	else 
	  Mixing_weight = Min_Weight;
      }
      else{ 
	Mixing_weight = Max_Weight;
	SCF_RENZOKU++;
      }
    }

    else if ( (int)sgn(History_Uele[0]-History_Uele[1])
	      !=(int)sgn(History_Uele[1]-History_Uele[2])
	      && NormRD[1]<NormRD[0]){

      /* tmp0 = Mixing_weight/2.0; */

      tmp0 = NormRD[1]/(largest(NormRD[1]+NormRD[0],10e-10))*Mixing_weight;

      if (tmp0<Max_Weight){
	if (Min_Weight<tmp0)
	  Mixing_weight = tmp0;
	else 
	  Mixing_weight = Min_Weight;
      }
      else 
	Mixing_weight = Max_Weight;

      SCF_RENZOKU = -1;
    }
  }

  Mix_wgt = Mixing_weight;

  /****************************************************
            finding a proper mixing weight
  ****************************************************/

  if (SCF_iter==1){
    w1 = 1.0;
    w2 = 1.0 - w1;
  }
  else{
    w1 = Mix_wgt;
    w2 = 1.0 - w1;
  }

  /****************************************************
       performing the simple mixing for Hamiltonian 
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      HisH1[m][0][Mc_AN][h_AN][i][j] = HisH1[m-1][0][Mc_AN][h_AN][i][j];
	      HisH1[m][1][Mc_AN][h_AN][i][j] = HisH1[m-1][1][Mc_AN][h_AN][i][j];
	      HisH1[m][2][Mc_AN][h_AN][i][j] = HisH1[m-1][2][Mc_AN][h_AN][i][j];
	      HisH1[m][3][Mc_AN][h_AN][i][j] = HisH1[m-1][3][Mc_AN][h_AN][i][j];

	      HisH2[m][0][Mc_AN][h_AN][i][j] = HisH2[m-1][0][Mc_AN][h_AN][i][j];
	      HisH2[m][1][Mc_AN][h_AN][i][j] = HisH2[m-1][1][Mc_AN][h_AN][i][j];
	      HisH2[m][2][Mc_AN][h_AN][i][j] = HisH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* mix the current Hamiltonian and the last one */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    H[0][Mc_AN][h_AN][i][j] = w2*HisH1[1][0][Mc_AN][h_AN][i][j] + w1*H[0][Mc_AN][h_AN][i][j];  
	    H[1][Mc_AN][h_AN][i][j] = w2*HisH1[1][1][Mc_AN][h_AN][i][j] + w1*H[1][Mc_AN][h_AN][i][j];  
	    H[2][Mc_AN][h_AN][i][j] = w2*HisH1[1][2][Mc_AN][h_AN][i][j] + w1*H[2][Mc_AN][h_AN][i][j];  
	    H[3][Mc_AN][h_AN][i][j] = w2*HisH1[1][3][Mc_AN][h_AN][i][j] + w1*H[3][Mc_AN][h_AN][i][j];  

	    HisH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j];
	    HisH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j];
	    HisH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j];
	    HisH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j];

	    iHNL[0][Mc_AN][h_AN][i][j] = w2*HisH2[1][0][Mc_AN][h_AN][i][j] + w1*iHNL[0][Mc_AN][h_AN][i][j];
	    iHNL[1][Mc_AN][h_AN][i][j] = w2*HisH2[1][1][Mc_AN][h_AN][i][j] + w1*iHNL[1][Mc_AN][h_AN][i][j];
	    iHNL[2][Mc_AN][h_AN][i][j] = w2*HisH2[1][2][Mc_AN][h_AN][i][j] + w1*iHNL[2][Mc_AN][h_AN][i][j];

	    HisH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j];
	    HisH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j];
	    HisH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else {

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		HisH1[m][spin][Mc_AN][h_AN][i][j] = HisH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* mix the current Hamiltonian and the last one */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
	      H[spin][Mc_AN][h_AN][i][j] = w2*HisH1[1][spin][Mc_AN][h_AN][i][j] + w1*H[spin][Mc_AN][h_AN][i][j];
	      HisH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /* In case of RMM-DIIS */

  if (Mixing_switch==1 || Mixing_switch==6){

    for (spin=0; spin<=SpinP_switch; spin++){

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];
	ian = Spe_Total_CNO[WhatSpecies[Gc_AN]];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];      
	  jan = Spe_Total_CNO[WhatSpecies[Gh_AN]];

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
  
              ResidualDM[2][spin][Mc_AN][h_AN][m][n] = DM[0][spin][Mc_AN][h_AN][m][n]
                                                      -DM[1][spin][Mc_AN][h_AN][m][n];

	      DM[2][spin][Mc_AN][h_AN][m][n] = DM[1][spin][Mc_AN][h_AN][m][n];
	      DM[1][spin][Mc_AN][h_AN][m][n] = DM[0][spin][Mc_AN][h_AN][m][n];
	    }
	  }

	  if ( (SpinP_switch==3 && ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
             || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1 )) && spin<=1 ){ 

	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){

	        iResidualDM[2][spin][Mc_AN][h_AN][m][n] = iDM[0][spin][Mc_AN][h_AN][m][n]
                                                         -iDM[1][spin][Mc_AN][h_AN][m][n];

		iDM[2][spin][Mc_AN][h_AN][m][n] = iDM[1][spin][Mc_AN][h_AN][m][n];
		iDM[1][spin][Mc_AN][h_AN][m][n] = iDM[0][spin][Mc_AN][h_AN][m][n];
	      }
	    }
	  }

	}
      }

    } /* spin */

  } /* if (Mixing_switch==1 || Mixing_switch==6) */

}





void Inverse(int n, double **a, double **ia)
{
  int method_flag=2;

  if (method_flag==0){

  /****************************************************
                  LU decomposition
                      0 to n
   NOTE:
   This routine does not consider the reduction of rank
  ****************************************************/

  int i,j,k;
  double w;
  double *x,*y;
  double **da;

  /***************************************************
    allocation of arrays: 

     x[List_YOUSO[39]]
     y[List_YOUSO[39]]
     da[List_YOUSO[39]][List_YOUSO[39]]
  ***************************************************/

  x = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  y = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    x[i] = 0.0;
    y[i] = 0.0;
  }

  da = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    da[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
    for (j=0; j<List_YOUSO[39]; j++){
      da[i][j] = 0.0;
    }
  }

  /* start calc. */

  if (n==-1){
    for (i=0; i<List_YOUSO[39]; i++){
      for (j=0; j<List_YOUSO[39]; j++){
	a[i][j] = 0.0;
      }
    }
  }
  else{
    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	da[i][j] = a[i][j];
      }
    }

    /****************************************************
                     LU factorization
    ****************************************************/

    for (k=0; k<=n-1; k++){
      w = 1.0/a[k][k];
      for (i=k+1; i<=n; i++){
	a[i][k] = w*a[i][k];
	for (j=k+1; j<=n; j++){
	  a[i][j] = a[i][j] - a[i][k]*a[k][j];
	}
      }
    }

    for (k=0; k<=n; k++){

      /****************************************************
                             Ly = b
      ****************************************************/

      for (i=0; i<=n; i++){
	if (i==k)
	  y[i] = 1.0;
	else
	  y[i] = 0.0;
	for (j=0; j<=i-1; j++){
	  y[i] = y[i] - a[i][j]*y[j];
	}
      }

      /****************************************************
                             Ux = y 
      ****************************************************/

      for (i=n; 0<=i; i--){
	x[i] = y[i];
	for (j=n; (i+1)<=j; j--){
	  x[i] = x[i] - a[i][j]*x[j];
	}
	x[i] = x[i]/a[i][i];
	ia[i][k] = x[i];
      }
    }

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	a[i][j] = da[i][j];
      }
    }
  }

  /***************************************************
    freeing of arrays: 

     x[List_YOUSO[39]]
     y[List_YOUSO[39]]
     da[List_YOUSO[39]][List_YOUSO[39]]
  ***************************************************/

  free(x);
  free(y);

  for (i=0; i<List_YOUSO[39]; i++){
    free(da[i]);
  }
  free(da);

  }
  
  else if (method_flag==1){

    int i,j,M,N,LDA,INFO;
    int *IPIV,LWORK;
    double *A,*WORK;

    A = (double*)malloc(sizeof(double)*(n+2)*(n+2));
    WORK = (double*)malloc(sizeof(double)*(n+2));
    IPIV = (int*)malloc(sizeof(int)*(n+2));

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
        A[i*(n+1)+j] = a[i][j];
      }
    }

    M = n + 1;
    N = M;
    LDA = M;
    LWORK = M;

    F77_NAME(dgetrf,DGETRF)( &M, &N, A, &LDA, IPIV, &INFO);
    F77_NAME(dgetri,DGETRI)( &N, A, &LDA, IPIV, WORK, &LWORK, &INFO);

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
        ia[i][j] = A[i*(n+1)+j];
      }
    }

    free(A);
    free(WORK);
    free(IPIV);
  }

  else if (method_flag==2){

    int N,i,j,k;
    double *A,*B,*ko;
    double sum;

    N = n + 1;

    A = (double*)malloc(sizeof(double)*(N+2)*(N+2));
    B = (double*)malloc(sizeof(double)*(N+2)*(N+2));
    ko = (double*)malloc(sizeof(double)*(N+2));

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        A[j*N+i] = a[i][j];
      }
    }

    Eigen_lapack3(A, ko, N, N); 

    for (i=0; i<N; i++){
      ko[i] = 1.0/(ko[i]+1.0e-13);
    } 

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        B[i*N+j] = A[i*N+j]*ko[i];
      }
    }

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        ia[i][j] = 0.0;
      }
    }

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        sum = 0.0;
	for (k=0; k<N; k++){
	  sum += A[k*N+i]*B[k*N+j];
	}
        ia[i][j] = sum;
      }
    }

    free(A);
    free(B);
    free(ko);
  }
}


void Inverse0(int n, double **a, double **ia)
{
  /****************************************************
                  LU decomposition
                      0 to n
   NOTE:
   This routine does not consider the reduction of rank
  ****************************************************/

  int i,j,k;
  double w;
  double *x,*y;
  double **da;

  /***************************************************
    allocation of arrays: 

    x[List_YOUSO[39]]
    y[List_YOUSO[39]]
    da[List_YOUSO[39]][List_YOUSO[39]]
  ***************************************************/

  x = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  y = (double*)malloc(sizeof(double)*List_YOUSO[39]);

  da = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    da[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  /* start calc. */

  if (n==-1){
    for (i=0; i<List_YOUSO[39]; i++){
      for (j=0; j<List_YOUSO[39]; j++){
	a[i][j] = 0.0;
      }
    }
  }
  else{
    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	da[i][j] = a[i][j];
      }
    }

    /****************************************************
                     LU factorization
    ****************************************************/

    for (k=0; k<=n-1; k++){
      w = 1.0/a[k][k];
      for (i=k+1; i<=n; i++){
	a[i][k] = w*a[i][k];
	for (j=k+1; j<=n; j++){
	  a[i][j] = a[i][j] - a[i][k]*a[k][j];
	}
      }
    }
    for (k=0; k<=n; k++){

      /****************************************************
                             Ly = b
      ****************************************************/

      for (i=0; i<=n; i++){
	if (i==k)
	  y[i] = 1.0;
	else
	  y[i] = 0.0;
	for (j=0; j<=i-1; j++){
	  y[i] = y[i] - a[i][j]*y[j];
	}
      }

      /****************************************************
                             Ux = y 
      ****************************************************/

      for (i=n; 0<=i; i--){
	x[i] = y[i];
	for (j=n; (i+1)<=j; j--){
	  x[i] = x[i] - a[i][j]*x[j];
	}
	x[i] = x[i]/a[i][i];
	ia[i][k] = x[i];
      }
    }

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	a[i][j] = da[i][j];
      }
    }
  }

  /***************************************************
    freeing of arrays: 

     x[List_YOUSO[39]]
     y[List_YOUSO[39]]
     da[List_YOUSO[39]][List_YOUSO[39]]
  ***************************************************/

  free(x);
  free(y);

  for (i=0; i<List_YOUSO[39]; i++){
    free(da[i]);
  }
  free(da);
}
