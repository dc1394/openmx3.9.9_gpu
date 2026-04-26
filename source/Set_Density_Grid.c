/**********************************************************************
  Set_Density_Grid.c:

     Set_Density_Grid.c is a subroutine to calculate a charge density
     on grid by one-particle wave functions.

  Log of Set_Density_Grid.c:

     22/Nov/2001  Released by T. Ozaki
     19/Apr/2013  Modified by A.M. Ito

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

#define  measure_time   0

static void SDG_Abort(const char *where, const char *msg)
{
  int myid = 0;

  MPI_Comm_rank(mpi_comm_level1,&myid);
  fprintf(stderr,"Set_Density_Grid: %s: %s on rank %d\n",where,msg,myid);
  fflush(stderr);
  MPI_Abort(mpi_comm_level1,1);
}

static void *SDG_Malloc(size_t count, size_t size, const char *where)
{
  void *ptr;

  if (size!=0 && SIZE_MAX/size<count){
    SDG_Abort(where,"allocation size overflow");
  }

  ptr = malloc(count*size==0 ? 1 : count*size);
  if (ptr==NULL){
    SDG_Abort(where,"malloc failed");
  }

  return ptr;
}

static void SDG_Add_Size(size_t *total, size_t add, const char *where)
{
  if (total==NULL){
    SDG_Abort(where,"invalid size accumulator");
  }
  if (SIZE_MAX-add<*total){
    SDG_Abort(where,"size accumulation overflow");
  }

  *total += add;
}

static size_t SDG_Product_Size(size_t a, size_t b, const char *where)
{
  if (b!=0 && SIZE_MAX/b<a){
    SDG_Abort(where,"size overflow");
  }

  return a*b;
}

static int SDG_Checked_MPI_Count(int count, int factor, const char *where)
{
  if (count<0 || factor<0){
    SDG_Abort(where,"negative MPI count");
  }
  if (factor!=0 && INT_MAX/factor<count){
    SDG_Abort(where,"MPI count overflow");
  }

  return count*factor;
}

static size_t SDG_Size_From_Int(int count, const char *where)
{
  if (count<0){
    SDG_Abort(where,"negative size");
  }

  return (size_t)count;
}

static void SDG_Add_Product_To_MPI_Count(int *count, int a, int b, const char *where)
{
  int product;

  if (count==NULL || *count<0){
    SDG_Abort(where,"invalid count");
  }
  product = SDG_Checked_MPI_Count(a,b,where);
  if (INT_MAX-product<*count){
    SDG_Abort(where,"MPI count accumulation overflow");
  }

  *count += product;
}

static unsigned long long SDG_Grid2D_Size(void)
{
  if (Ngrid1<=0 || Ngrid2<=0){
    SDG_Abort("grid size","non-positive Ngrid1 or Ngrid2");
  }

  return (unsigned long long)Ngrid1*(unsigned long long)Ngrid2;
}

static unsigned long long SDG_Grid2D_Index_From_GN(int GN,
                                                   unsigned long long N2D)
{
  unsigned long long n2D;

  if (GN<0){
    SDG_Abort("grid index","negative grid index");
  }
  if (Ngrid3<=0){
    SDG_Abort("grid index","non-positive Ngrid3");
  }

  n2D = (unsigned long long)(GN/Ngrid3);
  if (N2D<=n2D){
    SDG_Abort("grid index","out-of-range grid index");
  }

  return n2D;
}

static int SDG_Find_Peer_Slot(const int *peer_list, int num_peers, int peer, const char *where)
{
  int i;

  if (peer_list==NULL || num_peers<0){
    SDG_Abort(where,"invalid peer list");
  }

  for (i=0; i<num_peers; i++){
    if (peer_list[i]==peer) return i;
  }

  SDG_Abort(where,"missing self peer slot");
  return -1;
}

static void SDG_Check_Orbital_Counts(int NO0, int NO1, const char *where)
{
  if (NO0<0 || List_YOUSO[7]<NO0 || NO1<0 || List_YOUSO[7]<NO1){
    SDG_Abort(where,"orbital count exceeds temporary buffer");
  }
}



double Set_Density_Grid(int Cnt_kind, int Calc_CntOrbital_ON, double *****CDM, double **Density_Grid_B0)
{
  static int firsttime=1;
  int al,L0,Mul0,M0,p,size1,size2,spin_components;
  int Gc_AN,Mc_AN,Mh_AN,LN,AN,BN,CN;
  int Cwan,NO0,NO1,Rn,N,Hwan,i,j,k,n;
  int NN_S,NN_R;
  unsigned long long int N2D,n2D;
  int GN;
  int Max_Size,My_Max;
  size_t size_Tmp_Den_Grid;
  size_t tmp_den_grid_per_spin;
  size_t size_Den_Snd_Grid_A2B;
  size_t size_Den_Rcv_Grid_A2B;
  int h_AN,Gh_AN,Rnh,spin,Nc,GRc,Nh,Nog;
  int Nc_0,Nc_1,Nc_2,Nc_3,Nh_0,Nh_1,Nh_2,Nh_3;

  double threshold;
  double tmp0,tmp1,sk1,sk2,sk3,tot_den,sum;
  double tmp0_0,tmp0_1,tmp0_2,tmp0_3;
  double sum_0,sum_1,sum_2,sum_3;
  double d1,d2,d3,cop,sip,sit,cot;
  double x,y,z,Cxyz[4];
  double TStime,TEtime;
  double ***Tmp_Den_Grid;
  double **Tmp_Den_Grid_Data;
  double **Den_Snd_Grid_A2B;
  double **Den_Rcv_Grid_A2B;
  double *Den_Snd_Grid_A2B_Data;
  double *Den_Rcv_Grid_A2B_Data;
  double *tmp_array;
  double *tmp_array2;
  double *orbs0,*orbs1;
  double *orbs0_0,*orbs0_1,*orbs0_2,*orbs0_3;
  double *orbs1_0,*orbs1_1,*orbs1_2,*orbs1_3;
  double ***tmp_CDM;
  int *Snd_Size,*Rcv_Size;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_atom, Etime_atom;
  double time0,time1,time2;

  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  /* for OpenMP */
  int OMPID,Nthrds;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (SpinP_switch!=0 && SpinP_switch!=1 && SpinP_switch!=3){
    SDG_Abort("SpinP_switch","unsupported spin mode");
  }
  spin_components = SpinP_switch + 1;

  dtime(&TStime);

  /* allocation of arrays */

  tmp_den_grid_per_spin = 1;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = F_M2G[Mc_AN];
    if (GridN_Atom[Gc_AN]<0){
      SDG_Abort("Tmp_Den_Grid","negative GridN_Atom");
    }
    SDG_Add_Size(&tmp_den_grid_per_spin,(size_t)GridN_Atom[Gc_AN],
                 "Tmp_Den_Grid");
  }

  size_Tmp_Den_Grid =
    SDG_Product_Size((size_t)spin_components,tmp_den_grid_per_spin-1,
                     "Tmp_Den_Grid size");

  Tmp_Den_Grid = (double***)SDG_Malloc(SDG_Size_From_Int(spin_components,"spin_components"),
                                       sizeof(double**),
                                       "Tmp_Den_Grid");
  Tmp_Den_Grid_Data = (double**)SDG_Malloc(SDG_Size_From_Int(spin_components,
                                                             "spin_components"),
                                           sizeof(double*),
                                           "Tmp_Den_Grid_Data");

  for (i=0; i<spin_components; i++){
    size_t offset = 1;

    Tmp_Den_Grid[i] = (double**)SDG_Malloc(SDG_Size_From_Int(Matomnum+1,
                                                             "Matomnum+1"),
                                           sizeof(double*),
                                           "Tmp_Den_Grid[i]");
    Tmp_Den_Grid_Data[i] = (double*)SDG_Malloc(tmp_den_grid_per_spin,
                                               sizeof(double),
                                               "Tmp_Den_Grid_Data[i]");
    memset(Tmp_Den_Grid_Data[i],0,
           SDG_Product_Size(tmp_den_grid_per_spin,sizeof(double),
                            "Tmp_Den_Grid_Data memset"));
    Tmp_Den_Grid[i][0] = Tmp_Den_Grid_Data[i];

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = F_M2G[Mc_AN];
      Tmp_Den_Grid[i][Mc_AN] = &Tmp_Den_Grid_Data[i][offset];
      offset += (size_t)GridN_Atom[Gc_AN];
    }
  }

  size_Den_Snd_Grid_A2B = 0;
  Den_Snd_Grid_A2B = (double**)SDG_Malloc(SDG_Size_From_Int(numprocs,"numprocs"),
                                          sizeof(double*),
                                          "Den_Snd_Grid_A2B");
  for (ID=0; ID<numprocs; ID++){
    int count = SDG_Checked_MPI_Count(Num_Snd_Grid_A2B[ID],spin_components,
                                      "Den_Snd_Grid_A2B");
    SDG_Add_Size(&size_Den_Snd_Grid_A2B,(size_t)count,
                 "Den_Snd_Grid_A2B size");
  }
  Den_Snd_Grid_A2B_Data = (double*)SDG_Malloc(size_Den_Snd_Grid_A2B,
                                             sizeof(double),
                                             "Den_Snd_Grid_A2B_Data");
  {
    size_t offset = 0;
    for (ID=0; ID<numprocs; ID++){
      int count = SDG_Checked_MPI_Count(Num_Snd_Grid_A2B[ID],spin_components,
                                        "Den_Snd_Grid_A2B");
      Den_Snd_Grid_A2B[ID] = &Den_Snd_Grid_A2B_Data[offset];
      offset += (size_t)count;
    }
  }

  size_Den_Rcv_Grid_A2B = 0;
  Den_Rcv_Grid_A2B = (double**)SDG_Malloc(SDG_Size_From_Int(numprocs,"numprocs"),
                                          sizeof(double*),
                                          "Den_Rcv_Grid_A2B");
  for (ID=0; ID<numprocs; ID++){
    int count = SDG_Checked_MPI_Count(Num_Rcv_Grid_A2B[ID],spin_components,
                                      "Den_Rcv_Grid_A2B");
    SDG_Add_Size(&size_Den_Rcv_Grid_A2B,(size_t)count,
                 "Den_Rcv_Grid_A2B size");
  }
  Den_Rcv_Grid_A2B_Data = (double*)SDG_Malloc(size_Den_Rcv_Grid_A2B,
                                             sizeof(double),
                                             "Den_Rcv_Grid_A2B_Data");
  {
    size_t offset = 0;
    for (ID=0; ID<numprocs; ID++){
      int count = SDG_Checked_MPI_Count(Num_Rcv_Grid_A2B[ID],spin_components,
                                        "Den_Rcv_Grid_A2B");
      Den_Rcv_Grid_A2B[ID] = &Den_Rcv_Grid_A2B_Data[offset];
      offset += (size_t)count;
    }
  }

  /* PrintMemory */

  if (firsttime==1){
    PrintMemory("Set_Density_Grid: AtomDen_Grid",    (long int)(sizeof(double)*size_Tmp_Den_Grid), NULL);
    PrintMemory("Set_Density_Grid: Den_Snd_Grid_A2B",(long int)(sizeof(double)*size_Den_Snd_Grid_A2B), NULL);
    PrintMemory("Set_Density_Grid: Den_Rcv_Grid_A2B",(long int)(sizeof(double)*size_Den_Rcv_Grid_A2B), NULL);
    firsttime = 0;
  }

  /****************************************************
                when orbital optimization
  ****************************************************/

  if (Calc_CntOrbital_ON==1 && Cnt_kind==0 && Cnt_switch==1){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      /* COrbs_Grid */

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      NO0 = Spe_Total_CNO[Cwan];
      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

        al = -1;
	for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_CBasis[Cwan][L0]; Mul0++){
	    for (M0=0; M0<=2*L0; M0++){

	      al++;
	      tmp0 = 0.0;

	      for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
	        j = Spe_Trans_Orbital[Cwan][al][p];
	        tmp0 += CntCoes[Mc_AN][al][p]*Orbs_Grid[Mc_AN][Nc][j];/* AITUNE */
	      }

	      COrbs_Grid[Mc_AN][al][Nc] = (Type_Orbs_Grid)tmp0;
	    }
	  }
        }
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

    /**********************************************
     MPI:

     COrbs_Grid
    ***********************************************/

    /* allocation of arrays  */
    Snd_Size = (int*)SDG_Malloc(SDG_Size_From_Int(numprocs,"numprocs"),
                                sizeof(int),"Snd_Size");
    Rcv_Size = (int*)SDG_Malloc(SDG_Size_From_Int(numprocs,"numprocs"),
                                sizeof(int),"Rcv_Size");

    /* find data size for sending and receiving */

    My_Max = -10000;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){
        /*  sending size */
        if (F_Snd_Num[IDS]!=0){
          /* find data size */
          size1 = 0;
          for (n=0; n<F_Snd_Num[IDS]; n++){
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN];
            SDG_Add_Product_To_MPI_Count(&size1,GridN_Atom[Gc_AN],
                                         Spe_Total_CNO[Cwan],"COrbs_Grid send size");
          }

          Snd_Size[IDS] = size1;
          MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
        }
        else{
          Snd_Size[IDS] = 0;
        }

        /* receiving size */
        if (F_Rcv_Num[IDR]!=0){
          MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
          Rcv_Size[IDR] = size2;
        }
        else{
          Rcv_Size[IDR] = 0;
        }
        if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);
      }
      else{
        Snd_Size[IDS] = 0;
        Rcv_Size[IDR] = 0;
      }

      if (My_Max<Snd_Size[IDS]) My_Max = Snd_Size[IDS];
      if (My_Max<Rcv_Size[IDR]) My_Max = Rcv_Size[IDR];

    }

    MPI_Allreduce(&My_Max, &Max_Size, 1, MPI_INT, MPI_MAX, mpi_comm_level1);
    /* allocation of arrays */
    if (Max_Size<0){
      SDG_Abort("COrbs_Grid","negative maximum transfer size");
    }
    tmp_array  = (double*)SDG_Malloc((size_t)Max_Size,sizeof(double),"tmp_array");
    tmp_array2 = (double*)SDG_Malloc((size_t)Max_Size,sizeof(double),"tmp_array2");

    /* send and receive COrbs_Grid */

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){

        /* sending of data */

        if (F_Snd_Num[IDS]!=0){

          /* find data size */
          size1 = Snd_Size[IDS];

          /* multidimentional array to vector array */
          k = 0;
          for (n=0; n<F_Snd_Num[IDS]; n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN];
            NO0 = Spe_Total_CNO[Cwan];
            for (i=0; i<NO0; i++){
              for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
                tmp_array[k] = COrbs_Grid[Mc_AN][i][Nc];
                k++;
              }
            }
          }

          /* MPI_Isend */
          MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS,
                    tag, mpi_comm_level1, &request);
        }

        /* receiving of block data */

        if (F_Rcv_Num[IDR]!=0){

          /* find data size */
          size2 = Rcv_Size[IDR];

          /* MPI_Recv */
          MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

          k = 0;
          Mc_AN = F_TopMAN[IDR] - 1;
          for (n=0; n<F_Rcv_Num[IDR]; n++){
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN];
            NO0 = Spe_Total_CNO[Cwan];

            for (i=0; i<NO0; i++){
              for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
                COrbs_Grid[Mc_AN][i][Nc] = tmp_array2[k];
                k++;
              }
            }
          }
        }
        if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);
      }
    }

    /* freeing of arrays  */
    free(tmp_array);
    free(tmp_array2);
    free(Snd_Size);
    free(Rcv_Size);
  }

  /**********************************************
              calculate Tmp_Den_Grid
  ***********************************************/

  dtime(&time1);


  /* AITUNE ========================== */
  int OneD_Nloop = 0;
  int ai_MaxNc = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    int Gc_AN = M2G[Mc_AN];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD_Nloop++;
      if(ai_MaxNc < GridN_Atom[Gc_AN]) {ai_MaxNc = GridN_Atom[Gc_AN];}
    }
  }
  /* ai_MaxNc is maximum of GridN_Atom[] */

  int gNthrds;
#pragma omp parallel
  {
#pragma omp single
    {
      gNthrds = omp_get_num_threads();
    }
  }

  double*** ai_tmpDG_all = (double***)SDG_Malloc(SDG_Size_From_Int(gNthrds,"gNthrds"),
                                                sizeof(double**),
                                                "ai_tmpDG_all");

  /* ========================== AITUNE */

#pragma omp parallel shared(myid,G2ID,Orbs_Grid_FNAN,List_YOUSO,time_per_atom,Tmp_Den_Grid,Orbs_Grid,COrbs_Grid,Cnt_switch,Cnt_kind,GListTAtoms2,GListTAtoms1,NumOLG,CDM,SpinP_switch,WhatSpecies,ncn,F_G2M,natn,Spe_Total_CNO,M2G) private(OMPID,Nthrds,Mc_AN,h_AN,Stime_atom,Etime_atom,Gc_AN,Cwan,NO0,Gh_AN,Mh_AN,Rnh,Hwan,NO1,spin,i,j,tmp_CDM,Nog,Nc_0,Nc_1,Nc_2,Nc_3,Nh_0,Nh_1,Nh_2,Nh_3,orbs0_0,orbs0_1,orbs0_2,orbs0_3,orbs1_0,orbs1_1,orbs1_2,orbs1_3,sum_0,sum_1,sum_2,sum_3,tmp0_0,tmp0_1,tmp0_2,tmp0_3,Nc,Nh,orbs0,orbs1,sum,tmp0)
  {

    if (List_YOUSO[7]<=0){
	  SDG_Abort("List_YOUSO[7]","non-positive orbital buffer size");
	}

    {
      size_t orb_count = (size_t)List_YOUSO[7];
      size_t tmp_cdm_rows = SDG_Product_Size((size_t)spin_components,
                                             orb_count,"tmp_CDM rows");
      double *orb_work =
	(double*)SDG_Malloc(SDG_Product_Size(10,orb_count,"orb_work"),
	                    sizeof(double),"orb_work");
      double **tmp_CDM_ptrs =
	(double**)SDG_Malloc(tmp_cdm_rows,sizeof(double*),"tmp_CDM_ptrs");
      double *tmp_CDM_data =
	(double*)SDG_Malloc(SDG_Product_Size(tmp_cdm_rows,orb_count,
	                                     "tmp_CDM_data"),
	                    sizeof(double),"tmp_CDM_data");

      orbs0   = orb_work;
      orbs1   = orbs0   + orb_count;
      orbs0_0 = orbs1   + orb_count;
      orbs0_1 = orbs0_0 + orb_count;
      orbs0_2 = orbs0_1 + orb_count;
      orbs0_3 = orbs0_2 + orb_count;
      orbs1_0 = orbs0_3 + orb_count;
      orbs1_1 = orbs1_0 + orb_count;
      orbs1_2 = orbs1_1 + orb_count;
      orbs1_3 = orbs1_2 + orb_count;

      tmp_CDM = (double***)SDG_Malloc((size_t)spin_components,
                                      sizeof(double**),"tmp_CDM");
      for (i=0; i<spin_components; i++){
	tmp_CDM[i] = &tmp_CDM_ptrs[(size_t)i*orb_count];
	for (j=0; j<List_YOUSO[7]; j++){
	  tmp_CDM[i][j] = &tmp_CDM_data[((size_t)i*orb_count+(size_t)j)
	                                *orb_count];
	}
      }

    /* get info. on OpenMP */

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();


    /* AITUNE ========================== */


    double **ai_tmpDGs;
    double *ai_tmpDGs_data;
    ai_tmpDGs = (double**)SDG_Malloc((size_t)spin_components,sizeof(double*),
                                     "ai_tmpDGs");
    ai_tmpDGs_data =
      (double*)SDG_Malloc(SDG_Product_Size((size_t)spin_components,
                                           (size_t)ai_MaxNc,"ai_tmpDGs_data"),
                          sizeof(double),"ai_tmpDGs_data");
    for (spin=0; spin<spin_components; spin++){
      ai_tmpDGs[spin] = &ai_tmpDGs_data[(size_t)spin*(size_t)ai_MaxNc];
    }
    ai_tmpDG_all[OMPID] = ai_tmpDGs;
#pragma omp barrier
    /* ==================================== AITUNE */


    /* for (Mc_AN=(OMPID+1); Mc_AN<=Matomnum; Mc_AN+=Nthrds){ AITUNE */
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      /* set data on Mc_AN */

	      Gc_AN = M2G[Mc_AN];
	      Cwan = WhatSpecies[Gc_AN];
	      NO0 = Spe_Total_CNO[Cwan];

	      {
		size_t ai_grid_bytes =
		  SDG_Product_Size((size_t)GridN_Atom[Gc_AN],sizeof(double),
		                   "ai_tmpDGs memset");
		for (spin=0; spin<spin_components; spin++){
		  memset(ai_tmpDGs[spin],0,ai_grid_bytes);
		}
	      }

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	/* set data on h_AN */

	Gh_AN = natn[Gc_AN][h_AN];
	Mh_AN = F_G2M[Gh_AN];
	Rnh = ncn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	NO1 = Spe_Total_CNO[Hwan];
        SDG_Check_Orbital_Counts(NO0,NO1,"Tmp_Den_Grid orbitals");

		/* store CDM into tmp_CDM */

		for (spin=0; spin<spin_components; spin++){
		  for (i=0; i<NO0; i++){
		    double *tmp_CDM_i = tmp_CDM[spin][i];
		    double *CDM_i = CDM[spin][Mc_AN][h_AN][i];
		    for (j=0; j<NO1; j++){
		      tmp_CDM_i[j] = CDM_i[j];
		    }
		  }
		}

	/* summation of non-zero elements */
	/* for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]; Nog++){ */
#pragma omp for
	for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]-3; Nog+=4){

	  Nc_0 = GListTAtoms1[Mc_AN][h_AN][Nog];
	  Nc_1 = GListTAtoms1[Mc_AN][h_AN][Nog+1];
	  Nc_2 = GListTAtoms1[Mc_AN][h_AN][Nog+2];
	  Nc_3 = GListTAtoms1[Mc_AN][h_AN][Nog+3];

	  Nh_0 = GListTAtoms2[Mc_AN][h_AN][Nog];
	  Nh_1 = GListTAtoms2[Mc_AN][h_AN][Nog+1];
	  Nh_2 = GListTAtoms2[Mc_AN][h_AN][Nog+2];
	  Nh_3 = GListTAtoms2[Mc_AN][h_AN][Nog+3];

		  /* Now under the orbital optimization */
		  if (Cnt_kind==0 && Cnt_switch==1){
		    for (i=0; i<NO0; i++){
		      orbs0_0[i] = COrbs_Grid[Mc_AN][i][Nc_0];
	      orbs0_1[i] = COrbs_Grid[Mc_AN][i][Nc_1];
	      orbs0_2[i] = COrbs_Grid[Mc_AN][i][Nc_2];
	      orbs0_3[i] = COrbs_Grid[Mc_AN][i][Nc_3];
	    }
	    for (j=0; j<NO1; j++){
	      orbs1_0[j] = COrbs_Grid[Mh_AN][j][Nh_0];
	      orbs1_1[j] = COrbs_Grid[Mh_AN][j][Nh_1];
	      orbs1_2[j] = COrbs_Grid[Mh_AN][j][Nh_2];
		      orbs1_3[j] = COrbs_Grid[Mh_AN][j][Nh_3];
		    }

		    for (spin=0; spin<=SpinP_switch; spin++){
		      double *ai_tmpDG = ai_tmpDGs[spin];

		      /* Tmp_Den_Grid */

		      sum_0 = 0.0;
		      sum_1 = 0.0;
		      sum_2 = 0.0;
		      sum_3 = 0.0;

		      for (i=0; i<NO0; i++){
			double *tmp_CDM_i = tmp_CDM[spin][i];

			tmp0_0 = 0.0;
			tmp0_1 = 0.0;
			tmp0_2 = 0.0;
			tmp0_3 = 0.0;

			for (j=0; j<NO1; j++){
			  double cdmij = tmp_CDM_i[j];
			  tmp0_0 += orbs1_0[j]*cdmij;
			  tmp0_1 += orbs1_1[j]*cdmij;
			  tmp0_2 += orbs1_2[j]*cdmij;
			  tmp0_3 += orbs1_3[j]*cdmij;
			}

			sum_0 += orbs0_0[i]*tmp0_0;
			sum_1 += orbs0_1[i]*tmp0_1;
			sum_2 += orbs0_2[i]*tmp0_2;
			sum_3 += orbs0_3[i]*tmp0_3;
		      }

		      ai_tmpDG[Nc_0] += sum_0;
		      ai_tmpDG[Nc_1] += sum_1;
		      ai_tmpDG[Nc_2] += sum_2;
		      ai_tmpDG[Nc_3] += sum_3;
		    } /* spin */
		  }
		  /* else if ! "now under the orbital optimization" */
		  else{
		    Type_Orbs_Grid *orbs0_0g = Orbs_Grid[Mc_AN][Nc_0];
		    Type_Orbs_Grid *orbs0_1g = Orbs_Grid[Mc_AN][Nc_1];
		    Type_Orbs_Grid *orbs0_2g = Orbs_Grid[Mc_AN][Nc_2];
		    Type_Orbs_Grid *orbs0_3g = Orbs_Grid[Mc_AN][Nc_3];
		    Type_Orbs_Grid *orbs1_0g;
		    Type_Orbs_Grid *orbs1_1g;
		    Type_Orbs_Grid *orbs1_2g;
		    Type_Orbs_Grid *orbs1_3g;

	            if (G2ID[Gh_AN]==myid){
		      orbs1_0g = Orbs_Grid[Mh_AN][Nh_0];
		      orbs1_1g = Orbs_Grid[Mh_AN][Nh_1];
		      orbs1_2g = Orbs_Grid[Mh_AN][Nh_2];
		      orbs1_3g = Orbs_Grid[Mh_AN][Nh_3];
		    }
	            else{
		      orbs1_0g = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog  ];
		      orbs1_1g = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog+1];
		      orbs1_2g = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog+2];
		      orbs1_3g = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog+3];
		    }

		    for (spin=0; spin<=SpinP_switch; spin++){
		      double *ai_tmpDG = ai_tmpDGs[spin];

		      sum_0 = 0.0;
		      sum_1 = 0.0;
		      sum_2 = 0.0;
		      sum_3 = 0.0;

		      for (i=0; i<NO0; i++){
			double *tmp_CDM_i = tmp_CDM[spin][i];

			tmp0_0 = 0.0;
			tmp0_1 = 0.0;
			tmp0_2 = 0.0;
			tmp0_3 = 0.0;

			for (j=0; j<NO1; j++){
			  double cdmij = tmp_CDM_i[j];
			  tmp0_0 += (double)orbs1_0g[j]*cdmij;
			  tmp0_1 += (double)orbs1_1g[j]*cdmij;
			  tmp0_2 += (double)orbs1_2g[j]*cdmij;
			  tmp0_3 += (double)orbs1_3g[j]*cdmij;
			}

			sum_0 += (double)orbs0_0g[i]*tmp0_0;
			sum_1 += (double)orbs0_1g[i]*tmp0_1;
			sum_2 += (double)orbs0_2g[i]*tmp0_2;
			sum_3 += (double)orbs0_3g[i]*tmp0_3;
		      }

		      ai_tmpDG[Nc_0] += sum_0;
		      ai_tmpDG[Nc_1] += sum_1;
		      ai_tmpDG[Nc_2] += sum_2;
		      ai_tmpDG[Nc_3] += sum_3;
		    } /* spin */
		  }
		} /* Nog */

#pragma omp for
	for (Nog = NumOLG[Mc_AN][h_AN] - (NumOLG[Mc_AN][h_AN] % 4); Nog<NumOLG[Mc_AN][h_AN]; Nog++){
	  /*for (; Nog<NumOLG[Mc_AN][h_AN]; Nog++){*/

	  Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  Nh = GListTAtoms2[Mc_AN][h_AN][Nog];


		  if (Cnt_kind==0 && Cnt_switch==1){
		    for (i=0; i<NO0; i++){
		      orbs0[i] = COrbs_Grid[Mc_AN][i][Nc];
	    }
		    for (j=0; j<NO1; j++){
		      orbs1[j] = COrbs_Grid[Mh_AN][j][Nh];
		    }

		    for (spin=0; spin<=SpinP_switch; spin++){
		      double *ai_tmpDG = ai_tmpDGs[spin];

		      sum = 0.0;
		      for (i=0; i<NO0; i++){
			double *tmp_CDM_i = tmp_CDM[spin][i];
			tmp0 = 0.0;
			for (j=0; j<NO1; j++){
			  tmp0 += orbs1[j]*tmp_CDM_i[j];
			}
			sum += orbs0[i]*tmp0;
		      }

		      ai_tmpDG[Nc] += sum;
		    }
		  }
		  else{
		    Type_Orbs_Grid *orbs0g = Orbs_Grid[Mc_AN][Nc];
		    Type_Orbs_Grid *orbs1g;

		    if (G2ID[Gh_AN]==myid){
		      orbs1g = Orbs_Grid[Mh_AN][Nh];
		    }
		    else{
		      orbs1g = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog];
		    }

		    for (spin=0; spin<=SpinP_switch; spin++){
		      double *ai_tmpDG = ai_tmpDGs[spin];

		      sum = 0.0;
		      for (i=0; i<NO0; i++){
			double *tmp_CDM_i = tmp_CDM[spin][i];
			tmp0 = 0.0;
			for (j=0; j<NO1; j++){
			  tmp0 += (double)orbs1g[j]*tmp_CDM_i[j];
			}
			sum += (double)orbs0g[i]*tmp0;
		      }

		      ai_tmpDG[Nc] += sum;
		    }
		  }

		} /* Nog */

      } /* h_AN */

	      /* AITUNE   merge temporary buffer for all omp threads */
	      for (spin=0; spin<=SpinP_switch; spin++){
		double *tmp_den_grid = Tmp_Den_Grid[spin][Mc_AN];
		int Nc;
	#pragma omp for
		for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
		  double sum = 0.0;
		  int th;
		  for(th = 0; th < Nthrds; th++){
		    double *ai_tmpDG_th = ai_tmpDG_all[th][spin];
		    sum += ai_tmpDG_th[Nc];
		  }
		  tmp_den_grid[Nc] += sum;
		}
	      }

#pragma omp single
      {
        dtime(&Etime_atom);
        time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      }

	    } /* Mc_AN */

	    /* freeing of arrays */

      free(ai_tmpDGs_data);
      free(ai_tmpDGs);
      free(tmp_CDM_data);
      free(tmp_CDM_ptrs);
      free(tmp_CDM);
      free(orb_work);
    }

#pragma omp flush(Tmp_Den_Grid)

  } /* #pragma omp parallel */

  free(ai_tmpDG_all);

  dtime(&time2);
  if(myid==0 && measure_time){
    printf("Time for Part1=%18.5f\n",(time2-time1));fflush(stdout);
  }

  /******************************************************
      MPI communication from the partitions A to B
  ******************************************************/

  /* copy Tmp_Den_Grid to Den_Snd_Grid_A2B */

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_A2B[ID] = 0;

  N2D = SDG_Grid2D_Size();

  if (SpinP_switch==0){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      for (AN=0; AN<GridN_Atom[Gc_AN]; AN++){
	GN = GridListAtom[Mc_AN][AN];
	n2D = SDG_Grid2D_Index_From_GN(GN,N2D);
	ID = (int)(n2D*(unsigned long long int)numprocs/N2D);
	Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]] =
	  Tmp_Den_Grid[0][Mc_AN][AN];
	Num_Snd_Grid_A2B[ID]++;
      }
    }
  }
  else if (SpinP_switch==1){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      for (AN=0; AN<GridN_Atom[Gc_AN]; AN++){
	GN = GridListAtom[Mc_AN][AN];
	n2D = SDG_Grid2D_Index_From_GN(GN,N2D);
	ID = (int)(n2D*(unsigned long long int)numprocs/N2D);
	LN = Num_Snd_Grid_A2B[ID]*2;
	Den_Snd_Grid_A2B[ID][LN  ] = Tmp_Den_Grid[0][Mc_AN][AN];
	Den_Snd_Grid_A2B[ID][LN+1] = Tmp_Den_Grid[1][Mc_AN][AN];
	Num_Snd_Grid_A2B[ID]++;
      }
    }
  }
  else if (SpinP_switch==3){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      for (AN=0; AN<GridN_Atom[Gc_AN]; AN++){
	GN = GridListAtom[Mc_AN][AN];
	n2D = SDG_Grid2D_Index_From_GN(GN,N2D);
	ID = (int)(n2D*(unsigned long long int)numprocs/N2D);
	LN = Num_Snd_Grid_A2B[ID]*4;
	Den_Snd_Grid_A2B[ID][LN  ] = Tmp_Den_Grid[0][Mc_AN][AN];
	Den_Snd_Grid_A2B[ID][LN+1] = Tmp_Den_Grid[1][Mc_AN][AN];
	Den_Snd_Grid_A2B[ID][LN+2] = Tmp_Den_Grid[2][Mc_AN][AN];
	Den_Snd_Grid_A2B[ID][LN+3] = Tmp_Den_Grid[3][Mc_AN][AN];
	Num_Snd_Grid_A2B[ID]++;
      }
    }
  }

  /* MPI: A to B */

  request_send = (MPI_Request*)SDG_Malloc(SDG_Size_From_Int(NN_A2B_S,"NN_A2B_S"),
                                          sizeof(MPI_Request),
                                          "request_send A2B");
  request_recv = (MPI_Request*)SDG_Malloc(SDG_Size_From_Int(NN_A2B_R,"NN_A2B_R"),
                                          sizeof(MPI_Request),
                                          "request_recv A2B");
  stat_send = (MPI_Status*)SDG_Malloc(SDG_Size_From_Int(NN_A2B_S,"NN_A2B_S"),
                                      sizeof(MPI_Status),
                                      "stat_send A2B");
  stat_recv = (MPI_Status*)SDG_Malloc(SDG_Size_From_Int(NN_A2B_R,"NN_A2B_R"),
                                      sizeof(MPI_Status),
                                      "stat_recv A2B");

  NN_S = 0;
  NN_R = 0;

  tag = 999;
  for (ID=1; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (Num_Snd_Grid_A2B[IDS]!=0){
      int count = SDG_Checked_MPI_Count(Num_Snd_Grid_A2B[IDS],spin_components,
                                        "MPI_Isend A2B");
      MPI_Isend( &Den_Snd_Grid_A2B[IDS][0], count,
		 MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }

    if (Num_Rcv_Grid_A2B[IDR]!=0){
      int count = SDG_Checked_MPI_Count(Num_Rcv_Grid_A2B[IDR],spin_components,
                                        "MPI_Irecv A2B");
      MPI_Irecv( &Den_Rcv_Grid_A2B[IDR][0], count,
  	         MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }
  }

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* for myid */
  {
    int self_count = SDG_Checked_MPI_Count(Num_Rcv_Grid_A2B[myid],
                                           spin_components,"self copy A2B");
    memcpy(Den_Rcv_Grid_A2B[myid],Den_Snd_Grid_A2B[myid],
           SDG_Product_Size((size_t)self_count,sizeof(double),
                            "self copy A2B"));
  }

  /******************************************************
   superposition of rho_i to calculate charge density
   in the partition B.
  ******************************************************/

  /* initialize arrays */

  {
    size_t density_b_bytes =
      SDG_Product_Size((size_t)My_NumGridB_AB,sizeof(double),
                       "Density_Grid_B0 memset");
  for (spin=0; spin<spin_components; spin++){
      memset(Density_Grid_B0[spin],0,density_b_bytes);
    }
  }

  /* superposition of densities rho_i */

  if (SpinP_switch==0){
    for (ID=0; ID<numprocs; ID++){
      for (LN=0; LN<Num_Rcv_Grid_A2B[ID]; LN++){
	BN    = Index_Rcv_Grid_A2B[ID][3*LN+0];
	GRc   = Index_Rcv_Grid_A2B[ID][3*LN+2];
	if (Solver!=4 || (Solver==4 && atv_ijk[GRc][1]==0 )){
	  Density_Grid_B0[0][BN] += Den_Rcv_Grid_A2B[ID][LN];
	}
      }
    }
  }
  else if (SpinP_switch==1){
    for (ID=0; ID<numprocs; ID++){
      for (LN=0; LN<Num_Rcv_Grid_A2B[ID]; LN++){
	BN    = Index_Rcv_Grid_A2B[ID][3*LN+0];
	GRc   = Index_Rcv_Grid_A2B[ID][3*LN+2];
	if (Solver!=4 || (Solver==4 && atv_ijk[GRc][1]==0 )){
	  i = LN*2;
	  Density_Grid_B0[0][BN] += Den_Rcv_Grid_A2B[ID][i  ];
	  Density_Grid_B0[1][BN] += Den_Rcv_Grid_A2B[ID][i+1];
	}
      }
    }
  }
  else if (SpinP_switch==3){
    for (ID=0; ID<numprocs; ID++){
      for (LN=0; LN<Num_Rcv_Grid_A2B[ID]; LN++){
	BN    = Index_Rcv_Grid_A2B[ID][3*LN+0];
	GRc   = Index_Rcv_Grid_A2B[ID][3*LN+2];
	if (Solver!=4 || (Solver==4 && atv_ijk[GRc][1]==0 )){
	  i = LN*4;
	  Density_Grid_B0[0][BN] += Den_Rcv_Grid_A2B[ID][i  ];
	  Density_Grid_B0[1][BN] += Den_Rcv_Grid_A2B[ID][i+1];
	  Density_Grid_B0[2][BN] += Den_Rcv_Grid_A2B[ID][i+2];
	  Density_Grid_B0[3][BN] += Den_Rcv_Grid_A2B[ID][i+3];
	}
      }
    }
  }

  /****************************************************
   Conjugate complex of Density_Grid[3][MN] due to
   difference in the definition between density matrix
   and charge density
  ****************************************************/

  if (SpinP_switch==3){

    for (BN=0; BN<My_NumGridB_AB; BN++){
      Density_Grid_B0[3][BN] = -Density_Grid_B0[3][BN];
    }
  }

  /******************************************************
             MPI: from the partitions B to D
  ******************************************************/

  Density_Grid_Copy_B2D(Density_Grid_B0);

  /* freeing of arrays */

  for (i=0; i<spin_components; i++){
    free(Tmp_Den_Grid_Data[i]);
    free(Tmp_Den_Grid[i]);
  }
  free(Tmp_Den_Grid_Data);
  free(Tmp_Den_Grid);

  free(Den_Snd_Grid_A2B_Data);
  free(Den_Snd_Grid_A2B);

  free(Den_Rcv_Grid_A2B_Data);
  free(Den_Rcv_Grid_A2B);

  /* elapsed time */
  dtime(&TEtime);
  time0 = TEtime - TStime;
  if(myid==0 && measure_time) printf("time0=%18.5f\n",time0);

  return time0;
}



void Data_Grid_Copy_B2C_2(double **data_B, double **data_C)
{
  static int firsttime=1;
  int CN,BN,LN,spin,i,gp,NN_S,NN_R,spin_components;
  double *Work_Array_Snd_Grid_B2C;
  double *Work_Array_Rcv_Grid_B2C;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (SpinP_switch!=0 && SpinP_switch!=1 && SpinP_switch!=3){
    SDG_Abort("Data_Grid_Copy_B2C_2","unsupported spin mode");
  }
  spin_components = SpinP_switch + 1;

  /* allocation of arrays */

  Work_Array_Snd_Grid_B2C =
    (double*)SDG_Malloc((size_t)SDG_Checked_MPI_Count(GP_B2C_S[NN_B2C_S],
                                                       spin_components,
                                                       "Work_Array_Snd_Grid_B2C"),
                        sizeof(double),"Work_Array_Snd_Grid_B2C");
  Work_Array_Rcv_Grid_B2C =
    (double*)SDG_Malloc((size_t)SDG_Checked_MPI_Count(GP_B2C_R[NN_B2C_R],
                                                       spin_components,
                                                       "Work_Array_Rcv_Grid_B2C"),
                        sizeof(double),"Work_Array_Rcv_Grid_B2C");

  if (firsttime==1){
    PrintMemory("Data_Grid_Copy_B2C_2: Work_Array_Snd_Grid_B2C",
		(long int)(sizeof(double)*SDG_Checked_MPI_Count(GP_B2C_S[NN_B2C_S],
                                                                 spin_components,
                                                                 "PrintMemory B2C_2 send")), NULL);
    PrintMemory("Data_Grid_Copy_B2C_2: Work_Array_Rcv_Grid_B2C",
		(long int)(sizeof(double)*SDG_Checked_MPI_Count(GP_B2C_R[NN_B2C_R],
                                                                 spin_components,
                                                                 "PrintMemory B2C_2 recv")), NULL);
    firsttime = 0;
  }

  /******************************************************
             MPI: from the partitions B to C
  ******************************************************/

  request_send = (MPI_Request*)SDG_Malloc(SDG_Size_From_Int(NN_B2C_S,"NN_B2C_S"),
                                          sizeof(MPI_Request),
                                          "request_send B2C_2");
  request_recv = (MPI_Request*)SDG_Malloc(SDG_Size_From_Int(NN_B2C_R,"NN_B2C_R"),
                                          sizeof(MPI_Request),
                                          "request_recv B2C_2");
  stat_send = (MPI_Status*)SDG_Malloc(SDG_Size_From_Int(NN_B2C_S,"NN_B2C_S"),
                                      sizeof(MPI_Status),
                                      "stat_send B2C_2");
  stat_recv = (MPI_Status*)SDG_Malloc(SDG_Size_From_Int(NN_B2C_R,"NN_B2C_R"),
                                      sizeof(MPI_Status),
                                      "stat_recv B2C_2");

  NN_S = 0;
  NN_R = 0;

  /* MPI_Irecv */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];
    gp = GP_B2C_R[ID];

    if (IDR!=myid){
      int count = SDG_Checked_MPI_Count(Num_Rcv_Grid_B2C[IDR],spin_components,
                                        "MPI_Irecv B2C_2");
      MPI_Irecv( &Work_Array_Rcv_Grid_B2C[spin_components*gp], count,
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }

  }

  /* MPI_Isend */

  for (ID=0; ID<NN_B2C_S; ID++){

    IDS = ID_NN_B2C_S[ID];
    gp = GP_B2C_S[ID];

    /* copy Density_Grid_B to Work_Array_Snd_Grid_B2C */

    if (SpinP_switch==0){
      for (LN=0; LN<Num_Snd_Grid_B2C[IDS]; LN++){
	BN = Index_Snd_Grid_B2C[IDS][LN];
	Work_Array_Snd_Grid_B2C[gp+LN] = data_B[0][BN];
      }
    }
    else if (SpinP_switch==1){
      for (LN=0; LN<Num_Snd_Grid_B2C[IDS]; LN++){
	BN = Index_Snd_Grid_B2C[IDS][LN];
	i = 2*gp + 2*LN;
	Work_Array_Snd_Grid_B2C[i  ] = data_B[0][BN];
	Work_Array_Snd_Grid_B2C[i+1] = data_B[1][BN];
      }
    }
    else if (SpinP_switch==3){
      for (LN=0; LN<Num_Snd_Grid_B2C[IDS]; LN++){
	BN = Index_Snd_Grid_B2C[IDS][LN];
	i = 4*gp + 4*LN;
	Work_Array_Snd_Grid_B2C[i  ] = data_B[0][BN];
	Work_Array_Snd_Grid_B2C[i+1] = data_B[1][BN];
	Work_Array_Snd_Grid_B2C[i+2] = data_B[2][BN];
	Work_Array_Snd_Grid_B2C[i+3] = data_B[3][BN];
      }
    }

    if (IDS!=myid){
      int count = SDG_Checked_MPI_Count(Num_Snd_Grid_B2C[IDS],spin_components,
                                        "MPI_Isend B2C_2");
      MPI_Isend( &Work_Array_Snd_Grid_B2C[spin_components*gp], count,
			 MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* copy Work_Array_Rcv_Grid_B2C to data_C */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];

    if (IDR==myid){

      gp = GP_B2C_S[SDG_Find_Peer_Slot(ID_NN_B2C_S,NN_B2C_S,IDR,
                                       "Data_Grid_Copy_B2C_2 self")];

      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){

	CN = Index_Rcv_Grid_B2C[IDR][LN];

	if (SpinP_switch==0){
	  data_C[0][CN] = Work_Array_Snd_Grid_B2C[gp+LN];
	}
	else if (SpinP_switch==1){
	  data_C[0][CN] = Work_Array_Snd_Grid_B2C[2*gp+2*LN+0];
	  data_C[1][CN] = Work_Array_Snd_Grid_B2C[2*gp+2*LN+1];
	}
	else if (SpinP_switch==3){
	  data_C[0][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+0];
	  data_C[1][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+1];
	  data_C[2][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+2];
	  data_C[3][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+3];
	}
      } /* LN */

    }
    else {

      gp = GP_B2C_R[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){
	CN = Index_Rcv_Grid_B2C[IDR][LN];

	if (SpinP_switch==0){
	  data_C[0][CN] = Work_Array_Rcv_Grid_B2C[gp+LN];
	}
	else if (SpinP_switch==1){
	  data_C[0][CN] = Work_Array_Rcv_Grid_B2C[2*gp+2*LN+0];
	  data_C[1][CN] = Work_Array_Rcv_Grid_B2C[2*gp+2*LN+1];
	}
	else if (SpinP_switch==3){
	  data_C[0][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+0];
	  data_C[1][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+1];
	  data_C[2][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+2];
	  data_C[3][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+3];
	}
      }
    }
  }

  /* if (SpinP_switch==0),
     copy data_B[0] to data_B[1]
     copy data_C[0] to data_C[1]
  */

  if (SpinP_switch==0){
    memcpy(data_B[1],data_B[0],
           SDG_Product_Size((size_t)My_NumGridB_AB,sizeof(double),
                            "Data_Grid_Copy_B2C_2 data_B copy"));
    memcpy(data_C[1],data_C[0],
           SDG_Product_Size((size_t)My_NumGridC,sizeof(double),
                            "Data_Grid_Copy_B2C_2 data_C copy"));
  }

  /* freeing of arrays */
  free(Work_Array_Snd_Grid_B2C);
  free(Work_Array_Rcv_Grid_B2C);
}



void Data_Grid_Copy_B2C_1(double *data_B, double *data_C)
{
  static int firsttime=1;
  int CN,BN,LN,spin,i,gp,NN_S,NN_R;
  double *Work_Array_Snd_Grid_B2C;
  double *Work_Array_Rcv_Grid_B2C;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */

  Work_Array_Snd_Grid_B2C =
    (double*)SDG_Malloc((size_t)SDG_Checked_MPI_Count(GP_B2C_S[NN_B2C_S],1,
                                                       "Work_Array_Snd_Grid_B2C_1"),
                        sizeof(double),"Work_Array_Snd_Grid_B2C_1");
  Work_Array_Rcv_Grid_B2C =
    (double*)SDG_Malloc((size_t)SDG_Checked_MPI_Count(GP_B2C_R[NN_B2C_R],1,
                                                       "Work_Array_Rcv_Grid_B2C_1"),
                        sizeof(double),"Work_Array_Rcv_Grid_B2C_1");

  if (firsttime==1){
    PrintMemory("Data_Grid_Copy_B2C_1: Work_Array_Snd_Grid_B2C",
		(long int)(sizeof(double)*SDG_Checked_MPI_Count(GP_B2C_S[NN_B2C_S],1,
                                                                 "PrintMemory B2C_1 send")), NULL);
    PrintMemory("Data_Grid_Copy_B2C_1: Work_Array_Rcv_Grid_B2C",
		(long int)(sizeof(double)*SDG_Checked_MPI_Count(GP_B2C_R[NN_B2C_R],1,
                                                                 "PrintMemory B2C_1 recv")), NULL);
    firsttime = 0;
  }

  /******************************************************
             MPI: from the partitions B to C
  ******************************************************/

  request_send = (MPI_Request*)SDG_Malloc(SDG_Size_From_Int(NN_B2C_S,"NN_B2C_S"),
                                          sizeof(MPI_Request),
                                          "request_send B2C_1");
  request_recv = (MPI_Request*)SDG_Malloc(SDG_Size_From_Int(NN_B2C_R,"NN_B2C_R"),
                                          sizeof(MPI_Request),
                                          "request_recv B2C_1");
  stat_send = (MPI_Status*)SDG_Malloc(SDG_Size_From_Int(NN_B2C_S,"NN_B2C_S"),
                                      sizeof(MPI_Status),
                                      "stat_send B2C_1");
  stat_recv = (MPI_Status*)SDG_Malloc(SDG_Size_From_Int(NN_B2C_R,"NN_B2C_R"),
                                      sizeof(MPI_Status),
                                      "stat_recv B2C_1");

  NN_S = 0;
  NN_R = 0;

  /* MPI_Irecv */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];
    gp = GP_B2C_R[ID];

    if (IDR!=myid){
      int count = SDG_Checked_MPI_Count(Num_Rcv_Grid_B2C[IDR],1,
                                        "MPI_Irecv B2C_1");
      MPI_Irecv( &Work_Array_Rcv_Grid_B2C[gp], count,
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }
  }

  /* MPI_Isend */

  for (ID=0; ID<NN_B2C_S; ID++){

    IDS = ID_NN_B2C_S[ID];
    gp = GP_B2C_S[ID];

    /* copy Density_Grid_B to Work_Array_Snd_Grid_B2C */

    for (LN=0; LN<Num_Snd_Grid_B2C[IDS]; LN++){
      BN = Index_Snd_Grid_B2C[IDS][LN];
      Work_Array_Snd_Grid_B2C[gp+LN] = data_B[BN];
    }

    if (IDS!=myid){
      int count = SDG_Checked_MPI_Count(Num_Snd_Grid_B2C[IDS],1,
                                        "MPI_Isend B2C_1");
      MPI_Isend( &Work_Array_Snd_Grid_B2C[gp], count,
			 MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* copy Work_Array_Rcv_Grid_B2C to data_C */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];

    if (IDR==myid){
      gp = GP_B2C_S[SDG_Find_Peer_Slot(ID_NN_B2C_S,NN_B2C_S,IDR,
                                       "Data_Grid_Copy_B2C_1 self")];
      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){
	CN = Index_Rcv_Grid_B2C[IDR][LN];
	data_C[CN] = Work_Array_Snd_Grid_B2C[gp+LN];
      }
    }
    else{

      gp = GP_B2C_R[ID];
      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){
	CN = Index_Rcv_Grid_B2C[IDR][LN];
	data_C[CN] = Work_Array_Rcv_Grid_B2C[gp+LN];
      }
    }
  }

  /* freeing of arrays */
  free(Work_Array_Snd_Grid_B2C);
  free(Work_Array_Rcv_Grid_B2C);
}





void Density_Grid_Copy_B2D(double **Density_Grid_B0)
{
  static int firsttime=1;
  int DN,BN,LN,spin,i,gp,NN_S,NN_R,spin_components;
  double *Work_Array_Snd_Grid_B2D;
  double *Work_Array_Rcv_Grid_B2D;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (SpinP_switch!=0 && SpinP_switch!=1 && SpinP_switch!=3){
    SDG_Abort("Density_Grid_Copy_B2D","unsupported spin mode");
  }
  spin_components = SpinP_switch + 1;

  /* allocation of arrays */

  Work_Array_Snd_Grid_B2D =
    (double*)SDG_Malloc((size_t)SDG_Checked_MPI_Count(GP_B2D_S[NN_B2D_S],
                                                       spin_components,
                                                       "Work_Array_Snd_Grid_B2D"),
                        sizeof(double),"Work_Array_Snd_Grid_B2D");
  Work_Array_Rcv_Grid_B2D =
    (double*)SDG_Malloc((size_t)SDG_Checked_MPI_Count(GP_B2D_R[NN_B2D_R],
                                                       spin_components,
                                                       "Work_Array_Rcv_Grid_B2D"),
                        sizeof(double),"Work_Array_Rcv_Grid_B2D");

  if (firsttime==1){
    PrintMemory("Set_Density_Grid: Work_Array_Snd_Grid_B2D",
		(long int)(sizeof(double)*SDG_Checked_MPI_Count(GP_B2D_S[NN_B2D_S],
                                                                 spin_components,
                                                                 "PrintMemory B2D send")), NULL);
    PrintMemory("Set_Density_Grid: Work_Array_Rcv_Grid_B2D",
		(long int)(sizeof(double)*SDG_Checked_MPI_Count(GP_B2D_R[NN_B2D_R],
                                                                 spin_components,
                                                                 "PrintMemory B2D recv")), NULL);
    firsttime = 0;
  }

  /******************************************************
             MPI: from the partitions B to D
  ******************************************************/

  request_send = (MPI_Request*)SDG_Malloc(SDG_Size_From_Int(NN_B2D_S,"NN_B2D_S"),
                                          sizeof(MPI_Request),
                                          "request_send B2D");
  request_recv = (MPI_Request*)SDG_Malloc(SDG_Size_From_Int(NN_B2D_R,"NN_B2D_R"),
                                          sizeof(MPI_Request),
                                          "request_recv B2D");
  stat_send = (MPI_Status*)SDG_Malloc(SDG_Size_From_Int(NN_B2D_S,"NN_B2D_S"),
                                      sizeof(MPI_Status),
                                      "stat_send B2D");
  stat_recv = (MPI_Status*)SDG_Malloc(SDG_Size_From_Int(NN_B2D_R,"NN_B2D_R"),
                                      sizeof(MPI_Status),
                                      "stat_recv B2D");

  NN_S = 0;
  NN_R = 0;

  /* MPI_Irecv */

  for (ID=0; ID<NN_B2D_R; ID++){

    IDR = ID_NN_B2D_R[ID];
    gp = GP_B2D_R[ID];

    if (IDR!=myid){
      int count = SDG_Checked_MPI_Count(Num_Rcv_Grid_B2D[IDR],spin_components,
                                        "MPI_Irecv B2D");
      MPI_Irecv( &Work_Array_Rcv_Grid_B2D[spin_components*gp], count,
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }
  }

  /* MPI_Isend */

  for (ID=0; ID<NN_B2D_S; ID++){

    IDS = ID_NN_B2D_S[ID];
    gp = GP_B2D_S[ID];

    /* copy Density_Grid_B0 to Work_Array_Snd_Grid_B2D */

    if (SpinP_switch==0){
      for (LN=0; LN<Num_Snd_Grid_B2D[IDS]; LN++){
	BN = Index_Snd_Grid_B2D[IDS][LN];
	Work_Array_Snd_Grid_B2D[gp+LN] = Density_Grid_B0[0][BN];
      }
    }
    else if (SpinP_switch==1){
      for (LN=0; LN<Num_Snd_Grid_B2D[IDS]; LN++){
	BN = Index_Snd_Grid_B2D[IDS][LN];
	i = 2*gp + 2*LN;
	Work_Array_Snd_Grid_B2D[i  ] = Density_Grid_B0[0][BN];
	Work_Array_Snd_Grid_B2D[i+1] = Density_Grid_B0[1][BN];
      }
    }
    else if (SpinP_switch==3){
      for (LN=0; LN<Num_Snd_Grid_B2D[IDS]; LN++){
	BN = Index_Snd_Grid_B2D[IDS][LN];
	i = 4*gp + 4*LN;
	Work_Array_Snd_Grid_B2D[i  ] = Density_Grid_B0[0][BN];
	Work_Array_Snd_Grid_B2D[i+1] = Density_Grid_B0[1][BN];
	Work_Array_Snd_Grid_B2D[i+2] = Density_Grid_B0[2][BN];
	Work_Array_Snd_Grid_B2D[i+3] = Density_Grid_B0[3][BN];
      }
    }

    if (IDS!=myid){
      int count = SDG_Checked_MPI_Count(Num_Snd_Grid_B2D[IDS],spin_components,
                                        "MPI_Isend B2D");
      MPI_Isend( &Work_Array_Snd_Grid_B2D[spin_components*gp], count,
			 MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* copy Work_Array_Rcv_Grid_B2D to Density_Grid_D */

  for (ID=0; ID<NN_B2D_R; ID++){

    IDR = ID_NN_B2D_R[ID];

    if (IDR==myid){

      gp = GP_B2D_S[SDG_Find_Peer_Slot(ID_NN_B2D_S,NN_B2D_S,IDR,
                                       "Density_Grid_Copy_B2D self")];

      for (LN=0; LN<Num_Rcv_Grid_B2D[IDR]; LN++){

	DN = Index_Rcv_Grid_B2D[IDR][LN];

	if (SpinP_switch==0){
	  Density_Grid_D[0][DN] = Work_Array_Snd_Grid_B2D[gp+LN];
	}
	else if (SpinP_switch==1){
	  Density_Grid_D[0][DN] = Work_Array_Snd_Grid_B2D[2*gp+2*LN+0];
	  Density_Grid_D[1][DN] = Work_Array_Snd_Grid_B2D[2*gp+2*LN+1];
	}
	else if (SpinP_switch==3){
	  Density_Grid_D[0][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+0];
	  Density_Grid_D[1][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+1];
	  Density_Grid_D[2][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+2];
	  Density_Grid_D[3][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+3];
	}
      } /* LN */

    }

    else{

      gp = GP_B2D_R[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2D[IDR]; LN++){

	DN = Index_Rcv_Grid_B2D[IDR][LN];

	if (SpinP_switch==0){
	  Density_Grid_D[0][DN] = Work_Array_Rcv_Grid_B2D[gp+LN];
	}
	else if (SpinP_switch==1){
	  Density_Grid_D[0][DN] = Work_Array_Rcv_Grid_B2D[2*gp+2*LN+0];
	  Density_Grid_D[1][DN] = Work_Array_Rcv_Grid_B2D[2*gp+2*LN+1];
	}
	else if (SpinP_switch==3){
	  Density_Grid_D[0][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+0];
	  Density_Grid_D[1][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+1];
	  Density_Grid_D[2][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+2];
	  Density_Grid_D[3][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+3];
	}
      }

    }
  }

  /* if (SpinP_switch==0), copy Density_Grid to Density_Grid? */

  if (SpinP_switch==0){
    memcpy(Density_Grid_B0[1],Density_Grid_B0[0],
           SDG_Product_Size((size_t)My_NumGridB_AB,sizeof(double),
                            "Density_Grid_Copy_B2D B copy"));
    memcpy(Density_Grid_D[1],Density_Grid_D[0],
           SDG_Product_Size((size_t)My_NumGridD,sizeof(double),
                            "Density_Grid_Copy_B2D D copy"));
  }

  /* freeing of arrays */
  free(Work_Array_Snd_Grid_B2D);
  free(Work_Array_Rcv_Grid_B2D);
}


void diagonalize_nc_density(double **Density_Grid_B0)
{
  int BN,DN,Mc_AN,Gc_AN,Nog,GRc;
  double Re11,Re22,Re12,Im12;
  double phi[2],theta[2],sit,cot,sip,cop;
  double d1,d2,d3,x,y,z,Cxyz[4];
  double Nup[2],Ndown[2];
  /* for OpenMP */
  int OMPID,Nthrds;

  /************************************
     Density_Grid in the partition B
  ************************************/

#pragma omp parallel shared(Density_Grid_B0,My_NumGridB_AB) private(OMPID,Nthrds,BN,Re11,Re22,Re12,Im12,Nup,Ndown,theta,phi) default(none)
  {

    /* get info. on OpenMP */

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();

    for (BN=OMPID; BN<My_NumGridB_AB; BN+=Nthrds){

      Re11 = Density_Grid_B0[0][BN];
      Re22 = Density_Grid_B0[1][BN];
      Re12 = Density_Grid_B0[2][BN];
      Im12 = Density_Grid_B0[3][BN];

      EulerAngle_Spin( 1, Re11, Re22, Re12, Im12, Re12, -Im12, Nup, Ndown, theta, phi );

      /*
      if (    1.0e-7<fabs(Re11-Nup[0])
	   || 1.0e-7<fabs(Re22-Ndown[0])
	   || 1.0e-7<fabs(Re12-theta[0])
	   || 1.0e-7<fabs(Im12-phi[0]) ){

        printf("ZZZ1 BN=%2d Re11=%15.12f Re22=%15.12f Re12=%15.12f Im12=%15.12f\n",BN,Re11,Re22,Re12,Im12);
        printf("ZZZ2 BN=%2d Nup =%15.12f Ndn =%15.12f thet=%15.12f phi =%15.12f\n",BN,Nup[0],Ndown[0],theta[0],phi[0]);
      }
      */

      Density_Grid_B0[0][BN] = Nup[0];
      Density_Grid_B0[1][BN] = Ndown[0];
      Density_Grid_B0[2][BN] = theta[0];
      Density_Grid_B0[3][BN] = phi[0];
    }

#pragma omp flush(Density_Grid_B)

  } /* #pragma omp parallel */

  /************************************
     Density_Grid in the partition D
  ************************************/

#pragma omp parallel shared(Density_Grid_D,My_NumGridD) private(OMPID,Nthrds,DN,Re11,Re22,Re12,Im12,Nup,Ndown,theta,phi) default(none)
  {

    /* get info. on OpenMP */

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();

    for (DN=OMPID; DN<My_NumGridD; DN+=Nthrds){

      Re11 = Density_Grid_D[0][DN];
      Re22 = Density_Grid_D[1][DN];
      Re12 = Density_Grid_D[2][DN];
      Im12 = Density_Grid_D[3][DN];

      EulerAngle_Spin( 1, Re11, Re22, Re12, Im12, Re12, -Im12, Nup, Ndown, theta, phi );

      Density_Grid_D[0][DN] = Nup[0];
      Density_Grid_D[1][DN] = Ndown[0];
      Density_Grid_D[2][DN] = theta[0];
      Density_Grid_D[3][DN] = phi[0];
    }

#pragma omp flush(Density_Grid_D)

  } /* #pragma omp parallel */

}
