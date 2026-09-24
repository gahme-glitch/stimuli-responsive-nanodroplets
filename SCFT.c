#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

#include "parameters.h"

// Function
double funGaussDev();
void funPDESolver(int start, int stop, double *exparg, double *qr, double *wfield, fftw_plan forwardplan, fftw_plan reverseplan, int seedBool);
void funPDESolverBrush(int start, int stop, double *exparg, double *qr, double *wfield, fftw_plan forwardplan, fftw_plan reverseplan, double *mask);
void funPDESolver_periodic(fftw_complex *expfield, double *exparg, fftw_plan forwardplan, fftw_plan reverseplan, fftw_complex *indata, fftw_complex *outdata);
void funPDESolver_DirichletBC(fftw_complex *expfield, double *exparg, fftw_plan forwardplan, fftw_plan reverseplan, fftw_complex *indata, fftw_complex *indata_expanded, fftw_complex *outdata_expanded);
void funChemPotential(double *chempotR, double *qr, double *qmod, double *qrsum);
void funMultiplyQs(double *qr, double *qrmod, double *qchain);
void funDensity(int start, int stop, double *phir, double *qchain, double qrsum);
void funAddArray(int sign[2], double *array1, double *array2, double *sum);
void funVolFrac(double frac, double *arrayF);
double getPhiAvg(int fieldIndex);
double getCHI_N_EFF(int field1, int field2);
double getG(int field1, int field2);
bool isFieldActive(int fieldIndex);
void funUpdate(double *wfieldA, double *densityA, double *wfieldB, double *densityB, double *wfieldC, double *densityC, double *wfieldD, double *densityD, double *wfieldE, double *densityE,
			   double *wfieldF, double *densityF, double *wfieldG, double *densityG, double *wfieldH, double *densityH, double *wfieldI, double *densityI, double *wfieldJ, double *densityJ,
			   double *pfield, double dt, double *wfieldS1, double *densityS1, double *wfieldS2, double *densityS2, double *wfieldS3, double *densityS3);
double funFreeEnergy(double *wfieldA, double *densityA, double *wfieldB, double *densityB, double *wfieldC, double *densityC, double *wfieldD, double *densityD, double *wfieldE, double *densityE,
					 double *wfieldF, double *densityF, double *wfieldG, double *densityG, double *wfieldH, double *densityH, double *wfieldI, double *densityI, double *wfieldJ, double *densityJ,
					 double *pfield, double *wfieldS1, double *densityS1, double *wfieldS2, double *densityS2, double *wfieldS3, double *densityS3);

int NX, NY, NZ;
int NS_DBK_1 = 0, NS_DBK_1_X, NS_DBK_1_Y;
int NS_DBK_2 = 0, NS_DBK_2_X, NS_DBK_2_Y;
int NS_DBK_3 = 0, NS_DBK_3_X, NS_DBK_3_Y;
int NS_DBK_4 = 0, NS_DBK_4_X, NS_DBK_4_Y;
int NS_HOMO_A = 0, NS_HOMO_B = 0, NS_HOMO_C = 0, NS_HOMO_D = 0, NS_HOMO_E = 0, NS_HOMO_F = 0, NS_HOMO_G = 0, NS_HOMO_H = 0, NS_HOMO_I = 0, NS_HOMO_J = 0;
int NS_S1 = 0, NS_S2 = 0, NS_S3 = 0;
int NS_EFF;

double ALPHA_DBK_1, ALPHA_DBK_2, ALPHA_DBK_3, ALPHA_DBK_4;
double ALPHA_HOMO_A, ALPHA_HOMO_B, ALPHA_HOMO_C, ALPHA_HOMO_D, ALPHA_HOMO_E, ALPHA_HOMO_F, ALPHA_HOMO_G, ALPHA_HOMO_H, ALPHA_HOMO_I, ALPHA_HOMO_J;
double ALPHA_S1, ALPHA_S2, ALPHA_S3;

double CHI_N_AB_EFF = 0.0, CHI_N_AC_EFF = 0.0, CHI_N_AD_EFF = 0.0, CHI_N_AE_EFF = 0.0, CHI_N_AF_EFF = 0.0, CHI_N_AG_EFF = 0.0, CHI_N_AH_EFF = 0.0, CHI_N_AI_EFF = 0.0, CHI_N_AJ_EFF = 0.0, CHI_N_S1_A_EFF = 0.0, CHI_N_S2_A_EFF = 0.0, CHI_N_S3_A_EFF = 0.0;
double CHI_N_BC_EFF = 0.0, CHI_N_BD_EFF = 0.0, CHI_N_BE_EFF = 0.0, CHI_N_BF_EFF = 0.0, CHI_N_BG_EFF = 0.0, CHI_N_BH_EFF = 0.0, CHI_N_BI_EFF = 0.0, CHI_N_BJ_EFF = 0.0, CHI_N_S1_B_EFF = 0.0, CHI_N_S2_B_EFF = 0.0, CHI_N_S3_B_EFF = 0.0;
double CHI_N_CD_EFF = 0.0, CHI_N_CE_EFF = 0.0, CHI_N_CF_EFF = 0.0, CHI_N_CG_EFF = 0.0, CHI_N_CH_EFF = 0.0, CHI_N_CI_EFF = 0.0, CHI_N_CJ_EFF = 0.0, CHI_N_S1_C_EFF = 0.0, CHI_N_S2_C_EFF = 0.0, CHI_N_S3_C_EFF = 0.0;
double CHI_N_DE_EFF = 0.0, CHI_N_DF_EFF = 0.0, CHI_N_DG_EFF = 0.0, CHI_N_DH_EFF = 0.0, CHI_N_DI_EFF = 0.0, CHI_N_DJ_EFF = 0.0, CHI_N_S1_D_EFF = 0.0, CHI_N_S2_D_EFF = 0.0, CHI_N_S3_D_EFF = 0.0;
double CHI_N_EF_EFF = 0.0, CHI_N_EG_EFF = 0.0, CHI_N_EH_EFF = 0.0, CHI_N_EI_EFF = 0.0, CHI_N_EJ_EFF = 0.0, CHI_N_S1_E_EFF = 0.0, CHI_N_S2_E_EFF = 0.0, CHI_N_S3_E_EFF = 0.0;
double CHI_N_FG_EFF = 0.0, CHI_N_FH_EFF = 0.0, CHI_N_FI_EFF = 0.0, CHI_N_FJ_EFF = 0.0, CHI_N_S1_F_EFF = 0.0, CHI_N_S2_F_EFF = 0.0, CHI_N_S3_F_EFF = 0.0;
double CHI_N_GH_EFF = 0.0, CHI_N_GI_EFF = 0.0, CHI_N_GJ_EFF = 0.0, CHI_N_S1_G_EFF = 0.0, CHI_N_S2_G_EFF = 0.0, CHI_N_S3_G_EFF = 0.0;
double CHI_N_HI_EFF = 0.0, CHI_N_HJ_EFF = 0.0, CHI_N_S1_H_EFF = 0.0, CHI_N_S2_H_EFF = 0.0, CHI_N_S3_H_EFF = 0.0;
double CHI_N_IJ_EFF = 0.0, CHI_N_S1_I_EFF = 0.0, CHI_N_S2_I_EFF = 0.0, CHI_N_S3_I_EFF = 0.0;
double CHI_N_S1_J_EFF = 0.0, CHI_N_S2_J_EFF = 0.0, CHI_N_S3_J_EFF = 0.0;

double CHI_N_S1_S2_EFF = 0.0, CHI_N_S1_S3_EFF = 0.0;
double CHI_N_S2_S3_EFF = 0.0;

// Global array to store phi averages
double phi_avgs[13] = {0.0}; // Initialize all elements to 0.0

float MF;

unsigned long int iIter;
int N; // used for Dirichlet boundary conditions

// Define FieldInfo struct globally
typedef struct {
    double *wfield;
    double *density;
    double wInit;
    double delW;
    double rho_minus_rhoavg;
    double vol;
    char active;
} FieldInfo;

void printActiveFieldUpdate(FieldInfo *fields, int activeCount, size_t idx, double correctionFactor);

int main(int argc, char *argv[])
{
	// check if the volumes sum up to 1, otherwise do not execute the program
	double volumeCheck = VOL_DBK_1 + VOL_DBK_2 + VOL_DBK_3 + VOL_DBK_4 +
						VOL_HOMOPOLYMER_A + VOL_HOMOPOLYMER_B + VOL_HOMOPOLYMER_C + VOL_HOMOPOLYMER_D + VOL_HOMOPOLYMER_E +
						VOL_HOMOPOLYMER_F + VOL_HOMOPOLYMER_G + VOL_HOMOPOLYMER_H + VOL_HOMOPOLYMER_I + VOL_HOMOPOLYMER_J;

	double epsilon = 0.000001;

	if (fabs(1.0 - volumeCheck) >= epsilon) // compare to 1.0 volume of the system, small number accounts for the double precision error
	{
			FILE *fiderror;
			fiderror = fopen("DENSITY_CHECK_FAIL.error","w");
			fprintf(fiderror, "%lf", volumeCheck);
			fclose(fiderror);
			printf("                      <=============================\n");
			printf("Volumes do not sum up to 1.0, check the input parameters\n");
			printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
			exit(0);	
	}

	if (LX == 1.0) NX = 1; else NX = (int)((LX / DR_TARGET) + 0.5);
	if (LY == 1.0) NY = 1; else NY = (int)((LY / DR_TARGET) + 0.5);
	if (LZ == 1.0) NZ = 1; else NZ = (int)((LZ / DR_TARGET) + 0.5);

	// Set default values for effective chi parameters from parameters.h
	CHI_N_AB_EFF = CHI_N_AB; CHI_N_AC_EFF = CHI_N_AC; CHI_N_AD_EFF = CHI_N_AD; CHI_N_AE_EFF = CHI_N_AE; CHI_N_AF_EFF = CHI_N_AF; CHI_N_AG_EFF = CHI_N_AG; CHI_N_AH_EFF = CHI_N_AH; CHI_N_AI_EFF = CHI_N_AI; CHI_N_AJ_EFF = CHI_N_AJ; CHI_N_S1_A_EFF = CHI_N_S1_A; CHI_N_S2_A_EFF = CHI_N_S2_A; CHI_N_S3_A_EFF = CHI_N_S3_A;
	CHI_N_BC_EFF = CHI_N_BC; CHI_N_BD_EFF = CHI_N_BD; CHI_N_BE_EFF = CHI_N_BE; CHI_N_BF_EFF = CHI_N_BF; CHI_N_BG_EFF = CHI_N_BG; CHI_N_BH_EFF = CHI_N_BH; CHI_N_BI_EFF = CHI_N_BI; CHI_N_BJ_EFF = CHI_N_BJ; CHI_N_S1_B_EFF = CHI_N_S1_B; CHI_N_S2_B_EFF = CHI_N_S2_B; CHI_N_S3_B_EFF = CHI_N_S3_B;
	CHI_N_CD_EFF = CHI_N_CD; CHI_N_CE_EFF = CHI_N_CE; CHI_N_CF_EFF = CHI_N_CF; CHI_N_CG_EFF = CHI_N_CG; CHI_N_CH_EFF = CHI_N_CH; CHI_N_CI_EFF = CHI_N_CI; CHI_N_CJ_EFF = CHI_N_CJ; CHI_N_S1_C_EFF = CHI_N_S1_C; CHI_N_S2_C_EFF = CHI_N_S2_C; CHI_N_S3_C_EFF = CHI_N_S3_C;
	CHI_N_DE_EFF = CHI_N_DE; CHI_N_DF_EFF = CHI_N_DF; CHI_N_DG_EFF = CHI_N_DG; CHI_N_DH_EFF = CHI_N_DH; CHI_N_DI_EFF = CHI_N_DI; CHI_N_DJ_EFF = CHI_N_DJ; CHI_N_S1_D_EFF = CHI_N_S1_D; CHI_N_S2_D_EFF = CHI_N_S2_D; CHI_N_S3_D_EFF = CHI_N_S3_D;
	CHI_N_EF_EFF = CHI_N_EF; CHI_N_EG_EFF = CHI_N_EG; CHI_N_EH_EFF = CHI_N_EH; CHI_N_EI_EFF = CHI_N_EI; CHI_N_EJ_EFF = CHI_N_EJ; CHI_N_S1_E_EFF = CHI_N_S1_E; CHI_N_S2_E_EFF = CHI_N_S2_E; CHI_N_S3_E_EFF = CHI_N_S3_E;
	CHI_N_FG_EFF = CHI_N_FG; CHI_N_FH_EFF = CHI_N_FH; CHI_N_FI_EFF = CHI_N_FI; CHI_N_FJ_EFF = CHI_N_FJ; CHI_N_S1_F_EFF = CHI_N_S1_F; CHI_N_S2_F_EFF = CHI_N_S2_F; CHI_N_S3_F_EFF = CHI_N_S3_F;
	CHI_N_GH_EFF = CHI_N_GH; CHI_N_GI_EFF = CHI_N_GI; CHI_N_GJ_EFF = CHI_N_GJ; CHI_N_S1_G_EFF = CHI_N_S1_G; CHI_N_S2_G_EFF = CHI_N_S2_G; CHI_N_S3_G_EFF = CHI_N_S3_G;
	CHI_N_HI_EFF = CHI_N_HI; CHI_N_HJ_EFF = CHI_N_HJ; CHI_N_S1_H_EFF = CHI_N_S1_H; CHI_N_S2_H_EFF = CHI_N_S2_H; CHI_N_S3_H_EFF = CHI_N_S3_H;
	CHI_N_IJ_EFF = CHI_N_IJ; CHI_N_S1_I_EFF = CHI_N_S1_I; CHI_N_S2_I_EFF = CHI_N_S2_I; CHI_N_S3_I_EFF = CHI_N_S3_I;
	CHI_N_S1_J_EFF = CHI_N_S1_J; CHI_N_S2_J_EFF = CHI_N_S2_J; CHI_N_S3_J_EFF = CHI_N_S3_J;
	CHI_N_S1_S2_EFF = CHI_N_S1_S2; CHI_N_S1_S3_EFF = CHI_N_S1_S3;
	CHI_N_S2_S3_EFF = CHI_N_S2_S3;

	if (VOL_DBK_1 > 0.0)
	{
		NS_DBK_1 = floor(LEN_DBK_1 * NS);
		if (NS_DBK_1 % 2 == 0) NS_DBK_1 = NS_DBK_1 - 1; // NS must be odd
		NS_DBK_1_X = floor(F_INPUT_DBK_1 * NS_DBK_1) + 1;
		if (NS_DBK_1_X % 2 == 0) 
			NS_DBK_1_X = NS_DBK_1_X - 1;
		NS_DBK_1_Y = NS_DBK_1 - NS_DBK_1_X;

		ALPHA_DBK_1 = (double)((NS_DBK_1 - 1) * DS); // contour length
	} 

	if (VOL_DBK_2 > 0.0)
	{
		NS_DBK_2 = floor(LEN_DBK_2 * NS);
		if (NS_DBK_2 % 2 == 0) NS_DBK_2 = NS_DBK_2 - 1;
		NS_DBK_2_X = floor(F_INPUT_DBK_2 * NS_DBK_2) + 1;
		if (NS_DBK_2_X % 2 == 0) 
			NS_DBK_2_X = NS_DBK_2_X - 1;
		NS_DBK_2_Y = NS_DBK_2 - NS_DBK_2_X;

		ALPHA_DBK_2 = (double)((NS_DBK_2 - 1) * DS);
	} 

	if (VOL_DBK_3 > 0.0)
	{
		NS_DBK_3 = floor(LEN_DBK_3 * NS);
		if (NS_DBK_3 % 2 == 0) NS_DBK_3 = NS_DBK_3 - 1;
		NS_DBK_3_X = floor(F_INPUT_DBK_3 * NS_DBK_3) + 1;
		if (NS_DBK_3_X % 2 == 0) 
			NS_DBK_3_X = NS_DBK_3_X - 1;
		NS_DBK_3_Y = NS_DBK_3 - NS_DBK_3_X;

		ALPHA_DBK_3 = (double)((NS_DBK_3 - 1) * DS);
	} 

	if (VOL_DBK_4 > 0.0)
	{
		NS_DBK_4 = floor(LEN_DBK_4 * NS);
		if (NS_DBK_4 % 2 == 0) NS_DBK_4 = NS_DBK_4 - 1;
		NS_DBK_4_X = floor(F_INPUT_DBK_4 * NS_DBK_4) + 1;
		if (NS_DBK_4_X % 2 == 0) 
			NS_DBK_4_X = NS_DBK_4_X - 1;
		NS_DBK_4_Y = NS_DBK_4 - NS_DBK_4_X;

		ALPHA_DBK_4 = (double)((NS_DBK_4 - 1) * DS);
	} 

	// Homopolymers
	if (VOL_HOMOPOLYMER_A > 0.0) 
	{
		NS_HOMO_A = (floor(LEN_HOMO_A * NS));
		if (NS_HOMO_A % 2 == 0 || NS_HOMO_A > NS)
			NS_HOMO_A = NS_HOMO_A - 1;

		ALPHA_HOMO_A = (double)((NS_HOMO_A - 1) * DS);
	} 

	if (VOL_HOMOPOLYMER_B > 0.0) 
	{
		NS_HOMO_B = (floor(LEN_HOMO_B * NS));
		if (NS_HOMO_B % 2 == 0 || NS_HOMO_B > NS)
			NS_HOMO_B = NS_HOMO_B - 1;

		ALPHA_HOMO_B = (double)((NS_HOMO_B - 1) * DS);
	}

	if (VOL_HOMOPOLYMER_C > 0.0) 
	{
		NS_HOMO_C = (floor(LEN_HOMO_C * NS));
		if (NS_HOMO_C % 2 == 0 || NS_HOMO_C > NS)
			NS_HOMO_C = NS_HOMO_C - 1;

		ALPHA_HOMO_C = (double)((NS_HOMO_C - 1) * DS);
	} 

	if (VOL_HOMOPOLYMER_D > 0.0) 
	{
		NS_HOMO_D = (floor(LEN_HOMO_D * NS));
		if (NS_HOMO_D % 2 == 0 || NS_HOMO_D > NS)
			NS_HOMO_D = NS_HOMO_D - 1;

		ALPHA_HOMO_D = (double)((NS_HOMO_D - 1) * DS);
	}

	if (VOL_HOMOPOLYMER_E > 0.0) 
	{
		NS_HOMO_E = (floor(LEN_HOMO_E * NS));
		if (NS_HOMO_E % 2 == 0 || NS_HOMO_E > NS)
			NS_HOMO_E = NS_HOMO_E - 1;

		ALPHA_HOMO_E = (double)((NS_HOMO_E - 1) * DS);
	}

	if (VOL_HOMOPOLYMER_F > 0.0) 
	{
		NS_HOMO_F = (floor(LEN_HOMO_F * NS));
		if (NS_HOMO_F % 2 == 0 || NS_HOMO_F > NS)
			NS_HOMO_F = NS_HOMO_F - 1;

		ALPHA_HOMO_F = (double)((NS_HOMO_F - 1) * DS);
	} 

	if (VOL_HOMOPOLYMER_G > 0.0) 
	{
		NS_HOMO_G = (floor(LEN_HOMO_G * NS));
		if (NS_HOMO_G % 2 == 0 || NS_HOMO_G > NS)
			NS_HOMO_G = NS_HOMO_G - 1;

		ALPHA_HOMO_G = (double)((NS_HOMO_G - 1) * DS);
	}

	if (VOL_HOMOPOLYMER_H > 0.0) 
	{
		NS_HOMO_H = (floor(LEN_HOMO_H * NS));
		if (NS_HOMO_H % 2 == 0 || NS_HOMO_H > NS)
			NS_HOMO_H = NS_HOMO_H - 1;

		ALPHA_HOMO_H = (double)((NS_HOMO_H - 1) * DS);
	} 

	if (VOL_HOMOPOLYMER_I > 0.0) 
	{
		NS_HOMO_I = (floor(LEN_HOMO_I * NS));
		if (NS_HOMO_I % 2 == 0 || NS_HOMO_I > NS)
			NS_HOMO_I = NS_HOMO_I - 1;

		ALPHA_HOMO_I = (double)((NS_HOMO_I - 1) * DS);
	} 

	if (VOL_HOMOPOLYMER_J > 0.0) 
	{
		NS_HOMO_J = (floor(LEN_HOMO_J * NS));
		if (NS_HOMO_J % 2 == 0 || NS_HOMO_J > NS)
			NS_HOMO_J = NS_HOMO_J - 1;

		ALPHA_HOMO_J = (double)((NS_HOMO_J - 1) * DS);
	} 

	if (MASK)
	{
		if (VOL_S1 > 0.0) 
		{
			NS_S1 = (floor(LEN_S1 * NS));
			if (NS_S1 % 2 == 0)
				NS_S1 = NS_S1 - 1;

			ALPHA_S1 = (double)((NS_S1 - 1) * DS);
		}

		if (VOL_S2 > 0.0) 
		{
			NS_S2 = (floor(LEN_S2 * NS));
			if (NS_S2 % 2 == 0)
				NS_S2 = NS_S2 - 1;

			ALPHA_S2 = (double)((NS_S2 - 1) * DS);
		} 

		if (VOL_S3 > 0.0) 
		{
			NS_S3 = (floor(LEN_S3 * NS));
			if (NS_S3 % 2 == 0)
				NS_S3 = NS_S3 - 1;

			ALPHA_S3 = (double)((NS_S3 - 1) * DS);
		} 
	}

	// Average densities
	// Regular polymers (A-J)
    phi_avgs[0] = VOL_HOMOPOLYMER_A + F_INPUT_DBK_1 * VOL_DBK_1;        // A
    phi_avgs[1] = VOL_HOMOPOLYMER_B + (1.0 - F_INPUT_DBK_1) * VOL_DBK_1; // B
    phi_avgs[2] = VOL_HOMOPOLYMER_C;                                      // C
    phi_avgs[3] = VOL_HOMOPOLYMER_D + F_INPUT_DBK_2 * VOL_DBK_2;         // D
    phi_avgs[4] = VOL_HOMOPOLYMER_E + (1.0 - F_INPUT_DBK_2) * VOL_DBK_2; // E
    phi_avgs[5] = VOL_HOMOPOLYMER_F + F_INPUT_DBK_3 * VOL_DBK_3;         // F
    phi_avgs[6] = VOL_HOMOPOLYMER_G + (1.0 - F_INPUT_DBK_3) * VOL_DBK_3; // G
    phi_avgs[7] = VOL_HOMOPOLYMER_H + F_INPUT_DBK_4 * VOL_DBK_4;         // H
    phi_avgs[8] = VOL_HOMOPOLYMER_I + (1.0 - F_INPUT_DBK_4) * VOL_DBK_4; // I
    phi_avgs[9] = VOL_HOMOPOLYMER_J;                                      // J

	// if vol is set to zero, then turn off all the chi N not used
	// the densities will be zero and will cancel out in the udpate, but the correction factor will see the chi N
	if (fabs(phi_avgs[0]) <= epsilon) // the variable is considered to be zero
	{
		// turn off all A interactions
		CHI_N_AB_EFF = 0.0; CHI_N_AC_EFF = 0.0; CHI_N_AD_EFF = 0.0; CHI_N_AE_EFF = 0.0; CHI_N_AF_EFF = 0.0; CHI_N_AG_EFF = 0.0; CHI_N_AH_EFF = 0.0; CHI_N_AI_EFF = 0.0; CHI_N_AJ_EFF = 0.0; CHI_N_S1_A_EFF = 0.0; CHI_N_S2_A_EFF = 0.0; CHI_N_S3_A_EFF = 0.0;
	}
	if (fabs(phi_avgs[1]) <= epsilon)
	{
		// turn off all B interactions
		CHI_N_AB_EFF = 0.0; CHI_N_BC_EFF = 0.0; CHI_N_BD_EFF = 0.0; CHI_N_BE_EFF = 0.0; CHI_N_BF_EFF = 0.0; CHI_N_BG_EFF = 0.0; CHI_N_BH_EFF = 0.0; CHI_N_BI_EFF = 0.0; CHI_N_BJ_EFF = 0.0; CHI_N_S1_B_EFF = 0.0; CHI_N_S2_B_EFF = 0.0; CHI_N_S3_B_EFF = 0.0;
	}
	if (fabs(phi_avgs[2]) <= epsilon)
	{
		// turn off all C interactions
		CHI_N_AC_EFF = 0.0; CHI_N_BC_EFF = 0.0; CHI_N_CD_EFF = 0.0; CHI_N_CE_EFF = 0.0; CHI_N_CF_EFF = 0.0; CHI_N_CG_EFF = 0.0; CHI_N_CH_EFF = 0.0; CHI_N_CI_EFF = 0.0; CHI_N_CJ_EFF = 0.0; CHI_N_S1_C_EFF = 0.0; CHI_N_S2_C_EFF = 0.0; CHI_N_S3_C_EFF = 0.0;
	}
	if (fabs(phi_avgs[3]) <= epsilon)
	{
		// turn off all D interactions
		CHI_N_AD_EFF = 0.0; CHI_N_BD_EFF = 0.0; CHI_N_CD_EFF = 0.0; CHI_N_DE_EFF = 0.0; CHI_N_DF_EFF = 0.0; CHI_N_DG_EFF = 0.0; CHI_N_DH_EFF = 0.0; CHI_N_DI_EFF = 0.0; CHI_N_DJ_EFF = 0.0; CHI_N_S1_D_EFF = 0.0; CHI_N_S2_D_EFF = 0.0; CHI_N_S3_D_EFF = 0.0;
	}
	if (fabs(phi_avgs[4]) <= epsilon)
	{
		// turn off all E interactions
		CHI_N_AE_EFF = 0.0; CHI_N_BE_EFF = 0.0; CHI_N_CE_EFF = 0.0; CHI_N_DE_EFF = 0.0; CHI_N_EF_EFF = 0.0; CHI_N_EG_EFF = 0.0; CHI_N_EH_EFF = 0.0; CHI_N_EI_EFF = 0.0; CHI_N_EJ_EFF = 0.0; CHI_N_S1_E_EFF = 0.0; CHI_N_S2_E_EFF = 0.0; CHI_N_S3_E_EFF = 0.0;
	}
	if (fabs(phi_avgs[5]) <= epsilon)
	{
		// turn off all F interactions
		CHI_N_AF_EFF = 0.0; CHI_N_BF_EFF = 0.0; CHI_N_CF_EFF = 0.0; CHI_N_DF_EFF = 0.0; CHI_N_EF_EFF = 0.0; CHI_N_FG_EFF = 0.0; CHI_N_FH_EFF = 0.0; CHI_N_FI_EFF = 0.0; CHI_N_FJ_EFF = 0.0; CHI_N_S1_F_EFF = 0.0; CHI_N_S2_F_EFF = 0.0; CHI_N_S3_F_EFF = 0.0;
	}
	if (fabs(phi_avgs[6]) <= epsilon)
	{
		// turn off all G interactions
		CHI_N_AG_EFF = 0.0; CHI_N_BG_EFF = 0.0; CHI_N_CG_EFF = 0.0; CHI_N_DG_EFF = 0.0; CHI_N_EG_EFF = 0.0; CHI_N_FG_EFF = 0.0; CHI_N_GH_EFF = 0.0; CHI_N_GI_EFF = 0.0; CHI_N_GJ_EFF = 0.0; CHI_N_S1_G_EFF = 0.0; CHI_N_S2_G_EFF = 0.0; CHI_N_S3_G_EFF = 0.0;
	}
	if (fabs(phi_avgs[7]) <= epsilon)
	{
		// turn off all H interactions
		CHI_N_AH_EFF = 0.0; CHI_N_BH_EFF = 0.0; CHI_N_CH_EFF = 0.0; CHI_N_DH_EFF = 0.0; CHI_N_EH_EFF = 0.0; CHI_N_FH_EFF = 0.0; CHI_N_GH_EFF = 0.0; CHI_N_HI_EFF = 0.0; CHI_N_HJ_EFF = 0.0; CHI_N_S1_H_EFF = 0.0; CHI_N_S2_H_EFF = 0.0; CHI_N_S3_H_EFF = 0.0;
	}
	if (fabs(phi_avgs[8]) <= epsilon)
	{
		// turn off all I interactions
		CHI_N_AI_EFF = 0.0; CHI_N_BI_EFF = 0.0; CHI_N_CI_EFF = 0.0; CHI_N_DI_EFF = 0.0; CHI_N_EI_EFF = 0.0; CHI_N_FI_EFF = 0.0; CHI_N_GI_EFF = 0.0; CHI_N_HI_EFF = 0.0; CHI_N_IJ_EFF = 0.0; CHI_N_S1_I_EFF = 0.0; CHI_N_S2_I_EFF = 0.0; CHI_N_S3_I_EFF = 0.0;
	}
	if (fabs(phi_avgs[9]) <= epsilon)
	{
		// turn off all J interactions
		CHI_N_AJ_EFF = 0.0; CHI_N_BJ_EFF = 0.0; CHI_N_CJ_EFF = 0.0; CHI_N_DJ_EFF = 0.0; CHI_N_EJ_EFF = 0.0; CHI_N_FJ_EFF = 0.0; CHI_N_GJ_EFF = 0.0; CHI_N_HJ_EFF = 0.0; CHI_N_IJ_EFF = 0.0; CHI_N_S1_J_EFF = 0.0; CHI_N_S2_J_EFF = 0.0; CHI_N_S3_J_EFF = 0.0;
	}
	if (MASK) // check which ones you will use
	{
		if (fabs(VOL_S1) <= epsilon)
		{
			// turn off all S1 interactions
			CHI_N_S1_A_EFF = 0.0; CHI_N_S1_B_EFF = 0.0; CHI_N_S1_C_EFF = 0.0; CHI_N_S1_D_EFF = 0.0; CHI_N_S1_E_EFF = 0.0; CHI_N_S1_F_EFF = 0.0; CHI_N_S1_G_EFF = 0.0; CHI_N_S1_H_EFF = 0.0; CHI_N_S1_I_EFF = 0.0; CHI_N_S1_J_EFF = 0.0; CHI_N_S1_S2_EFF = 0.0; CHI_N_S1_S3_EFF = 0.0;
		}
		else
		{
			phi_avgs[10] = VOL_S1;
		}

		if (fabs(VOL_S2) <= epsilon)
		{
			// turn off all S2 interactions
			CHI_N_S2_A_EFF = 0.0; CHI_N_S2_B_EFF = 0.0; CHI_N_S2_C_EFF = 0.0; CHI_N_S2_D_EFF = 0.0; CHI_N_S2_E_EFF = 0.0; CHI_N_S2_F_EFF = 0.0; CHI_N_S2_G_EFF = 0.0; CHI_N_S2_H_EFF = 0.0; CHI_N_S2_I_EFF = 0.0; CHI_N_S2_J_EFF = 0.0; CHI_N_S1_S2_EFF = 0.0; CHI_N_S2_S3_EFF = 0.0;
		}
		else
		{
			phi_avgs[11] = VOL_S2;
		}

		if (fabs(VOL_S3) <= epsilon)
		{
			// turn off all S3 interactions
			CHI_N_S3_A_EFF = 0.0; CHI_N_S3_B_EFF = 0.0; CHI_N_S3_C_EFF = 0.0; CHI_N_S3_D_EFF = 0.0; CHI_N_S3_E_EFF = 0.0; CHI_N_S3_F_EFF = 0.0; CHI_N_S3_G_EFF = 0.0; CHI_N_S3_H_EFF = 0.0; CHI_N_S3_I_EFF = 0.0; CHI_N_S3_J_EFF = 0.0; CHI_N_S1_S3_EFF = 0.0; CHI_N_S2_S3_EFF = 0.0;
		}
		else
		{
			phi_avgs[12] = VOL_S3;
		}
	}
	else // turn off all interactions for the coatings
	{
		phi_avgs[10] = 0;
		phi_avgs[11] = 0;
		phi_avgs[12] = 0;

		CHI_N_S1_A_EFF = 0.0; CHI_N_S1_B_EFF = 0.0; CHI_N_S1_C_EFF = 0.0; CHI_N_S1_D_EFF = 0.0; CHI_N_S1_E_EFF = 0.0; CHI_N_S1_F_EFF = 0.0; CHI_N_S1_G_EFF = 0.0; CHI_N_S1_H_EFF = 0.0; CHI_N_S1_I_EFF = 0.0; CHI_N_S1_J_EFF = 0.0; CHI_N_S1_S2_EFF = 0.0; CHI_N_S1_S3_EFF = 0.0;
		CHI_N_S2_A_EFF = 0.0; CHI_N_S2_B_EFF = 0.0; CHI_N_S2_C_EFF = 0.0; CHI_N_S2_D_EFF = 0.0; CHI_N_S2_E_EFF = 0.0; CHI_N_S2_F_EFF = 0.0; CHI_N_S2_G_EFF = 0.0; CHI_N_S2_H_EFF = 0.0; CHI_N_S2_I_EFF = 0.0; CHI_N_S2_J_EFF = 0.0; CHI_N_S1_S2_EFF = 0.0; CHI_N_S2_S3_EFF = 0.0;
		CHI_N_S3_A_EFF = 0.0; CHI_N_S3_B_EFF = 0.0; CHI_N_S3_C_EFF = 0.0; CHI_N_S3_D_EFF = 0.0; CHI_N_S3_E_EFF = 0.0; CHI_N_S3_F_EFF = 0.0; CHI_N_S3_G_EFF = 0.0; CHI_N_S3_H_EFF = 0.0; CHI_N_S3_I_EFF = 0.0; CHI_N_S3_J_EFF = 0.0; CHI_N_S1_S3_EFF = 0.0; CHI_N_S2_S3_EFF = 0.0;
	}

	// Definition of various utility variables
	unsigned int i, j, k, iStart, iStop, field;
	FILE *fid;
	
	// Definition of variables used to initialize the random number generator
	unsigned short int iSeed[3];
	unsigned short int *pOldSeed;
	long int iCurrTime;
	
	double RMSDev;
	double H_fe;
	double H_fe_fraction;

	double chemPot_dbk_1 = 0.0, Qmod_dbk_1, Qsum_dbk_1;
	double chemPot_dbk_2 = 0.0, Qmod_dbk_2, Qsum_dbk_2;
	double chemPot_dbk_3 = 0.0, Qmod_dbk_3, Qsum_dbk_3;
	double chemPot_dbk_4 = 0.0, Qmod_dbk_4, Qsum_dbk_4;
	double chemPot_homo_A = 0.0, Qmod_homo_A, Qsum_homo_A;
	double chemPot_homo_B = 0.0, Qmod_homo_B, Qsum_homo_B;
	double chemPot_homo_C = 0.0, Qmod_homo_C, Qsum_homo_C;
	double chemPot_homo_D = 0.0, Qmod_homo_D, Qsum_homo_D;
	double chemPot_homo_E = 0.0, Qmod_homo_E, Qsum_homo_E;
	double chemPot_homo_F = 0.0, Qmod_homo_F, Qsum_homo_F;
	double chemPot_homo_G = 0.0, Qmod_homo_G, Qsum_homo_G;
	double chemPot_homo_H = 0.0, Qmod_homo_H, Qsum_homo_H;
	double chemPot_homo_I = 0.0, Qmod_homo_I, Qsum_homo_I;
	double chemPot_homo_J = 0.0, Qmod_homo_J, Qsum_homo_J;
	double chemPot_S1 = 0.0, Qmod_S1, Qsum_S1;
	double chemPot_S2 = 0.0, Qmod_S2, Qsum_S2;
	double chemPot_S3 = 0.0, Qmod_S3, Qsum_S3;

	// Definition of variables used to sum arrays
	int iSign[2];
	double vol_frac;

	// Fields
	double *aW_A = calloc((NX*NY*NZ),sizeof(*aW_A));
	double *aW_B = calloc((NX*NY*NZ),sizeof(*aW_B));
	double *aW_C = calloc((NX*NY*NZ),sizeof(*aW_C));
	double *aW_D = calloc((NX*NY*NZ),sizeof(*aW_D));
	double *aW_E = calloc((NX*NY*NZ),sizeof(*aW_E));
	double *aW_F = calloc((NX*NY*NZ),sizeof(*aW_F));
	double *aW_G = calloc((NX*NY*NZ),sizeof(*aW_G));
	double *aW_H = calloc((NX*NY*NZ),sizeof(*aW_H));
	double *aW_I = calloc((NX*NY*NZ),sizeof(*aW_I));
	double *aW_J = calloc((NX*NY*NZ),sizeof(*aW_J));
	double *aP = calloc((NX*NY*NZ),sizeof(*aP));

	// always set the fields, they will be saved in the densities files
	double *aW_S1 = aW_S1 = calloc((NX * NY * NZ), sizeof(*aW_S1));
	double *aW_S2 = aW_S2 = calloc((NX * NY * NZ), sizeof(*aW_S2));
	double *aW_S3 = aW_S3 = calloc((NX * NY * NZ), sizeof(*aW_S3));

	// diblock 1
	double *aQ_dbk_1 = NULL;
	double *aQMod_dbk_1 = NULL;
	double *aQMult_dbk_1 = NULL;
	double *aPhi_X_dbk_1 = NULL;
	double *aPhi_Y_dbk_1 = NULL;

	if (VOL_DBK_1 > 0.0)
	{
		aQ_dbk_1 = calloc((NX * NY * NZ * NS_DBK_1), sizeof(*aQ_dbk_1));
		aQMod_dbk_1 = calloc((NX * NY * NZ * NS_DBK_1), sizeof(*aQMod_dbk_1));
		aQMult_dbk_1 = calloc((NX * NY * NZ * NS_DBK_1), sizeof(*aQMult_dbk_1));
		aPhi_X_dbk_1 = calloc((NX * NY * NZ), sizeof(*aPhi_X_dbk_1));
		aPhi_Y_dbk_1 = calloc((NX * NY * NZ), sizeof(*aPhi_Y_dbk_1));
	}

	// diblock 2
	double *aQ_dbk_2 = NULL;
	double *aQMod_dbk_2 = NULL;
	double *aQMult_dbk_2 = NULL;
	double *aPhi_X_dbk_2 = NULL;
	double *aPhi_Y_dbk_2 = NULL;

	if (VOL_DBK_2 > 0.0)
	{
		aQ_dbk_2 = calloc((NX * NY * NZ * NS_DBK_2), sizeof(*aQ_dbk_2));
		aQMod_dbk_2 = calloc((NX * NY * NZ * NS_DBK_2), sizeof(*aQMod_dbk_2));
		aQMult_dbk_2 = calloc((NX * NY * NZ * NS_DBK_2), sizeof(*aQMult_dbk_2));
		aPhi_X_dbk_2 = calloc((NX * NY * NZ), sizeof(*aPhi_X_dbk_2));
		aPhi_Y_dbk_2 = calloc((NX * NY * NZ), sizeof(*aPhi_Y_dbk_2));
	}

	// diblock 3
	double *aQ_dbk_3 = NULL;
	double *aQMod_dbk_3 = NULL;
	double *aQMult_dbk_3 = NULL;
	double *aPhi_X_dbk_3 = NULL;
	double *aPhi_Y_dbk_3 = NULL;

	if (VOL_DBK_3 > 0.0)
	{
		aQ_dbk_3 = calloc((NX * NY * NZ * NS_DBK_3), sizeof(*aQ_dbk_3));
		aQMod_dbk_3 = calloc((NX * NY * NZ * NS_DBK_3), sizeof(*aQMod_dbk_3));
		aQMult_dbk_3 = calloc((NX * NY * NZ * NS_DBK_3), sizeof(*aQMult_dbk_3));
		aPhi_X_dbk_3 = calloc((NX * NY * NZ), sizeof(*aPhi_X_dbk_3));
		aPhi_Y_dbk_3 = calloc((NX * NY * NZ), sizeof(*aPhi_Y_dbk_3));
	}

	// diblock 4
	double *aQ_dbk_4 = NULL;
	double *aQMod_dbk_4 = NULL;
	double *aQMult_dbk_4 = NULL;
	double *aPhi_X_dbk_4 = NULL;
	double *aPhi_Y_dbk_4 = NULL;

	if (VOL_DBK_4 > 0.0)
	{
		aQ_dbk_4 = calloc((NX * NY * NZ * NS_DBK_4), sizeof(*aQ_dbk_4));
		aQMod_dbk_4 = calloc((NX * NY * NZ * NS_DBK_4), sizeof(*aQMod_dbk_4));
		aQMult_dbk_4 = calloc((NX * NY * NZ * NS_DBK_4), sizeof(*aQMult_dbk_4));
		aPhi_X_dbk_4 = calloc((NX * NY * NZ), sizeof(*aPhi_X_dbk_4));
		aPhi_Y_dbk_4 = calloc((NX * NY * NZ), sizeof(*aPhi_Y_dbk_4));
	}

	// Homopolymer A
	double *aQ_homo_A = NULL;
	double *aQMult_homo_A = NULL;
	double *aPhi_homo_A = NULL;

	if (VOL_HOMOPOLYMER_A > 0.0)
	{
		aQ_homo_A = calloc((NX * NY * NZ * NS_HOMO_A), sizeof(*aQ_homo_A));
		aQMult_homo_A = calloc((NX * NY * NZ * NS_HOMO_A), sizeof(*aQMult_homo_A));
		aPhi_homo_A = calloc((NX * NY * NZ), sizeof(*aPhi_homo_A));
	}

	// Homopolymer B
	double *aQ_homo_B = NULL;
	double *aQMult_homo_B = NULL;
	double *aPhi_homo_B = NULL;

	if (VOL_HOMOPOLYMER_B > 0.0)
	{
		aQ_homo_B = calloc((NX * NY * NZ * NS_HOMO_B), sizeof(*aQ_homo_B));
		aQMult_homo_B = calloc((NX * NY * NZ * NS_HOMO_B), sizeof(*aQMult_homo_B));
		aPhi_homo_B = calloc((NX * NY * NZ), sizeof(*aPhi_homo_B));
	}

	// Homopolymer C
	double *aQ_homo_C = NULL;
	double *aQMult_homo_C = NULL;
	double *aPhi_homo_C = NULL;
	
	if (VOL_HOMOPOLYMER_C > 0.0)
	{
		aQ_homo_C = calloc((NX * NY * NZ * NS_HOMO_C), sizeof(*aQ_homo_C));
		aQMult_homo_C = calloc((NX * NY * NZ * NS_HOMO_C), sizeof(*aQMult_homo_C));
		aPhi_homo_C = calloc((NX * NY * NZ), sizeof(*aPhi_homo_C));
	}
	
	// Homopolymer D
	double *aQ_homo_D = NULL;
	double *aQMult_homo_D = NULL;
	double *aPhi_homo_D = NULL;

	if (VOL_HOMOPOLYMER_D > 0.0)
	{
		aQ_homo_D = calloc((NX * NY * NZ * NS_HOMO_D), sizeof(*aQ_homo_D));
		aQMult_homo_D = calloc((NX * NY * NZ * NS_HOMO_D), sizeof(*aQMult_homo_D));
		aPhi_homo_D = calloc((NX * NY * NZ), sizeof(*aPhi_homo_D));
	}

	// Homopolymer E
	double *aQ_homo_E = NULL;
	double *aQMult_homo_E = NULL;
	double *aPhi_homo_E = NULL;
	
	if (VOL_HOMOPOLYMER_E > 0.0)
	{
		aQ_homo_E = calloc((NX * NY * NZ * NS_HOMO_E), sizeof(*aQ_homo_E));
		aQMult_homo_E = calloc((NX * NY * NZ * NS_HOMO_E), sizeof(*aQMult_homo_E));
		aPhi_homo_E = calloc((NX * NY * NZ), sizeof(*aPhi_homo_E));
	}
	
	// Homopolymer F
	double *aQ_homo_F = NULL;
	double *aQMult_homo_F = NULL;
	double *aPhi_homo_F = NULL;
	
	if (VOL_HOMOPOLYMER_F > 0.0)
	{
		aQ_homo_F = calloc((NX * NY * NZ * NS_HOMO_F), sizeof(*aQ_homo_F));
		aQMult_homo_F = calloc((NX * NY * NZ * NS_HOMO_F), sizeof(*aQMult_homo_F));
		aPhi_homo_F = calloc((NX * NY * NZ), sizeof(*aPhi_homo_F));
	}
	
	// Homopolymer G
	double *aQ_homo_G = NULL;
	double *aQMult_homo_G = NULL;
	double *aPhi_homo_G = NULL;
	
	if (VOL_HOMOPOLYMER_G > 0.0)
	{
		aQ_homo_G = calloc((NX * NY * NZ * NS_HOMO_G), sizeof(*aQ_homo_G));
		aQMult_homo_G = calloc((NX * NY * NZ * NS_HOMO_G), sizeof(*aQMult_homo_G));
		aPhi_homo_G = calloc((NX * NY * NZ), sizeof(*aPhi_homo_G));
	}
	
	// Homopolymer H
	double *aQ_homo_H = NULL;
	double *aQMult_homo_H = NULL;
	double *aPhi_homo_H = NULL;
	
	if (VOL_HOMOPOLYMER_H > 0.0)
	{
		aQ_homo_H = calloc((NX * NY * NZ * NS_HOMO_H), sizeof(*aQ_homo_H));
		aQMult_homo_H = calloc((NX * NY * NZ * NS_HOMO_H), sizeof(*aQMult_homo_H));
		aPhi_homo_H = calloc((NX * NY * NZ), sizeof(*aPhi_homo_H));
	}
	
	// Homopolymer I
	double *aQ_homo_I = NULL;
	double *aQMult_homo_I = NULL;
	double *aPhi_homo_I = NULL;
	
	if (VOL_HOMOPOLYMER_I > 0.0)
	{
		aQ_homo_I = calloc((NX * NY * NZ * NS_HOMO_I), sizeof(*aQ_homo_I));
		aQMult_homo_I = calloc((NX * NY * NZ * NS_HOMO_I), sizeof(*aQMult_homo_I));
		aPhi_homo_I = calloc((NX * NY * NZ), sizeof(*aPhi_homo_I));
	}
	
	// Homopolymer J
	double *aQ_homo_J = NULL;
	double *aQMult_homo_J = NULL;
	double *aPhi_homo_J = NULL;

	if (VOL_HOMOPOLYMER_J > 0.0)
	{
		aQ_homo_J = calloc((NX * NY * NZ * NS_HOMO_J), sizeof(*aQ_homo_J));
		aQMult_homo_J = calloc((NX * NY * NZ * NS_HOMO_J), sizeof(*aQMult_homo_J));
		aPhi_homo_J = calloc((NX * NY * NZ), sizeof(*aPhi_homo_J));
	}

	// Substrates
	// S1
	double *aQ_S1 = NULL;
	double *aQ_S1Mod = NULL;
	double *aQMult_S1 = NULL;
	double *aS1_mask = NULL;
	double *aPhi_S1 = calloc((NX * NY * NZ), sizeof(*aPhi_S1));
	
	// S2
	double *aQ_S2 = NULL;
	double *aQ_S2Mod = NULL;
	double *aQMult_S2 = NULL;
	double *aS2_mask = NULL;
	double *aPhi_S2 = calloc((NX * NY * NZ), sizeof(*aPhi_S2));

	// S3
	double *aQ_S3 = NULL;
	double *aQ_S3Mod = NULL;
	double *aQMult_S3 = NULL;
	double *aS3_mask = NULL;
	double *aPhi_S3 = calloc((NX * NY * NZ), sizeof(*aPhi_S3));

	// Total densities
	double *aPhi_A_Tot = calloc((NX * NY * NZ), sizeof(*aPhi_A_Tot));
	double *aPhi_B_Tot = calloc((NX * NY * NZ), sizeof(*aPhi_B_Tot));
	double *aPhi_C_Tot = calloc((NX * NY * NZ), sizeof(*aPhi_C_Tot));
	double *aPhi_D_Tot = calloc((NX * NY * NZ), sizeof(*aPhi_D_Tot));
	double *aPhi_E_Tot = calloc((NX * NY * NZ), sizeof(*aPhi_E_Tot));
	double *aPhi_F_Tot = calloc((NX * NY * NZ), sizeof(*aPhi_F_Tot));
	double *aPhi_G_Tot = calloc((NX * NY * NZ), sizeof(*aPhi_G_Tot));
	double *aPhi_H_Tot = calloc((NX * NY * NZ), sizeof(*aPhi_H_Tot));
	double *aPhi_I_Tot = calloc((NX * NY * NZ), sizeof(*aPhi_I_Tot));
	double *aPhi_J_Tot = calloc((NX * NY * NZ), sizeof(*aPhi_J_Tot));
	double *aPhi_Tot = calloc((NX * NY * NZ), sizeof(*aPhi_Tot));

	// Definition of the variables used for the FFT using FFTW
	fftw_complex *cDataIn;
	fftw_complex *cDataOut;
	fftw_plan pFFTPlanForward, pFFTPlanReverse;

	double *aExpArg = NULL;

	if (MASK)
	{
		printf("Expanding the Fourier Transform for Dirichlet boundary conditions. \n");
		N = 2 * (NZ + 1);

		cDataIn = (fftw_complex *)fftw_malloc(NX * NY * N * sizeof(fftw_complex));
		cDataOut = (fftw_complex *)fftw_malloc(NX * NY * N * sizeof(fftw_complex));

		pFFTPlanForward = fftw_plan_dft_3d(NX, NY, N, cDataIn, cDataOut, FFTW_FORWARD, FFTW_MEASURE);
		pFFTPlanReverse = fftw_plan_dft_3d(NX, NY, N, cDataIn, cDataOut, FFTW_BACKWARD, FFTW_MEASURE);

		aExpArg = calloc((NX * NY * N), sizeof(*aExpArg));
	}
	else
	{
		printf("Setting aExpArg for periodic boundary conditions. \n");
		// Allocate the input and output data arrays for the FFT
		cDataIn = (fftw_complex *)fftw_malloc(NX * NY * NZ * sizeof(fftw_complex));
		cDataOut = (fftw_complex *)fftw_malloc(NX * NY * NZ * sizeof(fftw_complex));

		pFFTPlanForward = fftw_plan_dft_3d(NX, NY, NZ, cDataIn, cDataOut, FFTW_FORWARD, FFTW_MEASURE);
		pFFTPlanReverse = fftw_plan_dft_3d(NX, NY, NZ, cDataIn, cDataOut, FFTW_BACKWARD, FFTW_MEASURE);

		aExpArg = calloc((NX * NY * NZ), sizeof(*aExpArg));
	}

	// Open the counter file and get the iteration number
	fid = fopen(COUNTER,"r");
	// ---
	if (fid == NULL) {
		perror("Error opening counter file");
		exit(EXIT_FAILURE);
	}
	//if (fscanf(fid, "%1d", &iIter) != 1) {
	if (fscanf(fid, "%lu", &iIter) != 1) {
		perror("Error reading the integer from counter file");
		exit(EXIT_FAILURE);
	}
	// ---
	fclose(fid);

	double coating_vol = 0.0;
	if (MASK) {
	// Adjust volume for coatings
		coating_vol = VOL_S1 + VOL_S2 + VOL_S3;
		printf("coating_vol = %f\n", coating_vol);
		// Adjust the volume fraction for the coatings in the system
		phi_avgs[0] = phi_avgs[0] * (1.0 - coating_vol);
		phi_avgs[1] = phi_avgs[1] * (1.0 - coating_vol);
		phi_avgs[2] = phi_avgs[2] * (1.0 - coating_vol);
		phi_avgs[3] = phi_avgs[3] * (1.0 - coating_vol);
		phi_avgs[4] = phi_avgs[4] * (1.0 - coating_vol);
		phi_avgs[5] = phi_avgs[5] * (1.0 - coating_vol);
		phi_avgs[6] = phi_avgs[6] * (1.0 - coating_vol);
		phi_avgs[7] = phi_avgs[7] * (1.0 - coating_vol);
		phi_avgs[8] = phi_avgs[8] * (1.0 - coating_vol);
		phi_avgs[9] = phi_avgs[9] * (1.0 - coating_vol);

		if (VOL_S1 > 0.0)
		{
			aQ_S1 = calloc((NX * NY * NZ * NS_S1), sizeof(*aQ_S1));
			aQ_S1Mod = calloc((NX * NY * NZ * NS_S1), sizeof(*aQ_S1Mod));
			aQMult_S1 = calloc((NX * NY * NZ * NS_S1), sizeof(*aQMult_S1));
		}

		if (VOL_S2 > 0.0)
		{
			aQ_S2 = calloc((NX * NY * NZ * NS_S2), sizeof(*aQ_S2));
			aQ_S2Mod = calloc((NX * NY * NZ * NS_S2), sizeof(*aQ_S2Mod));
			aQMult_S2 = calloc((NX * NY * NZ * NS_S2), sizeof(*aQMult_S2));
		}

		if (VOL_S3 > 0.0)
		{
			aQ_S3 = calloc((NX * NY * NZ * NS_S3), sizeof(*aQ_S3));
			aQ_S3Mod = calloc((NX * NY * NZ * NS_S3), sizeof(*aQ_S3Mod));
			aQMult_S3 = calloc((NX * NY * NZ * NS_S3), sizeof(*aQMult_S3));
		}

		fid = fopen(W_MASK, "r");
		if (fid == NULL)
		{
			perror("Error opening the w field mask file");
			exit(EXIT_FAILURE);
		}

		aS1_mask = calloc((NX * NY * NZ), sizeof(*aS1_mask));
		aS2_mask = calloc((NX * NY * NZ), sizeof(*aS2_mask));
		aS3_mask = calloc((NX * NY * NZ), sizeof(*aS3_mask));

		for (k = 0; k < NZ; k++)
		{
			for (j = 0; j < NY; j++)
			{
				for (i = 0; i < NX; i++)
				{
					size_t idx = k + NZ * (j + NY * i);

					int uMaskScan = fscanf(fid, "%lf %lf %lf", &aS1_mask[idx], &aS2_mask[idx], &aS3_mask[idx]);
				}
			}
		}

		fclose(fid);
		printf("Mask file read.\n");
	}

	// Check whether this is a new run (iIter = 0) or a continuation of a previous run
	if (iIter == 0)
	{
		// Use the current time to set the seed of the random number generator - ensures a random start point each time the program is run

		struct timespec ts;
		clock_gettime(CLOCK_REALTIME, &ts);	
		iCurrTime = ts.tv_nsec; // nano second res for the sed

		iSeed[0] = iCurrTime;
		iSeed[1] = iCurrTime+1;
		iSeed[2] = iCurrTime+2;
		printf("iSeed new = %hu, %hu, %hu\n\n", iSeed[0], iSeed[1], iSeed[2]);
		seed48(iSeed);
		
		// Initialize the fields
		for (k = 0; k < NZ; k++)
		{
			for (j = 0; j < NY; j++)
			{
				for (i = 0; i < NX; i++)
				{
					size_t idx = k + NZ * (j + NY * i);

					aW_A[idx] = 0.0;
					aW_B[idx] = 0.0;
					aW_C[idx] = 0.0;
					aW_D[idx] = 0.0;
					aW_E[idx] = 0.0;
					aW_F[idx] = 0.0;
					aW_G[idx] = 0.0;
					aW_H[idx] = 0.0;
					aW_I[idx] = 0.0;
					aW_J[idx] = 0.0;
					aP[idx] = 0.0;

					// even if you don't use this field, the files will save the densities and fields (for consistency)
					aW_S1[idx] = 0.0;
					aW_S2[idx] = 0.0;
					aW_S3[idx] = 0.0;

					// Set total densitites arrays to zero, there will be used always when saving the files 
					aPhi_A_Tot[idx] = 0.0;
					aPhi_B_Tot[idx] = 0.0;
					aPhi_C_Tot[idx] = 0.0;
					aPhi_D_Tot[idx] = 0.0;
					aPhi_E_Tot[idx] = 0.0;
					aPhi_F_Tot[idx] = 0.0;
					aPhi_G_Tot[idx] = 0.0;
					aPhi_H_Tot[idx] = 0.0;
					aPhi_I_Tot[idx] = 0.0;
					aPhi_J_Tot[idx] = 0.0;
					aPhi_Tot[idx] = 0.0;
					aPhi_S1[idx] = 0.0;
					aPhi_S2[idx] = 0.0;
					aPhi_S3[idx] = 0.0;
				}
			}
		}
	}
	else
	{
		// Get the values of the fields to continue the simulation from a saved state
		fid = fopen(RESTART_FIELDS,"r");
		if (fid == NULL) {
			perror("Error opening restart fields file");
			exit(EXIT_FAILURE);
		}
		
		for (k = 0; k < NZ; k++)
		{
			for (j = 0; j < NY; j++)
			{
				for (i = 0; i < NX; i++)
				{
					size_t idx = k + NZ * (j + NY * i);

					int scanFields = fscanf(fid, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", 
													&aW_A[idx], &aW_B[idx], &aW_C[idx], &aW_D[idx], &aW_E[idx], &aW_F[idx], &aW_G[idx], &aW_H[idx], &aW_I[idx], &aW_J[idx], &aW_S1[idx], &aW_S2[idx], &aW_S3[idx], &aP[idx]);
				}
			}
		}
		fclose(fid);
		
		// Get the values of the state of the random number generator to continue the simulation form a saved state
		fid = fopen(RESTART_RANDOM,"r");
		if (fid == NULL) {
			perror("Error opening restart random file");
			exit(EXIT_FAILURE);
		}
		int seedsScan = fscanf(fid,"%hu %hu %hu", &iSeed[0], &iSeed[1], &iSeed[2]);
		fclose(fid);
		
		// Reset the state of the random number generator to the saved state
		seed48(iSeed);
	}
	
	// Initialize the propagators for the PDE
	for (k = 0; k < NZ; k++)
	{
		for (j = 0; j < NY; j++)
		{
			for (i = 0; i < NX; i++)
			{
				size_t idx = k + NZ * (j + NY * i);

				// Dbk 1
				if (VOL_DBK_1 > 0.0)
				{
					aQ_dbk_1[0 + NS_DBK_1 * (idx)] = 1.0;
					aQMod_dbk_1[0 + NS_DBK_1 * (idx)] = 1.0;
				}

				// Dbk 2
				if (VOL_DBK_2 > 0.0)
				{
					aQ_dbk_2[0 + NS_DBK_2 * (idx)] = 1.0;
					aQMod_dbk_2[0 + NS_DBK_2 * (idx)] = 1.0;
				}

				// Dbk 3
				if (VOL_DBK_3 > 0.0)
				{
					aQ_dbk_3[0 + NS_DBK_3 * (idx)] = 1.0;
					aQMod_dbk_3[0 + NS_DBK_3 * (idx)] = 1.0;
				}

				// Dbk 4
				if (VOL_DBK_4 > 0.0)
				{
					aQ_dbk_4[0 + NS_DBK_4 * (idx)] = 1.0;
					aQMod_dbk_4[0 + NS_DBK_4 * (idx)] = 1.0;
				}

				// Homopolymer A
				if (VOL_HOMOPOLYMER_A > 0.0)
					aQ_homo_A[0 + NS_HOMO_A * (idx)] = 1.0;

				// Homopolymer B
				if (VOL_HOMOPOLYMER_B > 0.0)
					aQ_homo_B[0 + NS_HOMO_B * (idx)] = 1.0;

				// Homopolymer C
				if (VOL_HOMOPOLYMER_C > 0.0)
					aQ_homo_C[0 + NS_HOMO_C * (idx)] = 1.0;

				// Homopolymer D
				if (VOL_HOMOPOLYMER_D > 0.0)
					aQ_homo_D[0 + NS_HOMO_D * (idx)] = 1.0;

				// Homopolymer E
				if (VOL_HOMOPOLYMER_E > 0.0)
					aQ_homo_E[0 + NS_HOMO_E * (idx)] = 1.0;

				// Homopolymer F
				if (VOL_HOMOPOLYMER_F > 0.0)
					aQ_homo_F[0 + NS_HOMO_F * (idx)] = 1.0;

				// Homopolymer G
				if (VOL_HOMOPOLYMER_G > 0.0)
					aQ_homo_G[0 + NS_HOMO_G * (idx)] = 1.0;

				// Homopolymer H
				if (VOL_HOMOPOLYMER_H > 0.0)
					aQ_homo_H[0 + NS_HOMO_H * (idx)] = 1.0;
					
				// Homopolymer I
				if (VOL_HOMOPOLYMER_I > 0.0)
					aQ_homo_I[0 + NS_HOMO_I * (idx)] = 1.0;

				// Homopolymer J
				if (VOL_HOMOPOLYMER_J > 0.0)
					aQ_homo_J[0 + NS_HOMO_J * (idx)] = 1.0;

				// Substrate
				if (MASK)
				{
					if (VOL_S1 > 0.0)
					{
						aQ_S1[0 + NS_S1 * (idx)] = 1.0;
						aQ_S1Mod[0 + NS_S1 * (idx)] = 1.0;
					}

					if (VOL_S2 > 0.0)
					{
						aQ_S2[0 + NS_S2 * (idx)] = 1.0;
						aQ_S2Mod[0 + NS_S2 * (idx)] = 1.0;
					}

					if (VOL_S3 > 0.0)
					{
						aQ_S3[0 + NS_S3 * (idx)] = 1.0;
						aQ_S3Mod[0 + NS_S3 * (idx)] = 1.0;
					}
				}
			}
		}
	}
	
	// Initialize the exponential argument for the PDE solver - stays constant throughout the simulation
	double kx, ky, kz;
	if (MASK)
	{
		// for (k = 0; k < NZ; k++)
		for (k = 0; k < N; k++)
		{
			for (j = 0; j < NY; j++)
			{
				for (i = 0; i < NX; i++)
				{
		            size_t idx = k + N * (j + NY * i);

					kx = 2.0 * M_PI * (NX / 2.0 - fabs(i - NX / 2.0)) / NX;
					ky = 2.0 * M_PI * (NY / 2.0 - fabs(j - NY / 2.0)) / NY;
					kz = 2.0 * M_PI * (N / 2.0 - fabs(k - N / 2.0)) / N;

					aExpArg[idx] = exp(- ALPHAX * pow(kx, 2) - ALPHAY * pow(ky, 2) - ALPHAZ * (pow(kz, 2)));
				}
			}
		}
	}
	else 
	{
		for (k = 0; k < NZ; k++)
		{
			for (j = 0; j < NY; j++)
			{
				for (i = 0; i < NX; i++)
				{
					size_t idx = k + NZ * (j + NY * i);

					kx = 2.0 * M_PI * (NX / 2.0 - fabs(i - NX / 2.0)) / NX;
					ky = 2.0 * M_PI * (NY / 2.0 - fabs(j - NY / 2.0)) / NY;
					kz = 2.0 * M_PI * (NZ / 2.0 - fabs(k - NZ / 2.0)) / NZ;

					aExpArg[idx] = exp(-ALPHAX * pow(kx, 2) - ALPHAY * pow(ky, 2) - ALPHAZ * pow(kz, 2));
				}
			}
		}
	}
	
	// Write the file with the run parameters
	fid = fopen(RUN_PARAMS, "a");
	fprintf(fid, "-----------------------------------------\n\n");
	fprintf(fid, "These are the run parameter values: \n\n");
	fprintf(fid, "Iteration  = %ld \n", iIter);
	fprintf(fid, "NX, NY, NZ  = %d, %d, %d \n", NX, NY, NZ);
	fprintf(fid, "NS  = %d \n\n", NS);

	fprintf(fid, "DBK_1:: NS_DBK_1, NS_DBK_1_X, NS_DBK_1_Y, LEN_DBK_1, F_INPUT_DBK_1, VOL_DBK_1  = %d, %d, %d, %f, %f, %f \n", NS_DBK_1, NS_DBK_1_X, NS_DBK_1_Y, LEN_DBK_1, F_INPUT_DBK_1, VOL_DBK_1);
	fprintf(fid, "DBK_2:: NS_DBK_2, NS_DBK_2_X, NS_DBK_2_Y, LEN_DBK_2, F_INPUT_DBK_2, VOL_DBK_2  = %d, %d, %d, %f, %f, %f \n", NS_DBK_2, NS_DBK_2_X, NS_DBK_2_Y, LEN_DBK_2, F_INPUT_DBK_2, VOL_DBK_2);
	fprintf(fid, "DBK_3:: NS_DBK_3, NS_DBK_3_X, NS_DBK_3_Y, LEN_DBK_3, F_INPUT_DBK_3, VOL_DBK_3  = %d, %d, %d, %f, %f, %f \n", NS_DBK_3, NS_DBK_3_X, NS_DBK_3_Y, LEN_DBK_3, F_INPUT_DBK_3, VOL_DBK_3);
	fprintf(fid, "DBK_4:: NS_DBK_4, NS_DBK_4_X, NS_DBK_4_Y, LEN_DBK_4, F_INPUT_DBK_4, VOL_DBK_4  = %d, %d, %d, %f, %f, %f \n", NS_DBK_4, NS_DBK_4_X, NS_DBK_4_Y, LEN_DBK_4, F_INPUT_DBK_4, VOL_DBK_4);

	fprintf(fid, "HOMOPOLYMER A:: NS_HOMO_A, HOMOPOLYMER_LEN_A, VOL_HOMOPOLYMER_A  = %d, %f, %f \n", NS_HOMO_A, LEN_HOMO_A, VOL_HOMOPOLYMER_A);
	fprintf(fid, "HOMOPOLYMER B:: NS_HOMO_B, HOMOPOLYMER_LEN_B, VOL_HOMOPOLYMER_B  = %d, %f, %f \n", NS_HOMO_B, LEN_HOMO_B, VOL_HOMOPOLYMER_B);
	fprintf(fid, "HOMOPOLYMER C:: NS_HOMO_C, HOMOPOLYMER_LEN_C, VOL_HOMOPOLYMER_C  = %d, %f, %f \n", NS_HOMO_C, LEN_HOMO_C, VOL_HOMOPOLYMER_C);
	fprintf(fid, "HOMOPOLYMER D:: NS_HOMO_D, HOMOPOLYMER_LEN_D, VOL_HOMOPOLYMER_D  = %d, %f, %f \n", NS_HOMO_D, LEN_HOMO_D, VOL_HOMOPOLYMER_D);
	fprintf(fid, "HOMOPOLYMER E:: NS_HOMO_E, HOMOPOLYMER_LEN_E, VOL_HOMOPOLYMER_E  = %d, %f, %f \n", NS_HOMO_E, LEN_HOMO_E, VOL_HOMOPOLYMER_E);
	fprintf(fid, "HOMOPOLYMER F:: NS_HOMO_F, HOMOPOLYMER_LEN_F, VOL_HOMOPOLYMER_F  = %d, %f, %f \n", NS_HOMO_F, LEN_HOMO_F, VOL_HOMOPOLYMER_F);
	fprintf(fid, "HOMOPOLYMER G:: NS_HOMO_G, HOMOPOLYMER_LEN_G, VOL_HOMOPOLYMER_G  = %d, %f, %f \n", NS_HOMO_G, LEN_HOMO_G, VOL_HOMOPOLYMER_G);
	fprintf(fid, "HOMOPOLYMER H:: NS_HOMO_H, HOMOPOLYMER_LEN_H, VOL_HOMOPOLYMER_H  = %d, %f, %f \n", NS_HOMO_H, LEN_HOMO_H, VOL_HOMOPOLYMER_H);
	fprintf(fid, "HOMOPOLYMER I:: NS_HOMO_I, HOMOPOLYMER_LEN_I, VOL_HOMOPOLYMER_I  = %d, %f, %f \n", NS_HOMO_I, LEN_HOMO_I, VOL_HOMOPOLYMER_I);
	fprintf(fid, "HOMOPOLYMER J:: NS_HOMO_J, HOMOPOLYMER_LEN_J, VOL_HOMOPOLYMER_J  = %d, %f, %f \n\n", NS_HOMO_J, LEN_HOMO_J, VOL_HOMOPOLYMER_J);

	fprintf(fid, "G1 = %f, G2 = %f, G3 = %f, G4 = %f, G5 = %f, G6 = %f, G7 = %f, G8 = %f, G9 = %f, G10 = %f, G11 = %f, G12 = %f, G13 = %f \n\n", G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12, G13);

	fprintf(fid, "phi_A_avg = %lf, phi_B_avg = %lf, phi_C_avg = %lf, phi_D_avg = %lf, phi_E_avg = %lf, phi_F_avg = %lf, phi_G_avg = %lf, phi_H_avg = %lf, phi_I_avg = %lf, phi_J_avg = %lf, phi_S1_avg = %lf, phi_S2_avg = %lf, phi_S3_avg = %lf \n\n", 
					phi_avgs[0], phi_avgs[1], phi_avgs[2], phi_avgs[3], phi_avgs[4], phi_avgs[5], phi_avgs[6], phi_avgs[7], phi_avgs[8], phi_avgs[9], phi_avgs[10], phi_avgs[11], phi_avgs[12]);

	fprintf(fid, "Chi*N As, AB = %f, AC = %f, AD = %f, AE = %f, AF = %f, AG = %f, AH = %f, AI = %f, AJ = %f \n", CHI_N_AB_EFF, CHI_N_AC_EFF, CHI_N_AD_EFF, CHI_N_AE_EFF, CHI_N_AF_EFF, CHI_N_AG_EFF, CHI_N_AH_EFF, CHI_N_AI_EFF, CHI_N_AJ_EFF);
	fprintf(fid, "Chi*N Bs, BC = %f, BD = %f, BE = %f, BF = %f, BG = %f, BH = %f, BI = %f, BJ = %f \n", CHI_N_BC_EFF, CHI_N_BD_EFF, CHI_N_BE_EFF, CHI_N_BF_EFF, CHI_N_BG_EFF, CHI_N_BH_EFF, CHI_N_BI_EFF, CHI_N_BJ_EFF);
	fprintf(fid, "Chi*N Cs, CD = %f, CE = %f, CF = %f, CG = %f, CH = %f, CI = %f, CJ = %f \n", CHI_N_CD_EFF, CHI_N_CE_EFF, CHI_N_CF_EFF, CHI_N_CG_EFF, CHI_N_CH_EFF, CHI_N_CI_EFF, CHI_N_CJ_EFF);
	fprintf(fid, "Chi*N Ds, DE = %f, DF = %f, DG = %f, DH = %f, DI = %f, DJ = %f \n", CHI_N_DE_EFF, CHI_N_DF_EFF, CHI_N_DG_EFF, CHI_N_DH_EFF, CHI_N_DI_EFF, CHI_N_DJ_EFF);
	fprintf(fid, "Chi*N Es, EF = %f, EG = %f, EH = %f, EI = %f, EJ = %f \n", CHI_N_EF_EFF, CHI_N_EG_EFF, CHI_N_EH_EFF, CHI_N_EI_EFF, CHI_N_EJ_EFF);
	fprintf(fid, "Chi*N Fs, FG = %f, FH = %f, FI = %f, FJ = %f \n", CHI_N_FG_EFF, CHI_N_FH_EFF, CHI_N_FI_EFF, CHI_N_FJ_EFF);
	fprintf(fid, "Chi*N Gs, GH = %f, GI = %f, GJ = %f \n", CHI_N_GH_EFF, CHI_N_GI_EFF, CHI_N_GJ_EFF);
	fprintf(fid, "Chi*N Hs, HI = %f, HJ = %f \n", CHI_N_HI_EFF, CHI_N_HJ_EFF);
	fprintf(fid, "Chi*N Is, IJ = %f \n\n", CHI_N_IJ_EFF);

	fprintf(fid, "Substrate settings: \n");
	fprintf(fid, "MASK = %d, vol S[1+2+3]: %f \n", MASK, coating_vol);
	fprintf(fid, "S1: NS_S1, LEN_S1, VOL_S1  = %d, %f, %f \n", NS_S1, LEN_S1, VOL_S1);
	fprintf(fid, "S2: NS_S2, LEN_S2, VOL_S2  = %d, %f, %f \n", NS_S2, LEN_S2, VOL_S2);
	fprintf(fid, "S3: NS_S3, LEN_S3, VOL_S3  = %d, %f, %f \n\n", NS_S3, LEN_S3, VOL_S3);
	fprintf(fid, "CHI_N_S1_S2_EFF = %f, CHI_N_S1_S3_EFF = %f, CHI_N_S2_S3_EFF = %f \n", CHI_N_S1_S2_EFF, CHI_N_S1_S3_EFF, CHI_N_S2_S3_EFF);
	fprintf(fid, "CHI_N_S1_A_EFF = %f, CHI_N_S2_A_EFF = %f, CHI_N_S3_A_EFF = %f \n", CHI_N_S1_A_EFF, CHI_N_S2_A_EFF, CHI_N_S3_A_EFF);
	fprintf(fid, "CHI_N_S1_B_EFF = %f, CHI_N_S2_B_EFF = %f, CHI_N_S3_B_EFF = %f \n", CHI_N_S1_B_EFF, CHI_N_S2_B_EFF, CHI_N_S3_B_EFF);
	fprintf(fid, "CHI_N_S1_C_EFF = %f, CHI_N_S2_C_EFF = %f, CHI_N_S3_C_EFF = %f \n", CHI_N_S1_C_EFF, CHI_N_S2_C_EFF, CHI_N_S3_C_EFF);
	fprintf(fid, "CHI_N_S1_D_EFF = %f, CHI_N_S2_D_EFF = %f, CHI_N_S3_D_EFF = %f \n", CHI_N_S1_D_EFF, CHI_N_S2_D_EFF, CHI_N_S3_D_EFF);
	fprintf(fid, "CHI_N_S1_E_EFF = %f, CHI_N_S2_E_EFF = %f, CHI_N_S3_E_EFF = %f \n", CHI_N_S1_E_EFF, CHI_N_S2_E_EFF, CHI_N_S3_E_EFF);
	fprintf(fid, "CHI_N_S1_F_EFF = %f, CHI_N_S2_F_EFF = %f, CHI_N_S3_F_EFF = %f \n", CHI_N_S1_F_EFF, CHI_N_S2_F_EFF, CHI_N_S3_F_EFF);
	fprintf(fid, "CHI_N_S1_G_EFF = %f, CHI_N_S2_G_EFF = %f, CHI_N_S3_G_EFF = %f \n", CHI_N_S1_G_EFF, CHI_N_S2_G_EFF, CHI_N_S3_G_EFF);
	fprintf(fid, "CHI_N_S1_H_EFF = %f, CHI_N_S2_H_EFF = %f, CHI_N_S3_H_EFF = %f \n", CHI_N_S1_H_EFF, CHI_N_S2_H_EFF, CHI_N_S3_H_EFF);
	fprintf(fid, "CHI_N_S1_I_EFF = %f, CHI_N_S2_I_EFF = %f, CHI_N_S3_I_EFF = %f \n", CHI_N_S1_I_EFF, CHI_N_S2_I_EFF, CHI_N_S3_I_EFF);
	fprintf(fid, "CHI_N_S1_J_EFF = %f, CHI_N_S2_J_EFF = %f, CHI_N_S3_J_EFF = %f \n\n", CHI_N_S1_J_EFF, CHI_N_S2_J_EFF, CHI_N_S3_J_EFF);

	fprintf(fid, "DS  = %f \n", DS);
	fprintf(fid, "LX, LY, LZ  = %f, %f, %f \n", LX, LY, LZ);
	fprintf(fid, "DX, DY, DZ  = %f, %f, %f \n\n", DX, DY, DZ);
	fprintf(fid, "-----------------------------------------\n\n");
	fclose(fid);

	// This the main program loop
	while (iIter <= ITER)
	{
		iIter++;

		// Solving for Homopolymer A
		if (NS_HOMO_A > 0) {
			iStart = 0;
			iStop = NS_HOMO_A - 1;
			NS_EFF = NS_HOMO_A;
			if (iIter == 1) printf("NS_EFF = %d, for homopolymer A. \n", NS_EFF);

						if (iIter == 1) printf("Solving homopolymer A from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_homo_A, aW_A, pFFTPlanForward, pFFTPlanReverse, 0);
			funChemPotential(&chemPot_homo_A, aQ_homo_A, &Qmod_homo_A, &Qsum_homo_A);
			funMultiplyQs(aQ_homo_A, aQ_homo_A, aQMult_homo_A);
						if (iIter == 1) printf("Computing density for homopolymer A from %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_homo_A, aQMult_homo_A, Qsum_homo_A);
		}

		// Solving for Homopolymer B
		if (NS_HOMO_B > 0) {
			iStart = 0;
			iStop = NS_HOMO_B - 1;
			NS_EFF = NS_HOMO_B;
			if (iIter == 1) printf("NS_EFF = %d, for homopolymer B. \n", NS_EFF);

						if (iIter == 1) printf("Solving homopolymer B from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_homo_B, aW_B, pFFTPlanForward, pFFTPlanReverse, 0);
			funChemPotential(&chemPot_homo_B, aQ_homo_B, &Qmod_homo_B, &Qsum_homo_B);
			funMultiplyQs(aQ_homo_B, aQ_homo_B, aQMult_homo_B);
						if (iIter == 1) printf("Computing density for homopolymer B from %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_homo_B, aQMult_homo_B, Qsum_homo_B);
		}

		// Solving for Homopolymer C
		if (NS_HOMO_C > 0) {
			iStart = 0;
			iStop = NS_HOMO_C - 1;
			NS_EFF = NS_HOMO_C;
			if (iIter == 1) printf("NS_EFF = %d, for homopolymer C. \n", NS_EFF);

						if (iIter == 1) printf("Solving homopolymer C from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_homo_C, aW_C, pFFTPlanForward, pFFTPlanReverse, 0);
			funChemPotential(&chemPot_homo_C, aQ_homo_C, &Qmod_homo_C, &Qsum_homo_C);
			funMultiplyQs(aQ_homo_C, aQ_homo_C, aQMult_homo_C);
						if (iIter == 1) printf("Computing density for homopolymer C from %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_homo_C, aQMult_homo_C, Qsum_homo_C);
		}

		// Solving for Homopolymer D
		if (NS_HOMO_D > 0) {
			iStart = 0;
			iStop = NS_HOMO_D - 1;
			NS_EFF = NS_HOMO_D;
			if (iIter == 1) printf("NS_EFF = %d, for homopolymer D. \n", NS_EFF);

						if (iIter == 1) printf("Solving homopolymer D from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_homo_D, aW_D, pFFTPlanForward, pFFTPlanReverse, 0);
			funChemPotential(&chemPot_homo_D, aQ_homo_D, &Qmod_homo_D, &Qsum_homo_D);
			funMultiplyQs(aQ_homo_D, aQ_homo_D, aQMult_homo_D);
						if (iIter == 1) printf("Computing density for homopolymer D from %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_homo_D, aQMult_homo_D, Qsum_homo_D);
		}

		// Solving for Homopolymer E
		if (NS_HOMO_E > 0) {
			iStart = 0;
			iStop = NS_HOMO_E - 1;
			NS_EFF = NS_HOMO_E;
			if (iIter == 1) printf("NS_EFF = %d, for homopolymer E. \n", NS_EFF);

						if (iIter == 1) printf("Solving homopolymer E from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_homo_E, aW_E, pFFTPlanForward, pFFTPlanReverse, 0);
			funChemPotential(&chemPot_homo_E, aQ_homo_E, &Qmod_homo_E, &Qsum_homo_E);
			funMultiplyQs(aQ_homo_E, aQ_homo_E, aQMult_homo_E);
						if (iIter == 1) printf("Computing density for homopolymer E from %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_homo_E, aQMult_homo_E, Qsum_homo_E);
		}

		// Solving for Homopolymer F
		if (NS_HOMO_F > 0) {
			iStart = 0;
			iStop = NS_HOMO_F - 1;
			NS_EFF = NS_HOMO_F;
			if (iIter == 1) printf("NS_EFF = %d, for homopolymer F. \n", NS_EFF);

						if (iIter == 1) printf("Solving homopolymer F from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_homo_F, aW_F, pFFTPlanForward, pFFTPlanReverse, 0);
			funChemPotential(&chemPot_homo_F, aQ_homo_F, &Qmod_homo_F, &Qsum_homo_F);
			funMultiplyQs(aQ_homo_F, aQ_homo_F, aQMult_homo_F);
						if (iIter == 1) printf("Computing density for homopolymer F from %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_homo_F, aQMult_homo_F, Qsum_homo_F);
		}

		// Solving for Homopolymer G
		if (NS_HOMO_G > 0) {
			iStart = 0;
			iStop = NS_HOMO_G - 1;
			NS_EFF = NS_HOMO_G;
			if (iIter == 1) printf("NS_EFF = %d, for homopolymer G. \n", NS_EFF);

						if (iIter == 1) printf("Solving homopolymer G from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_homo_G, aW_G, pFFTPlanForward, pFFTPlanReverse, 0);
			funChemPotential(&chemPot_homo_G, aQ_homo_G, &Qmod_homo_G, &Qsum_homo_G);
			funMultiplyQs(aQ_homo_G, aQ_homo_G, aQMult_homo_G);
						if (iIter == 1) printf("Computing density for homopolymer G from %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_homo_G, aQMult_homo_G, Qsum_homo_G);
		}

		// Solving for Homopolymer H
		if (NS_HOMO_H > 0) {
			iStart = 0;
			iStop = NS_HOMO_H - 1;
			NS_EFF = NS_HOMO_H;
			if (iIter == 1) printf("NS_EFF = %d, for homopolymer H. \n", NS_EFF);

						if (iIter == 1) printf("Solving homopolymer H from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_homo_H, aW_H, pFFTPlanForward, pFFTPlanReverse, 0);
			funChemPotential(&chemPot_homo_H, aQ_homo_H, &Qmod_homo_H, &Qsum_homo_H);
			funMultiplyQs(aQ_homo_H, aQ_homo_H, aQMult_homo_H);
						if (iIter == 1) printf("Computing density for homopolymer H from %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_homo_H, aQMult_homo_H, Qsum_homo_H);
		}

		// Solving for Homopolymer I
		if (NS_HOMO_I > 0) {
			iStart = 0;
			iStop = NS_HOMO_I - 1;
			NS_EFF = NS_HOMO_I;
			if (iIter == 1) printf("NS_EFF = %d, for homopolymer I. \n", NS_EFF);

						if (iIter == 1) printf("Solving homopolymer I from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_homo_I, aW_I, pFFTPlanForward, pFFTPlanReverse, 0);
			funChemPotential(&chemPot_homo_I, aQ_homo_I, &Qmod_homo_I, &Qsum_homo_I);
			funMultiplyQs(aQ_homo_I, aQ_homo_I, aQMult_homo_I);
						if (iIter == 1) printf("Computing density for homopolymer I from %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_homo_I, aQMult_homo_I, Qsum_homo_I);
		}

		// Solving for Homopolymer J
		if (NS_HOMO_J > 0) {
			iStart = 0;
			iStop = NS_HOMO_J - 1;
			NS_EFF = NS_HOMO_J;
			if (iIter == 1) printf("NS_EFF = %d, for homopolymer J. \n", NS_EFF);

						if (iIter == 1) printf("Solving homopolymer J from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_homo_J, aW_J, pFFTPlanForward, pFFTPlanReverse, 0);
			funChemPotential(&chemPot_homo_J, aQ_homo_J, &Qmod_homo_J, &Qsum_homo_J);
			funMultiplyQs(aQ_homo_J, aQ_homo_J, aQMult_homo_J);
						if (iIter == 1) printf("Computing density for homopolymer J from %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_homo_J, aQMult_homo_J, Qsum_homo_J);
		}

		if (VOL_DBK_1 > 0) {
			NS_EFF = NS_DBK_1;
			if (iIter == 1) printf("NS_EFF = %d, for DBK_1. \n", NS_EFF);

			// A
			iStart = 0;
			iStop = NS_DBK_1_X - 1;
						if (iIter == 1) printf("Solving dbk1 X from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_dbk_1, aW_A, pFFTPlanForward, pFFTPlanReverse, ENABLE_SEED); // A block forward

			// B
			iStart = NS_DBK_1_X - 1;
			iStop = NS_DBK_1 - 1;
						if (iIter == 1) printf("Solving dbk1 Y from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_dbk_1, aW_B, pFFTPlanForward, pFFTPlanReverse, 0); // B block forward

			// diblock return
			// B
			iStart = 0;
			iStop = NS_DBK_1_Y;
						if (iIter == 1) printf("Solving dbk1 Y backwards from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQMod_dbk_1, aW_B, pFFTPlanForward, pFFTPlanReverse, 0); // B block reverse

			// A
			iStart = NS_DBK_1_Y;
			iStop = NS_DBK_1 - 1;
						if (iIter == 1) printf("Solving dbk1 X backwards from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQMod_dbk_1, aW_A, pFFTPlanForward, pFFTPlanReverse, 0); // C block reverse

			funChemPotential(&chemPot_dbk_1, aQMod_dbk_1, &Qmod_dbk_1, &Qsum_dbk_1);

			funMultiplyQs(aQ_dbk_1, aQMod_dbk_1, aQMult_dbk_1);

			iStart = 0; iStop = NS_DBK_1_X - 1;
						if (iIter == 1) printf("Computing density for DBK1-X %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_X_dbk_1, aQMult_dbk_1, Qsum_dbk_1);

			iStart = NS_DBK_1_X - 1; iStop = NS_DBK_1 - 1;
						if (iIter == 1) printf("Computing density for DBK1-Y %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_Y_dbk_1, aQMult_dbk_1, Qsum_dbk_1);

		}

		if (VOL_DBK_2 > 0) {
			NS_EFF = NS_DBK_2;
			if (iIter == 1) printf("NS_EFF = %d, for DBK_2. \n", NS_EFF);

			// X
			iStart = 0;
			iStop = NS_DBK_2_X - 1;
						if (iIter == 1) printf("Solving dbk2 X from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_dbk_2, aW_D, pFFTPlanForward, pFFTPlanReverse, 0); // A block forward

			// Y
			iStart = NS_DBK_2_X - 1;
			iStop = NS_DBK_2 - 1;
						if (iIter == 1) printf("Solving dbk2 Y from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_dbk_2,aW_E, pFFTPlanForward, pFFTPlanReverse, 0); // B block forward

			// diblock return
			// Y
			iStart = 0;
			iStop = NS_DBK_2_Y;
						if (iIter == 1) printf("Solving dbk2 Y backwards from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQMod_dbk_2, aW_E, pFFTPlanForward, pFFTPlanReverse, 0); // B block reverse

			// A
			iStart = NS_DBK_2_Y;
			iStop = NS_DBK_2 - 1;
						if (iIter == 1) printf("Solving dbk2 X backwards from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQMod_dbk_2, aW_D, pFFTPlanForward, pFFTPlanReverse, 0); // C block reverse

			funChemPotential(&chemPot_dbk_2, aQMod_dbk_2, &Qmod_dbk_2, &Qsum_dbk_2);

			funMultiplyQs(aQ_dbk_2, aQMod_dbk_2, aQMult_dbk_2);

			iStart = 0; iStop = NS_DBK_2_X - 1;
						if (iIter == 1) printf("Computing density for DBK2-X %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_X_dbk_2, aQMult_dbk_2, Qsum_dbk_2);

			iStart = NS_DBK_2_X - 1; iStop = NS_DBK_2 - 1;
						if (iIter == 1) printf("Computing density for DBK2-Y %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_Y_dbk_2, aQMult_dbk_2, Qsum_dbk_2);

		}

		if (VOL_DBK_3 > 0) {
			NS_EFF = NS_DBK_3;
			if (iIter == 1) printf("NS_EFF = %d, for DBK_3. \n", NS_EFF);

			// X
			iStart = 0;
			iStop = NS_DBK_3_X - 1;
						if (iIter == 1) printf("Solving dbk3 X from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_dbk_3, aW_F, pFFTPlanForward, pFFTPlanReverse, 0); // A block forward

			// Y
			iStart = NS_DBK_3_X - 1;
			iStop = NS_DBK_3 - 1;
						if (iIter == 1) printf("Solving dbk3 Y from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_dbk_3,aW_G, pFFTPlanForward, pFFTPlanReverse, 0); // B block forward

			// diblock return
			// Y
			iStart = 0;
			iStop = NS_DBK_3_Y;
						if (iIter == 1) printf("Solving dbk3 Y backwards from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQMod_dbk_3, aW_G, pFFTPlanForward, pFFTPlanReverse, 0); // B block reverse

			// A
			iStart = NS_DBK_3_Y;
			iStop = NS_DBK_3 - 1;
						if (iIter == 1) printf("Solving dbk3 X backwards from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQMod_dbk_3, aW_F, pFFTPlanForward, pFFTPlanReverse, 0); // C block reverse

			funChemPotential(&chemPot_dbk_3, aQ_dbk_3, &Qmod_dbk_3, &Qsum_dbk_3);

			funMultiplyQs(aQ_dbk_3, aQMod_dbk_3, aQMult_dbk_3);

			iStart = 0; iStop = NS_DBK_3_X - 1;
						if (iIter == 1) printf("Computing density for DBK3-X %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_X_dbk_3, aQMult_dbk_3, Qsum_dbk_3);

			iStart = NS_DBK_3_X - 1; iStop = NS_DBK_3 - 1;
						if (iIter == 1) printf("Computing density for DBK3-Y %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_Y_dbk_3, aQMult_dbk_3, Qsum_dbk_3);
		}

		if (VOL_DBK_4 > 0) {
			NS_EFF = NS_DBK_4;
			if (iIter == 1) printf("NS_EFF = %d, for DBK_4. \n", NS_EFF);

			// X
			iStart = 0;
			iStop = NS_DBK_4_X - 1;
						if (iIter == 1) printf("Solving dbk4 X from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_dbk_4, aW_H, pFFTPlanForward, pFFTPlanReverse, 0); // A block forward

			// Y
			iStart = NS_DBK_4_X - 1;
			iStop = NS_DBK_4 - 1;
						if (iIter == 1) printf("Solving dbk4 Y from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQ_dbk_4,aW_I, pFFTPlanForward, pFFTPlanReverse, 0); // B block forward

			// diblock return
			// Y
			iStart = 0;
			iStop = NS_DBK_4_Y;
						if (iIter == 1) printf("Solving dbk4 Y backwards from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQMod_dbk_4, aW_I, pFFTPlanForward, pFFTPlanReverse, 0); // B block reverse

			// A
			iStart = NS_DBK_4_Y;
			iStop = NS_DBK_4 - 1;
						if (iIter == 1) printf("Solving dbk4 X backwards from %d to %d.\n", iStart, iStop);
			funPDESolver(iStart, iStop, aExpArg, aQMod_dbk_4, aW_H, pFFTPlanForward, pFFTPlanReverse, 0); // C block reverse

			funChemPotential(&chemPot_dbk_4, aQ_dbk_4, &Qmod_dbk_4, &Qsum_dbk_4);

			funMultiplyQs(aQ_dbk_4, aQMod_dbk_4, aQMult_dbk_4);

			iStart = 0; iStop = NS_DBK_4_X - 1;
						if (iIter == 1) printf("Computing density for DBK4-X %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_X_dbk_4, aQMult_dbk_4, Qsum_dbk_4);

			iStart = NS_DBK_4_X - 1; iStop = NS_DBK_4 - 1;
						if (iIter == 1) printf("Computing density for DBK4-Y %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_Y_dbk_4, aQMult_dbk_4, Qsum_dbk_4);
		}

		// Substrates
		if (NS_S1 > 0) { // if not mask, it's set to zero in the beginning
			iStart = 0;
			iStop = NS_S1 - 1;
			NS_EFF = NS_S1;
			if (iIter == 1) printf("NS_EFF = %d, for S1. \n", NS_EFF);

						if (iIter == 1) printf("Solving S1 from %d to %d.\n", iStart, iStop);
			funPDESolverBrush(iStart, iStop, aExpArg, aQ_S1, aW_S1, pFFTPlanForward, pFFTPlanReverse, aS1_mask);
			funPDESolver(iStart, iStop, aExpArg, aQ_S1Mod, aW_S1, pFFTPlanForward, pFFTPlanReverse, 0);
			funChemPotential(&chemPot_S1, aQ_S1, &Qmod_S1, &Qsum_S1);
			funMultiplyQs(aQ_S1, aQ_S1Mod, aQMult_S1);
						if (iIter == 1) printf("Computing density for S1 from %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_S1, aQMult_S1, Qsum_S1);
		}

		if (NS_S2 > 0) {
			iStart = 0;
			iStop = NS_S2 - 1;
			NS_EFF = NS_S2;
			if (iIter == 1) printf("NS_EFF = %d, for S2. \n", NS_EFF);

						if (iIter == 1) printf("Solving S2 from %d to %d.\n", iStart, iStop);
			funPDESolverBrush(iStart, iStop, aExpArg, aQ_S2, aW_S2, pFFTPlanForward, pFFTPlanReverse, aS2_mask);
			funPDESolver(iStart, iStop, aExpArg, aQ_S2Mod, aW_S2, pFFTPlanForward, pFFTPlanReverse, 0);
			funChemPotential(&chemPot_S2, aQ_S2, &Qmod_S2, &Qsum_S2);
			funMultiplyQs(aQ_S2, aQ_S2Mod, aQMult_S2);
						if (iIter == 1) printf("Computing density for S2 from %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_S2, aQMult_S2, Qsum_S2);
		}

		if (NS_S3 > 0) {
			iStart = 0;
			iStop = NS_S3 - 1;
			NS_EFF = NS_S3;
			if (iIter == 1) printf("NS_EFF = %d, for S3. \n", NS_EFF);

						if (iIter == 1) printf("Solving S3 from %d to %d.\n", iStart, iStop);
			funPDESolverBrush(iStart, iStop, aExpArg, aQ_S3, aW_S3, pFFTPlanForward, pFFTPlanReverse, aS3_mask);
			funPDESolver(iStart, iStop, aExpArg, aQ_S3Mod, aW_S3, pFFTPlanForward, pFFTPlanReverse, 0);
			funChemPotential(&chemPot_S3, aQ_S3, &Qmod_S3, &Qsum_S3);
			funMultiplyQs(aQ_S3, aQ_S3Mod, aQMult_S3);
						if (iIter == 1) printf("Computing density for S3 from %d to %d.\n", iStart, iStop);
			funDensity(iStart, iStop, aPhi_S3, aQMult_S3, Qsum_S3);
		}

		// if vol is zero, the function won't performt the operation
		funVolFrac(VOL_DBK_1 * (1.0 - coating_vol), aPhi_X_dbk_1); // if no mask, coating_vol = 0, if mask is used check the volumes of S1, S2 and S3
		funVolFrac(VOL_DBK_1 * (1.0 - coating_vol), aPhi_Y_dbk_1);
		funVolFrac(VOL_DBK_2 * (1.0 - coating_vol), aPhi_X_dbk_2);
		funVolFrac(VOL_DBK_2 * (1.0 - coating_vol), aPhi_Y_dbk_2);
		funVolFrac(VOL_DBK_3 * (1.0 - coating_vol), aPhi_X_dbk_3);
		funVolFrac(VOL_DBK_3 * (1.0 - coating_vol), aPhi_Y_dbk_3);
		funVolFrac(VOL_DBK_4 * (1.0 - coating_vol), aPhi_X_dbk_4);
		funVolFrac(VOL_DBK_4 * (1.0 - coating_vol), aPhi_Y_dbk_4);
		funVolFrac(VOL_HOMOPOLYMER_A * (1.0 - coating_vol), aPhi_homo_A);
		funVolFrac(VOL_HOMOPOLYMER_B * (1.0 - coating_vol), aPhi_homo_B);
		funVolFrac(VOL_HOMOPOLYMER_C * (1.0 - coating_vol), aPhi_homo_C);
		funVolFrac(VOL_HOMOPOLYMER_D * (1.0 - coating_vol), aPhi_homo_D);
		funVolFrac(VOL_HOMOPOLYMER_E * (1.0 - coating_vol), aPhi_homo_E);
		funVolFrac(VOL_HOMOPOLYMER_F * (1.0 - coating_vol), aPhi_homo_F);
		funVolFrac(VOL_HOMOPOLYMER_G * (1.0 - coating_vol), aPhi_homo_G);
		funVolFrac(VOL_HOMOPOLYMER_H * (1.0 - coating_vol), aPhi_homo_H);
		funVolFrac(VOL_HOMOPOLYMER_I * (1.0 - coating_vol), aPhi_homo_I);
		funVolFrac(VOL_HOMOPOLYMER_J * (1.0 - coating_vol), aPhi_homo_J);
		if (MASK)
		{
			funVolFrac(VOL_S1, aPhi_S1);
			funVolFrac(VOL_S2, aPhi_S2);
			funVolFrac(VOL_S3, aPhi_S3);
		}

		// ---------------------------------------------------
		// Total densities for A, B, C, D, E, F, G, H, I, J
		iSign[0] = 1; iSign[1] = 1;
		funAddArray(iSign, aPhi_X_dbk_1, aPhi_homo_A, aPhi_A_Tot);
		funAddArray(iSign, aPhi_Y_dbk_1, aPhi_homo_B, aPhi_B_Tot);
		funAddArray(iSign, NULL, aPhi_homo_C, aPhi_C_Tot);
		funAddArray(iSign, aPhi_X_dbk_2, aPhi_homo_D, aPhi_D_Tot);
		funAddArray(iSign, aPhi_Y_dbk_2, aPhi_homo_E, aPhi_E_Tot);
		funAddArray(iSign, aPhi_X_dbk_3, aPhi_homo_F, aPhi_F_Tot);
		funAddArray(iSign, aPhi_Y_dbk_3, aPhi_homo_G, aPhi_G_Tot);
		funAddArray(iSign, aPhi_X_dbk_4, aPhi_homo_H, aPhi_H_Tot);
		funAddArray(iSign, aPhi_Y_dbk_4, aPhi_homo_I, aPhi_I_Tot);
		funAddArray(iSign, NULL, aPhi_homo_J, aPhi_J_Tot);

		// Total Density
		funAddArray(iSign, aPhi_A_Tot, aPhi_B_Tot, aPhi_Tot); // resets aPhi_Tot
		funAddArray(iSign, aPhi_Tot, aPhi_C_Tot, aPhi_Tot);
		funAddArray(iSign, aPhi_Tot, aPhi_D_Tot, aPhi_Tot);
		funAddArray(iSign, aPhi_Tot, aPhi_E_Tot, aPhi_Tot);
		funAddArray(iSign, aPhi_Tot, aPhi_F_Tot, aPhi_Tot);
		funAddArray(iSign, aPhi_Tot, aPhi_G_Tot, aPhi_Tot);
		funAddArray(iSign, aPhi_Tot, aPhi_H_Tot, aPhi_Tot);
		funAddArray(iSign, aPhi_Tot, aPhi_I_Tot, aPhi_Tot);
		funAddArray(iSign, aPhi_Tot, aPhi_J_Tot, aPhi_Tot);
		if (MASK)
		{
			funAddArray(iSign, aPhi_Tot, aPhi_S1, aPhi_Tot);
			funAddArray(iSign, aPhi_Tot, aPhi_S2, aPhi_Tot);
			funAddArray(iSign, aPhi_Tot, aPhi_S3, aPhi_Tot);
		}

		if (iIter > RELAXATION)
		{
			MF = 0.0;
		}
		else
		{
			MF = 1.0;
		}


		H_fe = funFreeEnergy(aW_A, aPhi_A_Tot, aW_B, aPhi_B_Tot, aW_C, aPhi_C_Tot, aW_D, aPhi_D_Tot, aW_E, aPhi_E_Tot, 
							aW_F, aPhi_F_Tot, aW_G, aPhi_G_Tot, aW_H, aPhi_H_Tot, aW_I, aPhi_I_Tot, aW_J, aPhi_J_Tot, 
							aP, aW_S1, aPhi_S1, aW_S2, aPhi_S2, aW_S3, aPhi_S3);
		
		H_fe_fraction = 0.0;
		if (VOL_DBK_1 > 0.0) H_fe_fraction += (VOL_DBK_1/ALPHA_DBK_1 * (1.0 - coating_vol)) * log(Qsum_dbk_1);
		if (VOL_DBK_2 > 0.0) H_fe_fraction += (VOL_DBK_2/ALPHA_DBK_2 * (1.0 - coating_vol)) * log(Qsum_dbk_2);
		if (VOL_DBK_3 > 0.0) H_fe_fraction += (VOL_DBK_3/ALPHA_DBK_3 * (1.0 - coating_vol)) * log(Qsum_dbk_3);
		if (VOL_DBK_4 > 0.0) H_fe_fraction += (VOL_DBK_4/ALPHA_DBK_4 * (1.0 - coating_vol)) * log(Qsum_dbk_4);
		if (VOL_HOMOPOLYMER_A > 0.0) H_fe_fraction += (VOL_HOMOPOLYMER_A/ALPHA_HOMO_A * (1.0 - coating_vol)) * log(Qsum_homo_A);
		if (VOL_HOMOPOLYMER_B > 0.0) H_fe_fraction += (VOL_HOMOPOLYMER_B/ALPHA_HOMO_B * (1.0 - coating_vol)) * log(Qsum_homo_B);
		if (VOL_HOMOPOLYMER_C > 0.0) H_fe_fraction += (VOL_HOMOPOLYMER_C/ALPHA_HOMO_C * (1.0 - coating_vol)) * log(Qsum_homo_C);
		if (VOL_HOMOPOLYMER_D > 0.0) H_fe_fraction += (VOL_HOMOPOLYMER_D/ALPHA_HOMO_D * (1.0 - coating_vol)) * log(Qsum_homo_D);
		if (VOL_HOMOPOLYMER_E > 0.0) H_fe_fraction += (VOL_HOMOPOLYMER_E/ALPHA_HOMO_E * (1.0 - coating_vol)) * log(Qsum_homo_E);
		if (VOL_HOMOPOLYMER_F > 0.0) H_fe_fraction += (VOL_HOMOPOLYMER_F/ALPHA_HOMO_F * (1.0 - coating_vol)) * log(Qsum_homo_F);
		if (VOL_HOMOPOLYMER_G > 0.0) H_fe_fraction += (VOL_HOMOPOLYMER_G/ALPHA_HOMO_G * (1.0 - coating_vol)) * log(Qsum_homo_G);
		if (VOL_HOMOPOLYMER_H > 0.0) H_fe_fraction += (VOL_HOMOPOLYMER_H/ALPHA_HOMO_H * (1.0 - coating_vol)) * log(Qsum_homo_H);
		if (VOL_HOMOPOLYMER_I > 0.0) H_fe_fraction += (VOL_HOMOPOLYMER_I/ALPHA_HOMO_I * (1.0 - coating_vol)) * log(Qsum_homo_I);
		if (VOL_HOMOPOLYMER_J > 0.0) H_fe_fraction += (VOL_HOMOPOLYMER_J/ALPHA_HOMO_J * (1.0 - coating_vol)) * log(Qsum_homo_J);
		if (MASK)
		{
			if (VOL_S1 > 0.0)
				H_fe_fraction += VOL_S1/ALPHA_S1 * log(Qsum_S1);

			if (VOL_S2 > 0.0)
				H_fe_fraction += VOL_S2/ALPHA_S2 * log(Qsum_S2);

			if (VOL_S3 > 0.0)
				H_fe_fraction += VOL_S3/ALPHA_S3 * log(Qsum_S3);
		}
		H_fe -= H_fe_fraction;


		funUpdate(aW_A, aPhi_A_Tot, aW_B, aPhi_B_Tot, aW_C, aPhi_C_Tot, aW_D, aPhi_D_Tot, aW_E, aPhi_E_Tot, 
					aW_F, aPhi_F_Tot, aW_G, aPhi_G_Tot, aW_H, aPhi_H_Tot, aW_I, aPhi_I_Tot, aW_J, aPhi_J_Tot, 
					aP, DT_NOISE, aW_S1, aPhi_S1, aW_S2, aPhi_S2, aW_S3, aPhi_S3);

		// --------------------------------------------------- Data saving

		// Compute the RMS deviation in the density
		RMSDev = 0.0;
			
		for (k = 0; k < NZ; k++)
		{
			for (j = 0; j < NY; j++)
			{
				for (i = 0; i < NX; i++)
				{
					size_t idx = k + NZ * (j + NY * i);

					RMSDev = RMSDev + pow((aPhi_Tot[idx] - 1.0), 2);
				}
			}
		}

		RMSDev = pow((RMSDev / (NX * NY * NZ)), 0.5);

		// Write the output files at pre-determined intervals
		if (fmod(iIter, D_WRT_ENERGIES) == 0)
		{
			fid = fopen(ENERGY,"a");
			fprintf(fid, "%lu %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
						iIter, chemPot_dbk_1, chemPot_dbk_2, chemPot_dbk_3, chemPot_dbk_4,
					chemPot_homo_A, chemPot_homo_B, chemPot_homo_C, chemPot_homo_D, chemPot_homo_E, chemPot_homo_F, chemPot_homo_G, chemPot_homo_H, chemPot_homo_I, chemPot_homo_J,
					chemPot_S1, chemPot_S2, chemPot_S3);
			fclose(fid);

			fid = fopen(FREE_ENERGY,"a");
			fprintf(fid,"%lu  %lf  %f \n", iIter, H_fe, MF);
			fclose(fid);

			fid = fopen(DEVIATIONS,"a");
			fprintf(fid,"%lu  %lf %lf %lf %lf \n", iIter, RMSDev, CHI_N_AB_EFF, CHI_N_AC_EFF, CHI_N_BC_EFF);
			fclose(fid);
		}

		if (fmod(iIter, D_WRT) == 0 || iIter == 1) // save densities files, always save the first iteration
		{
			int iFileNumber = 100000000 + iIter;
			char sFileName[30] = "";
			char sFileNumber[15];

			sprintf(sFileNumber, "%d", iFileNumber);
			strcat(sFileName, DENSITY);
			strcat(sFileName, sFileNumber);
			strcat(sFileName, ".dat");

			fid = fopen(sFileName, "w");

			for (k = 0; k < NZ; k++)
			{
				for (j = 0; j < NY; j++)
				{
					for (i = 0; i < NX; i++)
					{
						size_t idx = k + NZ * (j + NY * i);

						fprintf(fid, "%5d %5d %5d %10.6lf %10.6lf %10.6lf %10.6lf %10.6lf %10.6lf %10.6lf %10.6lf %10.6lf %10.6lf %10.6lf %10.6lf %10.6lf %12.6lf %12.6lf %12.6lf %12.6lf %12.6lf %12.6lf %12.6lf %12.6lf %12.6lf %12.6lf %12.6lf %12.6lf %12.6lf %12.6lf \n", i, j, k,
								aPhi_A_Tot[idx], aPhi_B_Tot[idx], aPhi_C_Tot[idx], aPhi_D_Tot[idx], aPhi_E_Tot[idx], aPhi_F_Tot[idx], aPhi_G_Tot[idx], aPhi_H_Tot[idx], aPhi_I_Tot[idx], aPhi_J_Tot[idx], aPhi_S1[idx], aPhi_S2[idx], aPhi_S3[idx],
								aW_A[idx], aW_B[idx], aW_C[idx], aW_D[idx], aW_E[idx], aW_F[idx], aW_G[idx], aW_H[idx], aW_I[idx], aW_J[idx], aW_S1[idx], aW_S2[idx], aW_S3[idx], aP[idx]);
					}
				}
			}
			
			fclose(fid);

			// Write the current iteration number
			fid = fopen(COUNTER, "w");
			fprintf(fid, "%lu", iIter);
			fclose(fid);

			// Write the current state of the fields to a file to enable restarting the simulation
			fid = fopen(RESTART_FIELDS,"w");
			for (k = 0; k < NZ; k++)
			{
				for (j = 0; j < NY; j++)
				{
					for (i = 0; i < NX; i++)
					{
						size_t idx = k + NZ * (j + NY * i);
						fprintf(fid,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf \n", 
								aW_A[idx], aW_B[idx], aW_C[idx], aW_D[idx], aW_E[idx], aW_F[idx], aW_G[idx], aW_H[idx], aW_I[idx], aW_J[idx], aW_S1[idx], aW_S2[idx], aW_S3[idx], aP[idx]);
					}
				}
			}
			fclose(fid);
			
			// Get the current state of the random number generator and reset this state
			pOldSeed = seed48(iSeed);
			iSeed[0] = *(pOldSeed);
			iSeed[1] = *(pOldSeed + 1);
			iSeed[2] = *(pOldSeed + 2);
			seed48(iSeed);
			//printf("iSeed before new save = %hu, %hu, %hu\n", iSeed[0], iSeed[1], iSeed[2]);
			
			// Write the current state of the random number generator to a file to enable restarting of the simulation
			fid = fopen(RESTART_RANDOM, "w");
			fprintf(fid, "%hu %hu %hu", *(pOldSeed), *(pOldSeed + 1), *(pOldSeed + 2));
			fclose(fid);
		}
	}
	
	fftw_destroy_plan(pFFTPlanForward);
	fftw_destroy_plan(pFFTPlanReverse);
    
	// Free the allocated memory at the end of the simulation
	fftw_free(cDataIn); // fft plan initialization
	fftw_free(cDataOut); // fft plan initialization
	free(aW_A);
	free(aW_B);
	free(aW_C);
	free(aW_D);
	free(aW_E);
	free(aW_F);
	free(aW_G);
	free(aW_H);
	free(aW_I);
	free(aW_J);
	free(aW_S1);
	free(aW_S2);
	free(aW_S3);
	free(aP);
	free(aQ_dbk_1);
	free(aQMod_dbk_1);
	free(aQMult_dbk_1);
	free(aPhi_X_dbk_1);
	free(aPhi_Y_dbk_1);
	free(aQ_dbk_2);
	free(aQMod_dbk_2);
	free(aQMult_dbk_2);
	free(aPhi_X_dbk_2);
	free(aPhi_Y_dbk_2);
	free(aQ_dbk_3);
	free(aQMod_dbk_3);
	free(aQMult_dbk_3);
	free(aPhi_X_dbk_3);
	free(aPhi_Y_dbk_3);
	free(aQ_dbk_4);
	free(aQMod_dbk_4);
	free(aQMult_dbk_4);
	free(aPhi_X_dbk_4);
	free(aPhi_Y_dbk_4);
	free(aQ_homo_A);
	free(aQMult_homo_A);
	free(aPhi_homo_A);
	free(aQ_homo_B);
	free(aQMult_homo_B);
	free(aPhi_homo_B);
	free(aQ_homo_C);
	free(aQMult_homo_C);
	free(aPhi_homo_C);
	free(aQ_homo_D);
	free(aQMult_homo_D);
	free(aPhi_homo_D);
	free(aQ_homo_E);
	free(aQMult_homo_E);
	free(aPhi_homo_E);
	free(aQ_homo_F);
	free(aQMult_homo_F);
	free(aPhi_homo_F);
	free(aQ_homo_G);
	free(aQMult_homo_G);
	free(aPhi_homo_G);
	free(aQ_homo_H);
	free(aQMult_homo_H);
	free(aPhi_homo_H);
	free(aQ_homo_I);
	free(aQMult_homo_I);
	free(aPhi_homo_I);
	free(aQ_homo_J);
	free(aQMult_homo_J);
	free(aPhi_homo_J);
	free(aQ_S1);
	free(aQ_S1Mod);
	free(aQMult_S1);
	free(aS1_mask);
	free(aPhi_S1);
	free(aQ_S2);
	free(aQ_S2Mod);
	free(aQMult_S2);
	free(aS2_mask);
	free(aPhi_S2);
	free(aQ_S3);
	free(aQ_S3Mod);
	free(aQMult_S3);
	free(aS3_mask);
	free(aPhi_S3);
	free(aPhi_A_Tot);
	free(aPhi_B_Tot);
	free(aPhi_C_Tot);
	free(aPhi_D_Tot);
	free(aPhi_E_Tot);
	free(aPhi_F_Tot);
	free(aPhi_G_Tot);
	free(aPhi_H_Tot);
	free(aPhi_I_Tot);
	free(aPhi_J_Tot);
	free(aPhi_Tot);
	free(aExpArg);
}

//////////////////////////////////////////////////////////////////////////////////////////////
//                                            The definitions of the functions follow                                                                     //
//////////////////////////////////////////////////////////////////////////////////////////////
double funGaussDev()
{
	double x1, x2, w, y;
 
	do
	{
		x1 = 2.0*drand48()-1.0;
		x2 = 2.0*drand48()-1.0;
		w = x1*x1 + x2*x2;
	} while (w >= 1.0);
	
	w = sqrt((-2.0*log(w))/w);
	return x1*w; 
}

void funVolFrac(double frac, double *arrayF)
{
	int i, j, k;

	if (frac > 0.0)
	{
		for (k = 0; k < NZ; k++)
		{
			for (j = 0; j < NY; j++)
			{
				for (i = 0; i < NX; i++)
				{
					size_t idx = k + NZ * (j + NY * i);

					arrayF[idx] = frac * arrayF[idx];
				}
			}
		}
	}
}

void funAddArray(int sign[2], double *array1, double *array2, double *sum)
{
	int i, j, k;

	if (array1 != NULL || array2 != NULL)
	{
		for (k = 0; k < NZ; k++)
		{
			for (j = 0; j < NY; j++)
			{
				for (i = 0; i < NX; i++)
				{
					size_t idx = k + NZ * (j + NY * i);

					if (array1 == NULL)
					{
						sum[idx] = array2[idx];
					}
					else if (array2 == NULL)
					{
						sum[idx] = array1[idx];
					}
					else
					{
						sum[idx] = (sign[0] * array1[idx]) + (sign[1] * array2[idx]);
					}
				}
			}
		}
	}
}

void funChemPotential(double *chempotR, double *qr, double *qmod, double *qrsum)
{
	int i,j,k;
	int iTot = NS_EFF-1;
	double dTheta;
	double dVol = NX*NY*NZ;
	
	*qrsum = 0.0;
		
	for (k = 0; k < NZ; k++)
	{
		for (j = 0; j < NY; j++)
		{
			for (i = 0; i < NX; i++)
			{
				size_t idx = NS_EFF * (k + NZ * (j + NY * i));

				*qrsum = *qrsum + qr[iTot + idx];
			}
		}
	}

	*qrsum = *qrsum / dVol;
	*qmod = pow(*qrsum, 2);

	if (*qmod == 0)
	{
		*chempotR = 0.0;
	}
	
	else {
 	   *chempotR = -log(*qmod)/2.0;
	}
	
	if (iIter == 1) printf("qrsum = %lf, qmod = %lf, chempotR = %lf\n", *qrsum, *qmod, *chempotR);
}

double funFreeEnergy(double *wfieldA, double *densityA, double *wfieldB, double *densityB, double *wfieldC, double *densityC, double *wfieldD, double *densityD, double *wfieldE, double *densityE, 
						double *wfieldF, double *densityF, double *wfieldG, double *densityG, double *wfieldH, double *densityH, double *wfieldI, double *densityI, double *wfieldJ, double *densityJ, 
							double *pfield, double *wfieldS1, double *densityS1, double *wfieldS2, double *densityS2, double *wfieldS3, double *densityS3)
{
	int i, j, k;
	double dVol = NX * NY * NZ;
	double H = 0.0;

	// From Karim's Matlab code
	double *H_int = calloc((NX * NY * NZ), sizeof(*H_int));

	for (k = 0; k < NZ; k++)
	{
		for (j = 0; j < NY; j++)
		{
			for (i = 0; i < NX; i++)
			{

				size_t idx = k + NZ * (j + NY * i);

				H_int[idx] = CHI_N_AB_EFF * densityA[idx] * densityB[idx] + CHI_N_AC_EFF * densityA[idx] * densityC[idx] + CHI_N_AD_EFF * densityA[idx] * densityD[idx] + CHI_N_AE_EFF * densityA[idx] * densityE[idx] + CHI_N_AF_EFF * densityA[idx] * densityF[idx] + CHI_N_AG_EFF * densityA[idx] * densityG[idx] + CHI_N_AH_EFF * densityA[idx] * densityH[idx] + CHI_N_AI_EFF * densityA[idx] * densityI[idx] + CHI_N_AJ_EFF * densityA[idx] * densityJ[idx] + CHI_N_S1_A_EFF * densityA[idx] * densityS1[idx] + CHI_N_S2_A_EFF * densityA[idx] * densityS2[idx] + CHI_N_S3_A_EFF * densityA[idx] * densityS3[idx] +
							CHI_N_BC_EFF * densityB[idx] * densityC[idx] + CHI_N_BD_EFF * densityB[idx] * densityD[idx] + CHI_N_BE_EFF * densityB[idx] * densityE[idx] + CHI_N_BF_EFF * densityB[idx] * densityF[idx] + CHI_N_BG_EFF * densityB[idx] * densityG[idx] + CHI_N_BH_EFF * densityB[idx] * densityH[idx] + CHI_N_BI_EFF * densityB[idx] * densityI[idx] + CHI_N_BJ_EFF * densityB[idx] * densityJ[idx] + CHI_N_S1_B_EFF * densityB[idx] * densityS1[idx] + CHI_N_S2_B_EFF * densityB[idx] * densityS2[idx] + CHI_N_S3_B_EFF * densityB[idx] * densityS3[idx] +
							CHI_N_CD_EFF * densityC[idx] * densityD[idx] + CHI_N_CE_EFF * densityC[idx] * densityE[idx] + CHI_N_CF_EFF * densityC[idx] * densityF[idx] + CHI_N_CG_EFF * densityC[idx] * densityG[idx] + CHI_N_CH_EFF * densityC[idx] * densityH[idx] + CHI_N_CI_EFF * densityC[idx] * densityI[idx] + CHI_N_CJ_EFF * densityC[idx] * densityJ[idx] + CHI_N_S1_C_EFF * densityC[idx] * densityS1[idx] + CHI_N_S2_C_EFF * densityC[idx] * densityS2[idx] + CHI_N_S3_C_EFF * densityC[idx] * densityS3[idx] +
							CHI_N_DE_EFF * densityD[idx] * densityE[idx] + CHI_N_DF_EFF * densityD[idx] * densityF[idx] + CHI_N_DG_EFF * densityD[idx] * densityG[idx] + CHI_N_DH_EFF * densityD[idx] * densityH[idx] + CHI_N_DI_EFF * densityD[idx] * densityI[idx] + CHI_N_DJ_EFF * densityD[idx] * densityJ[idx] + CHI_N_S1_D_EFF * densityD[idx] * densityS1[idx] + CHI_N_S2_D_EFF * densityD[idx] * densityS2[idx] + CHI_N_S3_D_EFF * densityD[idx] * densityS3[idx] +
							CHI_N_EF_EFF * densityE[idx] * densityF[idx] + CHI_N_EG_EFF * densityE[idx] * densityG[idx] + CHI_N_EH_EFF * densityE[idx] * densityH[idx] + CHI_N_EI_EFF * densityE[idx] * densityI[idx] + CHI_N_EJ_EFF * densityE[idx] * densityJ[idx] + CHI_N_S1_E_EFF * densityE[idx] * densityS1[idx] + CHI_N_S2_E_EFF * densityE[idx] * densityS2[idx] + CHI_N_S3_E_EFF * densityE[idx] * densityS3[idx] +
							CHI_N_FG_EFF * densityF[idx] * densityG[idx] + CHI_N_FH_EFF * densityF[idx] * densityH[idx] + CHI_N_FI_EFF * densityF[idx] * densityI[idx] + CHI_N_FJ_EFF * densityF[idx] * densityJ[idx] + CHI_N_S1_F_EFF * densityF[idx] * densityS1[idx] + CHI_N_S2_F_EFF * densityF[idx] * densityS2[idx] + CHI_N_S3_F_EFF * densityF[idx] * densityS3[idx] +
							CHI_N_GH_EFF * densityG[idx] * densityH[idx] + CHI_N_GI_EFF * densityG[idx] * densityI[idx] + CHI_N_GJ_EFF * densityG[idx] * densityJ[idx] + CHI_N_S1_G_EFF * densityG[idx] * densityS1[idx] + CHI_N_S2_G_EFF * densityG[idx] * densityS2[idx] + CHI_N_S3_G_EFF * densityG[idx] * densityS3[idx] +
							CHI_N_HI_EFF * densityH[idx] * densityI[idx] + CHI_N_HJ_EFF * densityH[idx] * densityJ[idx] + CHI_N_S1_H_EFF * densityH[idx] * densityS1[idx] + CHI_N_S2_H_EFF * densityH[idx] * densityS2[idx] + CHI_N_S3_H_EFF * densityH[idx] * densityS3[idx] +
							CHI_N_IJ_EFF * densityI[idx] * densityJ[idx] + CHI_N_S1_I_EFF * densityI[idx] * densityS1[idx] + CHI_N_S2_I_EFF * densityI[idx] * densityS2[idx] + CHI_N_S3_I_EFF * densityI[idx] * densityS3[idx] +
							CHI_N_S1_J_EFF * densityJ[idx] * densityS1[idx] + CHI_N_S2_J_EFF * densityJ[idx] * densityS2[idx] + CHI_N_S3_J_EFF * densityJ[idx] * densityS3[idx] +
							CHI_N_S1_S2_EFF * densityS1[idx] * densityS2[idx] + CHI_N_S1_S3_EFF * densityS1[idx] * densityS3[idx] + 
							CHI_N_S2_S3_EFF * densityS2[idx] * densityS3[idx] 
							- wfieldA[idx] * densityA[idx] - wfieldB[idx] * densityB[idx] - wfieldC[idx] * densityC[idx] - wfieldD[idx] * densityD[idx] - wfieldE[idx] * densityE[idx] - wfieldF[idx] * densityF[idx] - wfieldG[idx] * densityG[idx] - wfieldH[idx] * densityH[idx] - wfieldI[idx] * densityI[idx] - wfieldJ[idx] * densityJ[idx] - wfieldS1[idx] * densityS1[idx] - wfieldS2[idx] * densityS2[idx] - wfieldS3[idx] * densityS3[idx]
							- pfield[idx] * (1 - densityA[idx] - densityB[idx] - densityC[idx] - densityD[idx] - densityE[idx] - densityF[idx] - densityG[idx] - densityH[idx] - densityI[idx] - densityJ[idx] - densityS1[idx] - densityS2[idx] - densityS3[idx]);
				H = H + H_int[idx];
			}
		}
	}

	H = H / dVol;

	free(H_int);
	return H;
}

// Get average phi value for a given field index
double getPhiAvg(int fieldIndex) {
    if (fieldIndex >= 0 && fieldIndex < 13) {
        return phi_avgs[fieldIndex];
    }
    return 0.0;
}

// Get CHI_N_EFF value for interaction between two fields
double getCHI_N_EFF(int field1, int field2) {
    // Create a lookup table for CHI_N_EFF values
    double chi_table[13][13] = {
        // A     B           C           D           E           F           G           H           I           J           S1          S2          S3
/*A*/   {0.0,   CHI_N_AB_EFF, CHI_N_AC_EFF, CHI_N_AD_EFF, CHI_N_AE_EFF, CHI_N_AF_EFF, CHI_N_AG_EFF, CHI_N_AH_EFF, CHI_N_AI_EFF, CHI_N_AJ_EFF, CHI_N_S1_A_EFF, CHI_N_S2_A_EFF, CHI_N_S3_A_EFF},
/*B*/   {CHI_N_AB_EFF, 0.0,   CHI_N_BC_EFF, CHI_N_BD_EFF, CHI_N_BE_EFF, CHI_N_BF_EFF, CHI_N_BG_EFF, CHI_N_BH_EFF, CHI_N_BI_EFF, CHI_N_BJ_EFF, CHI_N_S1_B_EFF, CHI_N_S2_B_EFF, CHI_N_S3_B_EFF},
/*C*/   {CHI_N_AC_EFF, CHI_N_BC_EFF, 0.0,   CHI_N_CD_EFF, CHI_N_CE_EFF, CHI_N_CF_EFF, CHI_N_CG_EFF, CHI_N_CH_EFF, CHI_N_CI_EFF, CHI_N_CJ_EFF, CHI_N_S1_C_EFF, CHI_N_S2_C_EFF, CHI_N_S3_C_EFF},
/*D*/   {CHI_N_AD_EFF, CHI_N_BD_EFF, CHI_N_CD_EFF, 0.0,   CHI_N_DE_EFF, CHI_N_DF_EFF, CHI_N_DG_EFF, CHI_N_DH_EFF, CHI_N_DI_EFF, CHI_N_DJ_EFF, CHI_N_S1_D_EFF, CHI_N_S2_D_EFF, CHI_N_S3_D_EFF},
/*E*/   {CHI_N_AE_EFF, CHI_N_BE_EFF, CHI_N_CE_EFF, CHI_N_DE_EFF, 0.0,   CHI_N_EF_EFF, CHI_N_EG_EFF, CHI_N_EH_EFF, CHI_N_EI_EFF, CHI_N_EJ_EFF, CHI_N_S1_E_EFF, CHI_N_S2_E_EFF, CHI_N_S3_E_EFF},
/*F*/   {CHI_N_AF_EFF, CHI_N_BF_EFF, CHI_N_CF_EFF, CHI_N_DF_EFF, CHI_N_EF_EFF, 0.0,   CHI_N_FG_EFF, CHI_N_FH_EFF, CHI_N_FI_EFF, CHI_N_FJ_EFF, CHI_N_S1_F_EFF, CHI_N_S2_F_EFF, CHI_N_S3_F_EFF},
/*G*/   {CHI_N_AG_EFF, CHI_N_BG_EFF, CHI_N_CG_EFF, CHI_N_DG_EFF, CHI_N_EG_EFF, CHI_N_FG_EFF, 0.0,   CHI_N_GH_EFF, CHI_N_GI_EFF, CHI_N_GJ_EFF, CHI_N_S1_G_EFF, CHI_N_S2_G_EFF, CHI_N_S3_G_EFF},
/*H*/   {CHI_N_AH_EFF, CHI_N_BH_EFF, CHI_N_CH_EFF, CHI_N_DH_EFF, CHI_N_EH_EFF, CHI_N_FH_EFF, CHI_N_GH_EFF, 0.0,   CHI_N_HI_EFF, CHI_N_HJ_EFF, CHI_N_S1_H_EFF, CHI_N_S2_H_EFF, CHI_N_S3_H_EFF},
/*I*/   {CHI_N_AI_EFF, CHI_N_BI_EFF, CHI_N_CI_EFF, CHI_N_DI_EFF, CHI_N_EI_EFF, CHI_N_FI_EFF, CHI_N_GI_EFF, CHI_N_HI_EFF, 0.0,   CHI_N_IJ_EFF, CHI_N_S1_I_EFF, CHI_N_S2_I_EFF, CHI_N_S3_I_EFF},
/*J*/   {CHI_N_AJ_EFF, CHI_N_BJ_EFF, CHI_N_CJ_EFF, CHI_N_DJ_EFF, CHI_N_EJ_EFF, CHI_N_FJ_EFF, CHI_N_GJ_EFF, CHI_N_HJ_EFF, CHI_N_IJ_EFF, 0.0,   CHI_N_S1_J_EFF, CHI_N_S2_J_EFF, CHI_N_S3_J_EFF},
/*S1*/  {CHI_N_S1_A_EFF, CHI_N_S1_B_EFF, CHI_N_S1_C_EFF, CHI_N_S1_D_EFF, CHI_N_S1_E_EFF, CHI_N_S1_F_EFF, CHI_N_S1_G_EFF, CHI_N_S1_H_EFF, CHI_N_S1_I_EFF, CHI_N_S1_J_EFF, 0.0,   CHI_N_S1_S2_EFF, CHI_N_S1_S3_EFF},
/*S2*/  {CHI_N_S2_A_EFF, CHI_N_S2_B_EFF, CHI_N_S2_C_EFF, CHI_N_S2_D_EFF, CHI_N_S2_E_EFF, CHI_N_S2_F_EFF, CHI_N_S2_G_EFF, CHI_N_S2_H_EFF, CHI_N_S2_I_EFF, CHI_N_S2_J_EFF, CHI_N_S1_S2_EFF, 0.0,   CHI_N_S2_S3_EFF},
/*S3*/  {CHI_N_S3_A_EFF, CHI_N_S3_B_EFF, CHI_N_S3_C_EFF, CHI_N_S3_D_EFF, CHI_N_S3_E_EFF, CHI_N_S3_F_EFF, CHI_N_S3_G_EFF, CHI_N_S3_H_EFF, CHI_N_S3_I_EFF, CHI_N_S3_J_EFF, CHI_N_S1_S3_EFF, CHI_N_S2_S3_EFF, 0.0}
    };

    if (field1 >= 0 && field1 < 13 && field2 >= 0 && field2 < 13) {
        return chi_table[field1][field2];
    }
    return 0.0;
}

// Get G coefficient for interaction between two fields
double getG(int field1, int field2) {
    if (field1 == field2) {
        return G1;  // Self-interaction always uses G1
    }
    
    // All other interactions use G2 (which is G1/6.0)
    return G2;  // Since G2 through G13 are all equal to G1/6.0
}

// Helper function to check if a field should be active
bool isFieldActive(int fieldIndex) {
    switch (fieldIndex) {
        case 0:  // A
        case 1:  // B
            return VOL_DBK_1 > 0.0 || VOL_HOMOPOLYMER_A > 0.0 || VOL_HOMOPOLYMER_B > 0.0;
        case 2:  // C
            return VOL_HOMOPOLYMER_C > 0.0;
        case 3:  // D
        case 4:  // E
            return VOL_DBK_2 > 0.0 || VOL_HOMOPOLYMER_D > 0.0 || VOL_HOMOPOLYMER_E > 0.0;
        case 5:  // F
        case 6:  // G
            return VOL_DBK_3 > 0.0 || VOL_HOMOPOLYMER_F > 0.0 || VOL_HOMOPOLYMER_G > 0.0;
        case 7:  // H
        case 8:  // I
            return VOL_DBK_4 > 0.0 || VOL_HOMOPOLYMER_H > 0.0 || VOL_HOMOPOLYMER_I > 0.0;
        case 9:  // J
            return VOL_HOMOPOLYMER_J > 0.0;
        case 10: // S1
            return VOL_S1 > 0.0;
        case 11: // S2
            return VOL_S2 > 0.0;
        case 12: // S3
            return VOL_S3 > 0.0;
        default:
            return false;
    }
}

void funUpdate(double *wfieldA, double *densityA, double *wfieldB, double *densityB, double *wfieldC, double *densityC, 
               double *wfieldD, double *densityD, double *wfieldE, double *densityE, double *wfieldF, double *densityF, 
               double *wfieldG, double *densityG, double *wfieldH, double *densityH, double *wfieldI, double *densityI, 
               double *wfieldJ, double *densityJ, double *pfield, double dt, double *wfieldS1, double *densityS1, 
               double *wfieldS2, double *densityS2, double *wfieldS3, double *densityS3)
{

    // Initialize field array
    FieldInfo fields[13] = {
        {wfieldA, densityA, 0.0, 0.0, 0.0, VOL_HOMOPOLYMER_A, 0},
        {wfieldB, densityB, 0.0, 0.0, 0.0, VOL_HOMOPOLYMER_B, 0},
        {wfieldC, densityC, 0.0, 0.0, 0.0, VOL_HOMOPOLYMER_C, 0},
        {wfieldD, densityD, 0.0, 0.0, 0.0, VOL_HOMOPOLYMER_D, 0},
        {wfieldE, densityE, 0.0, 0.0, 0.0, VOL_HOMOPOLYMER_E, 0},
        {wfieldF, densityF, 0.0, 0.0, 0.0, VOL_HOMOPOLYMER_F, 0},
        {wfieldG, densityG, 0.0, 0.0, 0.0, VOL_HOMOPOLYMER_G, 0},
        {wfieldH, densityH, 0.0, 0.0, 0.0, VOL_HOMOPOLYMER_H, 0},
        {wfieldI, densityI, 0.0, 0.0, 0.0, VOL_HOMOPOLYMER_I, 0},
        {wfieldJ, densityJ, 0.0, 0.0, 0.0, VOL_HOMOPOLYMER_J, 0},
        {wfieldS1, densityS1, 0.0, 0.0, 0.0, VOL_S1, 0},
        {wfieldS2, densityS2, 0.0, 0.0, 0.0, VOL_S2, 0},
        {wfieldS3, densityS3, 0.0, 0.0, 0.0, VOL_S3, 0}
    };

    // Additional DBK volume checks
    if (VOL_DBK_1 > 0.0) {
        fields[0].active = 1;  // A
        fields[1].active = 1;  // B
    }
    if (VOL_DBK_2 > 0.0) {
        fields[3].active = 1;  // D
        fields[4].active = 1;  // E
    }
    if (VOL_DBK_3 > 0.0) {
        fields[5].active = 1;  // F
        fields[6].active = 1;  // G
    }
    if (VOL_DBK_4 > 0.0) {
        fields[7].active = 1;  // H
        fields[8].active = 1;  // I
    }

    // Mark fields as active based on volume
    int activeCount = 0;
    for (int i = 0; i < 13; i++) {
        if (fields[i].vol > 0.0 || fields[i].active) {
            fields[i].active = 1;
            activeCount++;
        }
    }

    double pAvg = 0.0;
    double dVol = NX * NY * NZ;
    
    // Main loop over grid points
    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                size_t idx = k + NZ * (j + NY * i);
                
                // Initialize field values for this grid point
                for (int f = 0; f < 13; f++) {
                    if (fields[f].active) {
                        fields[f].wInit = fields[f].wfield[idx];
                        fields[f].rho_minus_rhoavg = fields[f].density[idx] - getPhiAvg(f); // You need to implement getPhiAvg
                    }
                }

                // Calculate delW for active fields
                for (int f = 0; f < 13; f++) {
                    if (!fields[f].active) continue;
                    
                    fields[f].delW = -fields[f].wInit + pfield[idx];
                    
                    // Add CHI_N terms for interactions with other active fields
                    for (int g = 0; g < 13; g++) {
                        if (!fields[g].active || f == g) continue;
                        fields[f].delW += getCHI_N_EFF(f, g) * fields[g].rho_minus_rhoavg; // You need to implement getCHI_N_EFF
                    }
                }

				// Calculate correction factor for active fields
				double correctionFactor = 0.0;
				if (activeCount > 1) {  // Only needed if more than one field is active
				    for (int f = 0; f < 13; f++) {
				        if (!fields[f].active) continue;

				        for (int g = f + 1; g < 13; g++) {  // Start from f+1 to avoid counting twice
				            if (!fields[g].active) continue;

				            // For each pair of active fields (f,g), add their contribution to inactive fields
				            for (int h = 0; h < 13; h++) {
				                if (fields[h].active && h != f && h != g) {
				                    correctionFactor += getCHI_N_EFF(f, g) * fields[h].rho_minus_rhoavg;
				                }
				            }
				        }
				    }
				}

                // Update fields and calculate pfield
                double pfield_sum = 0.0;
                for (int f = 0; f < 13; f++) {
                    if (fields[f].active) {
                        fields[f].wfield[idx] = fields[f].wInit;
                        for (int g = 0; g < 13; g++) {
                            if (fields[g].active) {
                                fields[f].wfield[idx] += getG(f, g) * fields[g].delW; // You need to implement getG
                            }
                        }
                        fields[f].wfield[idx] += dt * MF * funGaussDev();
                        pfield_sum += fields[f].wfield[idx];
                    } 
                }

                // Update pfield
                pfield[idx] = (pfield_sum + correctionFactor) / activeCount;
                pAvg += pfield[idx];

				if (iIter == 1 && idx == 0) {
					printActiveFieldUpdate(fields, activeCount, idx, correctionFactor);
				}
            }
        }
    }

    // Final pfield adjustment
    pAvg /= dVol;
    for (size_t idx = 0; idx < NX * NY * NZ; idx++) {
        pfield[idx] -= pAvg;
    }
}

void printActiveFieldUpdate(FieldInfo *fields, int activeCount, size_t idx, double correctionFactor) {
    printf("\n=== Active Field Update at Index %zu ===\n", idx);
    
    // Print active fields and their values
    printf("\n=== Active Fields ===\n");
    for (int f = 0; f < 13; f++) {
        if (fields[f].active) {
            if (f < 10) {
                printf("Field %c: wInit = %.6f, rho-rhoAvg = %.6f - %.6f = %.6f\n",
                       'A' + f, fields[f].wInit, fields[f].density[idx], getPhiAvg(f), fields[f].rho_minus_rhoavg);
            } else {
                printf("Field S%d: wInit = %.6f, rho-rhoAvg = %.6f - %.6f = %.6f\n",
                       f - 9, fields[f].wInit, fields[f].density[idx], getPhiAvg(f), fields[f].rho_minus_rhoavg);
            }
        }
    }
    
    // Print delW calculations for active fields
    printf("\n=== delW Calculations ===\n");
    for (int f = 0; f < 13; f++) {
        if (fields[f].active) {
            if (f < 10) {
                printf("\nField %c delW = ", 'A' + f);
            } else {
                printf("\nField S%d delW = ", f - 9);
            }
            
            for (int g = 0; g < 13; g++) {
                if (fields[g].active && f != g) {
                    if (g < 10) {
                        printf("+ %.6f * (rho%c-rhoAvg%c) ",
                               getCHI_N_EFF(f, g), 'A' + g, 'A' + g);
                    } else {
                        printf("+ %.6f * (rhoS%d-rhoAvgS%d) ",
                               getCHI_N_EFF(f, g), g - 9, g - 9);
                    }
                }
            }
            if (f < 10) {
                printf("+ pfield - wInit_%c = %.6f\n", 'A' + f, fields[f].delW);
            } else {
                printf("+ pfield - wInit_S%d = %.6f\n", f - 9, fields[f].delW);
            }
        }
    }
    
	printf("\n=== Correction Factor Calculation ===\n");
	printf("correctionFactor =");
	for (int f = 0; f < 13; f++) {
	    if (!fields[f].active) continue;
	
	    for (int g = f + 1; g < 13; g++) {
	        if (!fields[g].active) continue;
	
	        printf("\n  + CHI_N_%s%s_EFF * (",
	               (f < 10) ? (char[]){(char)('A' + f), '\0'} : (char[]){'S', (char)('1' + f - 10), '\0'},
	               (g < 10) ? (char[]){(char)('A' + g), '\0'} : (char[]){'S', (char)('1' + g - 10), '\0'});
	
	        int termCount = 0;
	        for (int h = 0; h < 13; h++) {
	            if (fields[h].active && h != f && h != g) {
	                if (termCount > 0) printf(" + ");
	                if (h < 10) {
	                    printf("(rho%c-rhoAvg%c)", 'A' + h, 'A' + h);
	                } else {
	                    printf("(rhoS%d-rhoAvgS%d)", h - 9, h - 9);
	                }
	                termCount++;
	            }
	        }
	        printf(")");
	    }
	}
	printf(" = %.6f\n", correctionFactor);
    
    // Print field updates
    printf("\n=== Field Updates ===\n");
    for (int f = 0; f < 13; f++) {
        if (fields[f].active) {
            if (f < 10) {
                printf("wfield%c = wInit_%c", 'A' + f, 'A' + f);
            } else {
                printf("wfieldS%d = wInit_S%d", f - 9, f - 9);
            }
            
            for (int g = 0; g < 13; g++) {
                if (fields[g].active) {
                    if (g < 10) {
                        printf(" + G%d * delW_%c",
                               (f == g) ? 1 : 2, 'A' + g);
                    } else {
                        printf(" + G%d * delW_S%d",
                               (f == g) ? 1 : 2, g - 9);
                    }
                }
            }
            printf(" + dt*MF*GaussDev\n");
        }
    }
    
    // Print pfield computation
    printf("\n=== Pressure Field Update ===\n");
    printf("pfield = (");
    for (int f = 0; f < 13; f++) {
        if (fields[f].active) {
            if (f < 10) {
                printf("wfield%c + ", 'A' + f);
            } else {
                printf("wfieldS%d + ", f - 9);
            }
        }
    }
    printf("correctionFactor) / %d\n", activeCount);
    
    printf("\n=====================================\n");
}

void funMultiplyQs(double *qr, double *qrmod, double *qchain)
{
	int i,j,k,chain;
	int iTot = NS_EFF-1;
	
	for (k = 0; k < NZ; k++)
	{
		for (j = 0; j < NY; j++)
		{
			for (i = 0; i < NX; i++)
			{
				size_t idx = NS_EFF * (k + NZ * (j + NY * i));

				for (chain = 0; chain < NS_EFF; chain++)
				{
					qchain[chain + idx] = qr[chain + idx] * qrmod[(iTot - chain) + idx];
					/*
						qr +============>
								 			<============+ qrmod
					*/
					//printf("qchain[%d,%d,%d,%d] = %lf\n", i, j, k, chain, qchain[chain + idx]);
				}
			}
		}
	}
}

void funDensity(int start, int stop, double *phir, double *qchain, double qrsum)
{
	int i, j, k, index = 0;
	double phi = 0.0;
	double DS_EFF = 1.0 / (NS_EFF - 1);

	for (k = 0; k < NZ; k++)
	{
		for (j = 0; j < NY; j++)
		{
			for (i = 0; i < NX; i++)
			{
				size_t idx = (k + NZ * (j + NY * i));
				phi = 0.0;

				phi = qchain[start + NS_EFF * idx] + qchain[stop + NS_EFF * idx];
				if (iIter == 1 && i == 0 && j == 0 && k == 0) printf("phi[%d,%d,%d,%d] = %lf\n", i, j, k, index, phi);

				for (index = start+1; index <= (stop - 1); index = index+2)
				{
					phi = phi + 4.0 * qchain[index + NS_EFF * idx];
					if (iIter == 1 && i == 0 && j == 0 && k == 0) printf("phi[%d,%d,%d,%d] = %lf\n", i, j, k, index, phi);
				}

				for (index = start+2; index <= (stop- 2); index = index+2)
				{
					phi = phi + 2.0 * qchain[index + NS_EFF * idx];
					if (iIter == 1 && i == 0 && j == 0 && k == 0) printf("phi[%d,%d,%d,%d] = %lf\n", i, j, k, index, phi);
				}

				phi = phi / 3.0;
				if (iIter == 1 && i == 0 && j == 0 && k == 0) printf("phi[%d,%d,%d,%d] / 3.0 = %lf\n", i, j, k, index, phi);

				phir[idx] = phi * (DS_EFF / qrsum); // Integrate over the normalized chain 
				if (iIter == 1 && i == 0 && j == 0 && k == 0) printf("phir[%d,%d,%d,%d] = %lf\n", i, j, k, index, phir[idx]);
			}
		}
	}
}

void funPDESolver(int start, int stop, double *exparg, double *qr, double *wfield, fftw_plan forwardplan, fftw_plan reverseplan, int seedBool)
{
	int i, j, k, index;
	double dVol;

	fftw_complex *indata;
	fftw_complex *outdata;
	fftw_complex *indata_mirrored;
	fftw_complex *aExpField;

	// allocating arrays once
	indata = (fftw_complex *)fftw_malloc(NX * NY * NZ * sizeof(fftw_complex));
	aExpField = (fftw_complex *)fftw_malloc(NX * NY * NZ * sizeof(fftw_complex));
	if (MASK)
	{
		dVol = NX * NY * N;
		outdata = (fftw_complex *)fftw_malloc(NX * NY * N * sizeof(fftw_complex));
		indata_mirrored = (fftw_complex *)fftw_malloc(NX * NY * N * sizeof(fftw_complex));
	}
	else
	{
		dVol = NX * NY * NZ;
		outdata = (fftw_complex *)fftw_malloc(NX * NY * NZ * sizeof(fftw_complex));
	}

	// Initial conditions
	for (k = 0; k < NZ; k++)
	{
		for (j = 0; j < NY; j++)
		{
			for (i = 0; i < NX; i++)
			{
				size_t idx = k + NZ * (j + NY * i);

				aExpField[idx] = exp(DS * (-wfield[idx] / 2.0) * (1 + 0 * I));

				if (seedBool == 1 && iIter <= SEED_STEPS)
				{
					if (i > ((NX / 2) - SEED_AREA_DELTA) && i < ((NX / 2) + SEED_AREA_DELTA) && j > ((NY / 2) - SEED_AREA_DELTA) && j < ((NY / 2) + SEED_AREA_DELTA) && k > ((NZ / 4) - SEED_AREA_DELTA) && k < ((NZ / 4) + SEED_AREA_DELTA))
					{
						qr[start + NS_EFF * (idx)] = 1.0;
						indata[idx] = (qr[start + NS_EFF * (idx)] + 0 * I);
						if (iIter == 1)
						{
							printf("Seeded at [%d,%d,%d]\n", i, j, k);
						}
					}
					else
					{
						qr[start + NS_EFF * (idx)] = 0.0;
						indata[idx] = (qr[start + NS_EFF * (idx)] + 0 * I);
					}
				}
				else if (seedBool == 1 && iIter > SEED_STEPS)
				{
					qr[start + NS_EFF * (idx)] = 1.0;
					indata[idx] = (qr[start + NS_EFF * (idx)] + 0 * I);
				}
				else
				{
					indata[idx] = (qr[start + NS_EFF * (idx)] + 0 * I);
				}
			}
		}
	}

	// Propagating
	for (index = start; index < stop; index++)
	{
		if (MASK)
		{
			funPDESolver_DirichletBC(aExpField, exparg, forwardplan, reverseplan, indata, indata_mirrored, outdata);

		}
		else
		{
			funPDESolver_periodic(aExpField, exparg, forwardplan, reverseplan, indata, outdata);
		}

		for (k = 0; k < NZ; k++)
		{
			for (j = 0; j < NY; j++)
			{
				for (i = 0; i < NX; i++)
				{
		            size_t idx = k + NZ * (j + NY * i);

					indata[idx] = indata[idx] / dVol;
					qr[(index + 1) + NS_EFF * (idx)] = creal(indata[idx]);
				}
			}
		}
	}

	fftw_free(indata);
	fftw_free(aExpField);
	fftw_free(outdata);

	if (MASK)
	{
		fftw_free(indata_mirrored);
	}
}

void funPDESolverBrush(int start, int stop, double *exparg, double *qr, double *wfield, fftw_plan forwardplan, fftw_plan reverseplan, double *mask)
{
	int i, j, k, index;
	double dVol = NX * NY * N;

	fftw_complex *indata;
	fftw_complex *outdata;
	fftw_complex *indata_mirrored;
	fftw_complex *aExpField;

	// allocating arrays once
	indata = (fftw_complex *)fftw_malloc(NX * NY * NZ * sizeof(fftw_complex));
	aExpField = (fftw_complex *)fftw_malloc(NX * NY * NZ * sizeof(fftw_complex));
	outdata = (fftw_complex *)fftw_malloc(NX * NY * N * sizeof(fftw_complex));
	indata_mirrored = (fftw_complex *)fftw_malloc(NX * NY * N * sizeof(fftw_complex));

	// Initial conditions
	for (k = 0; k < NZ; k++)
	{
		for (j = 0; j < NY; j++)
		{
			for (i = 0; i < NX; i++)
			{
				size_t idx = k + NZ * (j + NY * i);

				aExpField[idx] = exp(DS * (-wfield[idx] / 2.0) * (1 + 0 * I));

				qr[start + NS_EFF * (idx)] = mask[idx];

				indata[idx] = (qr[start + NS_EFF * (idx)] + 0 * I);
			}
		}
	}

	// Propagating
	for (index = start; index < stop; index++)
	{
		funPDESolver_DirichletBC(aExpField, exparg, forwardplan, reverseplan, indata, indata_mirrored, outdata);

		for (k = 0; k < NZ; k++)
		{
			for (j = 0; j < NY; j++)
			{
				for (i = 0; i < NX; i++)
				{
		            size_t idx = k + NZ * (j + NY * i);

					indata[idx] = indata[idx] / dVol;
					qr[(index + 1) + NS_EFF * (idx)] = creal(indata[idx]);
				}
			}
		}
	}
	
	fftw_free(indata);
	fftw_free(outdata);
	fftw_free(aExpField);
	fftw_free(indata_mirrored);
}

void funPDESolver_periodic(fftw_complex *expfield, double *exparg, fftw_plan forwardplan, fftw_plan reverseplan, fftw_complex *indata, fftw_complex *outdata)
{
	int i, j, k;

	// 1. Multiply by the exponential field
	for (k = 0; k < NZ; k++)
	{
		for (j = 0; j < NY; j++)
		{
			for (i = 0; i < NX; i++)
			{
	           	size_t idx = k + NZ * (j + NY * i);

				indata[idx] = indata[idx] * expfield[idx];
			}
		}
	}
			
	// 2. Forward FFT
	fftw_execute_dft(forwardplan, indata, outdata);

	// 3. Multiply by the exponential argument
	for (k = 0; k < NZ; k++)
	{
		for (j = 0; j < NY; j++)
		{
			for (i = 0; i < NX; i++)
			{
	            size_t idx = k + NZ * (j + NY * i);

				indata[idx] = outdata[idx] * exparg[idx];
			}
		}
	}

	// 4. Inverse FFT
	fftw_execute_dft(reverseplan, indata, outdata);

	// 5. Multiplying by the exponential field
	for (k = 0; k < NZ; k++)
	{
		for (j = 0; j < NY; j++)
		{
			for (i = 0; i < NX; i++)
			{
	            size_t idx = k + NZ * (j + NY * i);

				indata[idx] = outdata[idx] * expfield[idx];
			}
		}
	}
}

void funPDESolver_DirichletBC(fftw_complex *expfield, double *exparg, fftw_plan forwardplan, fftw_plan reverseplan, fftw_complex *indata, fftw_complex *indata_expanded, fftw_complex *outdata_expanded)
{
	int i, j, k;

	// 1. Multiply by the exponential field and expanding for the Dirichlet BC
	// 1.1 with the indata array expanded
	size_t idx;
	size_t original_idx;

	/*  The array is expanded in the following way for Dirichlet BC:
 	*                    /\          /\
 	*                   /  \    /\  /  \
 	*                  /    \  /  \/    \    /\
 	*                 /      \/          \  /  \
 	*         (zero) /                    \/    \ (zero)
	*                                                   \    /\                    /
	*                                                    \  /  \          /\      /
	*                                                     \/    \    /\  /  \    /
	*                                                            \  /  \/    \  /
	*                                                             \/          \/
	*                                                                     
 	*     
 	*/

	// first part
	for (k = 0; k < NZ; k++)
	{
		for (j = 0; j < NY; j++)
		{
			for (i = 0; i < NX; i++)
			{
	           	idx = (k + 1) + N * (j + NY * i);
				original_idx = k + NZ * (j + NY * i);

				indata_expanded[idx] = indata[original_idx] * expfield[original_idx];
				if (k == 0) // note the (k + 1) in the idx
					indata_expanded[idx - 1] = 0.0 + 0.0 * I;
			}
		}
	}

	for (j = 0; j < NY; j++)
	{
		for (i = 0; i < NX; i++)
		{
			idx = (NZ + 1) + N * (j + NY * i);

			// middle position
			indata_expanded[idx] = 0.0 + 0.0*I;
		}
	}

	// second mirrored part
	for (k = (NZ + 2); k < N; k++)
	{
		for (j = 0; j < NY; j++)
		{
			for (i = 0; i < NX; i++)
			{
	           	idx = k + N * (j + NY * i);
				original_idx = (N - 1 - k) + NZ * (j + NY * i);

				indata_expanded[idx] = -indata[original_idx] * expfield[original_idx];
			}
		}
	}

	// 2. Forward FFT
	fftw_execute_dft(forwardplan, indata_expanded, outdata_expanded);

	// 3. Multiply by the exponential argument
	for (k = 0; k < N; k++)
	{
		for (j = 0; j < NY; j++)
		{
			for (i = 0; i < NX; i++)
			{
				size_t idx = k + N * (j + NY * i);

				indata_expanded[idx] = outdata_expanded[idx] * exparg[idx];
			}
		}
	}

	// 4. Inverse FFT
	fftw_execute_dft(reverseplan, indata_expanded, outdata_expanded);

	// 5. Multiplying by the exponential field
	for (k = 0; k < N; k++)
	{
		for (j = 0; j < NY; j++)
		{
			for (i = 0; i < NX; i++)
			{
				size_t idx = (k + 1) + N * (j + NY * i); // not taken into account the first element that is zero
				size_t original_idx = k + NZ * (j + NY * i);

				if (k < NZ)
				{
					indata[original_idx] = outdata_expanded[idx] * expfield[original_idx];
				}
			}
		}
	}
}
