// Size of the system
#define LX 20.0
#define LY 1.0
#define LZ 8.4

#define ITER 300000 // Number of iteraions
#define D_WRT 1000 // Frequency of writing to files
#define RELAXATION 0 // Iteration where the noise is turned off
#define D_WRT_ENERGIES   1 // Frequency of writing free energie and chemical potentials to files
#define DT_NOISE    10.0 // Amplitude of the noise
#define MASK 1  // Enables brushes on the substrate, 0 = no brushes, 1 = brushes

// Droplet seeeding in the beginning of the simulation, 0 = no seed, 1 = seed
#define ENABLE_SEED 1
#define SEED_STEPS 1  // number of iterations for the seed 
#define SEED_AREA_DELTA 1 // delta for the area of the seed around the center, min value = 1

#define NS 101 // Length of the chain
#define DR_TARGET   0.1
#define DS (1.0 / (NS - 1))
#define G1 0.01      // Controls the time step of the evolution of the fields

#define CHI_N_AB 0.0
#define CHI_N_AC 0.0
#define CHI_N_AD 0.0
#define CHI_N_AE 0.0
#define CHI_N_AF 0.0
#define CHI_N_AG 0.0
#define CHI_N_AH 0.0
#define CHI_N_AI 0.0
#define CHI_N_AJ 0.0

#define CHI_N_BC 18.0
#define CHI_N_BD 0.0
#define CHI_N_BE 0.0
#define CHI_N_BF 0.0
#define CHI_N_BG 0.0
#define CHI_N_BH 0.0
#define CHI_N_BI 0.0
#define CHI_N_BJ 0.0

#define CHI_N_CD 0.0
#define CHI_N_CE 0.0
#define CHI_N_CF 0.0
#define CHI_N_CG 0.0
#define CHI_N_CH 0.0
#define CHI_N_CI 0.0
#define CHI_N_CJ 0.0

#define CHI_N_DE 0.0
#define CHI_N_DF 0.0
#define CHI_N_DG 0.0
#define CHI_N_DH 0.0
#define CHI_N_DI 0.0
#define CHI_N_DJ 0.0

#define CHI_N_EF 0.0
#define CHI_N_EG 0.0
#define CHI_N_EH 0.0
#define CHI_N_EI 0.0
#define CHI_N_EJ 0.0

#define CHI_N_FG 0.0
#define CHI_N_FH 0.0
#define CHI_N_FI 0.0
#define CHI_N_FJ 0.0

#define CHI_N_GH 0.0
#define CHI_N_GI 0.0
#define CHI_N_GJ 0.0

#define CHI_N_HI 0.0
#define CHI_N_HJ 0.0

#define CHI_N_IJ 0.0

#define F_INPUT_DBK_1 0.5 // with A and B field
#define LEN_DBK_1 1.0
#define F_INPUT_DBK_2 0.5 // with D and E field
#define LEN_DBK_2 1.0
#define F_INPUT_DBK_3 0.3 // with F and G field
#define LEN_DBK_3 1.0
#define F_INPUT_DBK_4 0.5 // with H and I field
#define LEN_DBK_4 1.0

#define VOL_DBK_1         0.186012
#define VOL_DBK_2         0.0
#define VOL_DBK_3         0.0
#define VOL_DBK_4         0.0
#define VOL_HOMOPOLYMER_A 0.0
#define VOL_HOMOPOLYMER_B 0.0
#define VOL_HOMOPOLYMER_C 0.813988
#define VOL_HOMOPOLYMER_D 0.0
#define VOL_HOMOPOLYMER_E 0.0
#define VOL_HOMOPOLYMER_F 0.0
#define VOL_HOMOPOLYMER_G 0.0
#define VOL_HOMOPOLYMER_H 0.0
#define VOL_HOMOPOLYMER_I 0.0
#define VOL_HOMOPOLYMER_J 0.0

#define LEN_HOMO_A 1.0   
#define LEN_HOMO_B 1.0
#define LEN_HOMO_C 1.0
#define LEN_HOMO_D 1.0
#define LEN_HOMO_E 1.0
#define LEN_HOMO_F 1.0
#define LEN_HOMO_G 1.0
#define LEN_HOMO_H 1.0
#define LEN_HOMO_I 1.0
#define LEN_HOMO_J 1.0

// Brushes on the substrate
#define VOL_S1 0.0
#define LEN_S1 0.1 

#define VOL_S2 0.04
#define LEN_S2 0.1

#define VOL_S3 0.0
#define LEN_S3 0.1

#define CHI_N_S1_S2 10.0
#define CHI_N_S1_S3 10.0
#define CHI_N_S2_S3 10.0

#define CHI_N_S1_A 50.0
#define CHI_N_S1_B 50.0
#define CHI_N_S1_C 0.0
#define CHI_N_S1_D 0.0
#define CHI_N_S1_E 0.0
#define CHI_N_S1_F 0.0
#define CHI_N_S1_G 0.0
#define CHI_N_S1_H 0.0
#define CHI_N_S1_I 0.0
#define CHI_N_S1_J 0.0

#define CHI_N_S2_A 0.0
#define CHI_N_S2_B 0.0
#define CHI_N_S2_C 0.0
#define CHI_N_S2_D 0.0
#define CHI_N_S2_E 0.0
#define CHI_N_S2_F 0.0
#define CHI_N_S2_G 0.0
#define CHI_N_S2_H 0.0
#define CHI_N_S2_I 0.0
#define CHI_N_S2_J 0.0

#define CHI_N_S3_A 0.0
#define CHI_N_S3_B 0.0
#define CHI_N_S3_C 0.0
#define CHI_N_S3_D 0.0
#define CHI_N_S3_E 0.0
#define CHI_N_S3_F 0.0
#define CHI_N_S3_G 0.0
#define CHI_N_S3_H 0.0
#define CHI_N_S3_I 0.0
#define CHI_N_S3_J 0.0

/* Lines above were generated using the bash script */
#define G2   G1 / 6.0           // G1 / 6
#define G3   G1 / 6.0           // G1 / 6
#define G4   G1 / 6.0           // G1 / 6
#define G5   G1 / 6.0           // G1 / 6
#define G6   G1 / 6.0           // G1 / 6
#define G7   G1 / 6.0           // G1 / 6
#define G8   G1 / 6.0           // G1 / 6
#define G9   G1 / 6.0           // G1 / 6
#define G10   G1 / 6.0           // G1 / 6
#define G11   G1 / 6.0           // G1 / 6
#define G12   G1 / 6.0           // G1 / 6
#define G13   G1 / 6.0           // G1 / 6

#define DX  (LX/NX)
#define DY  (LY/NY)
#define DZ  (LZ/NZ)

#define ALPHAX  (DS/(DX*DX))
#define ALPHAY  (DS/(DY*DY))
#define ALPHAZ  (DS/(DZ*DZ))

// File names
#define COUNTER "counter.dat"
#define DENSITY "densities_dbk_"
#define DEVIATIONS  "deviations.dat"
#define ENERGY  "energies.dat"
#define RESTART_FIELDS  "restart_fields.dat"
#define RESTART_RANDOM  "restart_random.dat"
#define RUN_PARAMS  "run.txt"
#define W_MASK  "wMask.dat"
#define FREE_ENERGY "free_energy.dat"
