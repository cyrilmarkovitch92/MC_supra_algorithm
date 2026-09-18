README
------

This folder contains the following files.

./ortho/ortho_h2h_crosssections.txt  - Cross sections for H + ortho-H_2 collisions.
./ortho/ortho_h_h2_rates.txt  - Rate coefficients for H + ortho_H_2 collisions.
./para/para_h2h_crosssections.txt  - Cross sections for H + para-H_2 collisions.
./para/para_h2h_.txt - Rate coefficients for H + para_H_2 collisions.
./hd/h_hd_crosssections.txt - Cross sections for the H + HD collisions.
./hd/h_hd_rates.txt - Rate coefficients for the H + HD collisions.

Cross Section files
-------------------
The basis for the H_2 or HD molecule, used in the calculation, is given at the start of the file. This serves as a key to the column and row notation when looking up the cross sections. 

The first column gives the 'level numbers', the second column gives the vibrational quantum number ,v, the third column gives the rotational quantum number, J, and the final column gives the energy (in Kelvin) of the level with respect to the ground state of the molecule.

The cross sections follow the basis set key. The cross sections are given in units (10^-16 cm^2).

The first column gives the barycentric collision energy, in Kelvin.

The second column gives the 'level number' of the final state. This 'level number' corresponds to the level with the same 'level number' in the basis set, given at the start of the file. For example a 'final level number' of 13 corresponds to a final state v=1 J=9 (for the case of ortho_H_2).

The following columns are numbered from 1-54 for the H_2 case and 1-120 for the HD case. The column numbers correspond to the initial state of the transition. For example a column number of 10 and a 'final level number' of 3 corresponds to the v=1 J=7 --> v=0 J=5 transition (for the case of ortho-H_2).

Rate coefficient files
----------------------
The basis for the H_2 or HD molecule, used in the calculation, is given at the start of the file. This serves as a key to the column and row notation when looking up the rate coefficients. 

The first column gives the 'level numbers', the second column gives the vibrational quantum number ,v, the third column gives the rotational quantum number, J, and the final column gives the energy (in Kelvin) of the level with respect to the ground state of the molecule.

Following the basis set is the first kinetic temperature the rate coefficients are calculated at. This is given in Kelvin.

The rate coefficients immediately follow the corresponding kinetic temperature, in a 54 x 54 matrix for the H_2 case (for the HD case the matrix is 120 x 120). The rate coefficients are given in units cm^3 s^-1. The columns of the matrix correspond to the initial level of the transition and the rows correspond to the final level of the transition. For example, using the key at the start of the file, the element at column 3 and row 2 corresponds to the v=0 J=5 --> v=0 J=3 rate coefficient (for the case of ortho-H_2).

If the wished to find the coefficient for the v=1 J=9 --> v=0 J=7 we would use the basis set key to find column number 13 (initial level) and row number 4 (final level), for the case of ortho-H_2.

Following this matrix is the next kinetic temperature and corresponding rate coefficients. The format repeats for the various kinetic temperatures and their corresponding rate coefficients.