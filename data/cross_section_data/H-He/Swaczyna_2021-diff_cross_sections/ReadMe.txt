Differential cross sections for collision of He atoms with H+, He+, H0, and He0

Author:
Pawel Swaczyna
Department of Astrophysical Sciences, Princeton University
swaczyna@princeton.edu

This archive contains differential cross sections obtained in connection with
the study presented in: Swaczyna, P., Rahmanifard, F., Zirnstein, E. J.,
McComas, D. J., Heerikhuisen, J., "Slowdown and Heating of Interstellar Neutral
Helium by Elastic Collisions Beyond the Heliopause", submitted to ApJL

The archive includes 6 folders with the following differential cross sections:
1) H0He0el: elastic collision H^0 + He^0 -> H^0 + He^0
2) He0He0el: elastic collision He^0 + He^0 -> He^0 + He^0
3) H1He0el: elastic collision H^+ + He^0 -> H^+ + He^0
4) He0He1in: indistinguishable particle collision He^0 + He^+ -> He^0 + He^+
5) He0He1el: elastic collision He^0 + He^+ -> He^0 + He^+
6) He0He1ct: charge transfer collision He^0 + He^+ -> He^+ + He^0
See details of the collision types in Appendix A of the above paper.

Each folder contains 661 files with cross sections
for center-of-mass collision energies from 10^-4 to 10^4 eV.

The file name structure is: dcsCOLLISION_1000log10e_SXXXX.dat, where:
COLLISION indicates type of collision, as shown above,
SXXXX is equal 1000*log_10(E_CM), where E_CM is center-of-mass collision energy.

The structure of the files:
* 18 lines of header with some general information about the collision
* 768 lines with 3 columns presenting differential cross sections:
  * column 1: scattering angle (rad)
  * column 2: dsigma/dOmega*2*pi*sin(the) (a.u. = 2.8002852056E-17 cm^2)
  * column 3: weight for cross section integration (approx. width of angle bin)
