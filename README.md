Author : Cyril Markovitch

----- Presentation
Monte-Carlo algorithm to calculate the energy distribution of fast atom and the yield of chemical reactions.


1. 



------- Questions

1. How to add a reaction to the algorithm ?

	a. Add a class in the file physics/reactions.py  
    	--> Important remark : Elastic collisions has to be the first reaction because when building tables in the file data/build_tables, 
        indexes are based on the fact that elastic collisions are given in REACTION_list first

	b. Add the cross section in the file data/build_CS_dico.py
    	--> Remark for rovibrational transition : If rovib is modelled with a unique cross section and Delta_E along with the stopping cross section
        Then a temperature dependancy is added.

	c. Run in the terminal " python3 -m data.build_CS_dico " --> Build the dictionnary of cross section

	d. Run in the terminal " python3 -m data.build_tables " --> Build the tables used in the MC code of all the data needed
