This folder contains the adjacency matrices for social networks from 10 groups. In each group folder you'll find .csv-files with the adjacency 
matrix of that network (see below for network definitions). The names of the individuals (e.g., '0007DEEABC') are sorted so that the individuals
within a group (e.g., G1) appear at the same position of the adjacency matrix across networks. Each network is computed from values collected 
over one week, over the span of two weeks. So for every network (e.g., interaction count), there are two matrices: "*resD7_1" for the first week,
"*resD7_2" for the second week. If you'd prefer different temporal resolutions, I have the day-by-day and 3-day resolution already pre-processed.

Definitions (general/metadata):
	group: a set of 10 individual mice living together in the NoSeMaze for 3 weeks
	repetition: the n-th time an individual participated in a "group". E.g., Mouse '0007DEEABC' is in Group 1 with repetition 1, and then it is
again part of Group 7 with repetition 2

Definitions (networks):
	approach_prop: proportion of total approaches from all approaches of that animal. E.g. A(1,2) = .15 means that the animal in the 1st row approached
the animal in the 2nd column in 15% of all its approaches
	
	approaches: total count of approaches

	HWI_t: social preference index (half-weighted index, Whitehead)

	interactions: total count of interaction events

	mean_dist: mean distance during interaction events

	paired_social: proportion of social interaction time spent with this interactin partner. E.g. A(1,2) = .15 means that the animal in the 1st row spent
15% of all its time socially interacting with the animal in the 2nd column

	t_mean: mean duration of a single interaction event

	t_summed: total interaction time of dyad
	