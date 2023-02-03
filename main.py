import search_strategy.random_search as rs

import search_strategy.ga_search as gs


dir = "output/architecture_py/"
dir_csv = "output/architecture_csv/"
rs_exp_name = "test_cc/"
ga_exp_name = "test_ga/"
nb_archi = 1

rs.random_search(dir, rs_exp_name, nb_archi)
#df = rs.get_best(dir_csv, rs_exp_name)

#gs.evol_search(dir, ga_exp_name, dir_csv ,nb_archi)


