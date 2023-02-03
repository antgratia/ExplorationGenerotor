import utils.java_call as jc
import numpy as np
import pandas as pd
import pygad
import os

import tensorflow as tf

def evol_search(dir, exp_name, dir_csv, nb_archi, nb_mutation=20, num_parents_mating=4, parent_selection_type="sss", 
                crossover_probability=0.1, mutation_probability=0.1, stop_criteria="saturate_3"):

    current_localisation = os.getcwd()

    # init population
    jc.get_some_architectures(exp_name, str(nb_archi))

    os.chdir(dir+exp_name)
    for i in range(1,nb_archi+1):
        # exec file
        os.system("python " +"architecture_"+str(i)+".py")
        
    os.chdir(current_localisation)

    out_sml_file = read_sml_file(exp_name, nb_archi)

    dico = create_dico(out_sml_file)

    init_pop = encode(dico, out_sml_file)

    init_pop = tf.keras.preprocessing.sequence.pad_sequences(init_pop, padding="post")

    

    # Generation
    def fitness_function(solution, solution_idx):
        counting = 1 + solution_idx

        l = list(filter(lambda x: x != 0, solution))
        l = decode(dico, l)
        l = concat(l)

        resp = jc.verify_sml(l)

        if(resp):
            # create architecture modify
            jc.create_spe_archi(exp_name+str(counting)+"/", str(counting), l)

            # exec generated file 
            os.chdir(dir+exp_name)
            os.system("python " +str(counting)+"/"+"architecture_"+str(counting)+".py")
            os.chdir(current_localisation)

            try:
                # read result in csv file
                df = pd.read_csv(dir_csv+exp_name+str(counting)+"/"+"architecture_results.csv", names=['file_name', 'training_time(s)', 'test_result_loss', 'test_result_acc', 'train_result_acc', 'train_result_loss', 'nb_layers', 'epochs'])
                
                counting += 1
                # return accuracy
                return df['test_result_acc'].iloc[-1]

            # file not found
            except: return 0
        # architecture non valid
        else: return 0

    
    ga_instance = pygad.GA(num_generations=nb_mutation,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       initial_population=init_pop,
                       gene_type=int,
                       parent_selection_type=parent_selection_type,
                       crossover_type="uniform",
                       mutation_type="random",
                       mutation_probability=mutation_probability,
                       crossover_probability=crossover_probability,
                       stop_criteria = stop_criteria)
                       

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    #print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

def create_dico(architectures):
    dico = {}
    i = 1
    for a in architectures:
        for word in a.split(" ")[:-1]:
            if(word not in dico):
                dico[word] = i
                i=i+1
    return dico

def encode(dico, architectures):
    list_architecture_enc = []
    for architecture in architectures:
        a_encode = []
        for word in architecture.split(" ")[:-1]:
            a_encode.append(dico[word])
        list_architecture_enc.append(a_encode)
    return list_architecture_enc

def decode(dico, architecture):
    a_decode = []
    for elem in architecture:
        a_decode.append([k for k, v in dico.items() if v == elem])
    return a_decode

def concat(archi):
    concat_str = ""
    for a in archi:
        concat_str += a[0] + ' '

    return concat_str

def read_sml_file(exp_name, nb_archi):
    dir_sml = "output/architecture_sml/"

    pop = []

    for i in range(1, nb_archi+1):
        f = open(dir_sml+exp_name+"architecture_"+str(i)+".sml", "r")
        output = f.read()
        pop.append(output)

    return pop

def read_csv_file(exp_name):
    dir_csv = "output/architecture_csv/"

    df = pd.read_csv(dir_csv+exp_name+"architecture_results.csv", names=['file_name', 'training_time(s)', 'test_result_loss', 'test_result_acc', 'train_result_acc', 'train_result_loss', 'nb_layers', 'epochs'])


    return df['test_result_acc'].to_numpy(dtype=str)


# crossover
def crossover_func(parents, offspring_size, ga_instance):

    parents_size = parents.shape[0]
    architecture_len = parents.shape[1]

    offspring_evolution = np.empty((1, architecture_len), dtype=np.int32)

    crossover_point = np.random.randint(low=0, high=architecture_len)
    parent1_idx = np.random.randint(low=0, high=parents_size)
    parent2_idx = np.random.randint(low=0, high=parents_size)

    offspring_evolution[0, :crossover_point] = parents[parent1_idx, :crossover_point]
    
    offspring_evolution[0, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring_evolution

# mutation 
def mutation_func(offspring, ga_instance):
    architecture_len = offspring.shape[1]

    for l in range(architecture_len):
        mutation_p = np.random.choice([0, 1], p=[0.9, 0.1])

        if mutation_p == 1:
            # Mutation activate
            micro_len = 10
            random_mutation = np.random.randint(low=0, high=micro_len)

            offspring[0, l] = random_mutation
    return offspring