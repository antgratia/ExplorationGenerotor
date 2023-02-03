import utils.java_call as jc
#import utils.bat_call as bs
import os
import pandas as pd


def random_search(dir, exp_name, nb_archi):

    current_localisation = os.getcwd()
    jc.get_some_architectures(exp_name, str(nb_archi))

    os.chdir(dir+exp_name)
    for i in range(1,nb_archi+1):
        os.system("python " +"architecture_"+str(i)+".py")
        #bs.execute_py_file(dir+exp_name, "architecture_"+str(i)+".py")

    os.chdir(current_localisation)



def get_best(dir_csv, rs_exp_name):
    df = pd.read_csv(dir_csv+rs_exp_name+"architecture_results.csv", names=['file_name', 'training_time(s)', 'test_result_loss', 'test_result_acc', 'train_result_acc', 'train_result_loss', 'nb_layers', 'epochs'])

    print(df['test_result_acc'])
    print(df['train_result_acc'])

    return df
