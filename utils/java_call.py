import subprocess

PROGRAM_NAME = "sml_test.jar"

def verify_sml(sml):
    p = subprocess.Popen(['java', '-jar', PROGRAM_NAME, sml], stdout= subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = p.communicate()
    p.wait()
    if(p.returncode == 2 ):
        return False
    else: return True


def create_spe_archi(name_archi, num, sml):
    subprocess.call(['java', '-jar', PROGRAM_NAME, name_archi, num, sml])


def get_some_architectures(name_archi, nb_archi):
    subprocess.call(['java', '-jar', PROGRAM_NAME, name_archi, nb_archi])
