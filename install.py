import os
import subprocess

script_dir = os.path.dirname(os.path.realpath(__file__))


def install_no_gurobi():
    """
    Install required Julia packages.
    """
    subprocess.check_call(['julia', os.path.join(script_dir, 'install_no_gurobi.jl')])

def install_with_gurobi():
    """
    Install required Julia packages + Gurobi.
    """
    subprocess.check_call(['julia', os.path.join(script_dir, 'install_with_gurobi.jl')])