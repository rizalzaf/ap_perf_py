import os
import subprocess

script_dir = os.path.dirname(os.path.realpath(__file__))


def install():
    """
    Install required Julia packages.
    """
    subprocess.check_call(['julia', os.path.join(script_dir, 'install.jl')])
