# set package
from julia import Main

Main.eval("using AdversarialPrediction")
Main.eval("import AdversarialPrediction: define, constraint")

class Metric:
    def __init__(self, metric_str):
        self.metric_str = metric_str

        # check name and args
        name_args = metric_str.split("@metric")[1].strip().split("\n")[0].split()
        self.name = name_args[0]
        self.args = name_args[1:]

        # run on julia
        Main.eval(metric_str)

    def initialize(self, *args):
        init_str = self.name + "(" + ( ", ".join(map(str, args)) ) + ")"
        # construct metric object
        self.pm = Main.eval(init_str)

    def special_case_positive(self):
        Main.special_case_positive_b(self.pm)

    def special_case_negative(self):
        Main.special_case_negative_b(self.pm)

    def cs_special_case_positive(self, val):
        Main.cs_special_case_positive_b(self.pm, val)

    def cs_special_case_negative(self, val):
        Main.cs_special_case_negative_b(self.pm, val)

    def compute_metric(self, yhat, y):
        return Main.compute_metric(self.pm, yhat, y)

    def compute_constraints(self, yhat, y):
        return Main.compute_constraints(self.pm, yhat, y)

    def objective(self, psi, y):
        return Main.objective(self.pm, psi, y)

    @staticmethod
    def set_solver(val):
        if val == "GUROBI":
            Main.eval("using JuMP")
            Main.eval("using Gurobi")
            Main.eval("const GUROBI_ENV = Gurobi.Env()")
        else:
            Main.eval("using JuMP")
            Main.eval("using ECOS")
