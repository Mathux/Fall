from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.kernel_ridge import KernelRidge
from src.models.localizedlasso import LocalizedLasso
from src.models.fall import Fall


class _TunableModels:
    def __init__(self):
        self.model = self.classmodel()
        
    def __call__(self):
        self.model.trial2params = self.trial2params

        def search_param_funct(trial):
            return self.trial2params(self.trial(trial))
        
        self.model.search_params = search_param_funct
        return self.model


def _called(cls):
    return cls()()


@_called
class KNN(_TunableModels):
    classmodel = KNeighborsRegressor
    
    def trial(self, trial):
        # nb of neighboors inside anchors models
        step_nneighbors = trial.suggest_int("step_nneighbors", 1, 10)

        # type of prediction
        weights = trial.suggest_categorical("weights", ['uniform', "distance"])
        return {"step_nneighbors": step_nneighbors, "weights": weights}
    
    def trial2params(self, dico):
        return {"n_neighbors": 5 * dico["step_nneighbors"],
                "weights": dico["weights"]}
    

@_called
class Fall(_TunableModels):
    classmodel = Fall

    def trial(self, trial):
        # nb of anchors
        step_k = trial.suggest_int("step_k", 1, 10)

        # nb of neighboors inside anchors models
        step_Kanchors = trial.suggest_int("step_Kanchors", 1, 10)

        # nb of neighboors for prediction
        step_Kprediction = trial.suggest_int("step_Kprediction", 1, 10)

        # alpha
        exp_alpha = trial.suggest_int("exp_alpha", -3, 3)

        # l1_ratio
        step_l1_ratio = trial.suggest_int("step_l1_ratio", 0, 10)

        # lamda regularization
        exp_lam = trial.suggest_int("exp_lam", -3, 3)
        
        # bias
        bias = trial.suggest_categorical("bias", [True, False]),

        return {"step_k": step_k,
                "step_Kanchors": step_Kanchors,
                "exp_alpha": exp_alpha,
                "step_l1_ratio": step_l1_ratio,
                "bias": bias,
                "exp_lam": exp_lam,
                "step_Kprediction": step_Kprediction}

    def trial2params(self, dico):
        return {"k": 20 * dico["step_k"],
                "K_anchors": 5 * dico["step_Kanchors"],
                "K_prediction": 5 * dico["step_Kprediction"],
                "alpha": pow(10, dico["exp_alpha"]),
                "l1_ratio": 0.1 * dico["step_l1_ratio"],
                "lam": pow(10, dico["exp_lam"]),
                "bias": dico["bias"]}


@_called
class NetworkLasso(_TunableModels):
    classmodel = LocalizedLasso

    def trial(self, trial):
        return {"biasflag": trial.suggest_categorical("biasflag", [True, False]),
                "exp_lamnet": trial.suggest_int("exp_lamnet", -3, 3)}

    def trial2params(self, dico):
        return {"biasflag": dico["biasflag"],
                "lam_net": pow(10, dico["exp_lamnet"]),
                "lam_exc": 0,
                "K": 5}


@_called
class KRR(_TunableModels):
    classmodel = KernelRidge

    def trial(self, trial):
        params = {"kernel": trial.suggest_categorical("kernel", ["rbf", "poly"]),
                  "exp_alpha": trial.suggest_int("exp_alpha", -3, 3)}
        if params["kernel"] == "rbf":
            params["exp_gamma"] = trial.suggest_int("exp_gamma", -3, 3)
        elif params["kernel"] == "poly":
            params["degree"] = trial.suggest_int('degree', 1, 10)
            params["exp_coef0"] = trial.suggest_int("exp_coef0", -3, 3)
        return params

    def trial2params(self, dico):
        params = {"alpha": pow(10, dico["exp_alpha"])}
        if dico["kernel"] == "rbf":
            params["gamma"] = pow(10, dico["exp_gamma"])
        elif dico["kernel"] == "poly":
            params["degree"] = dico["degree"]
            params["coef0"] = pow(10, dico["exp_coef0"])

        return params

    
@_called
class Ridge(_TunableModels):
    classmodel = Ridge

    def trial(self, trial):
        return {"exp_alpha": trial.suggest_int("exp_alpha", -3, 3),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False])}

    def trial2params(self, dico):
        return {"alpha": pow(10, dico["exp_alpha"]),
                "fit_intercept": dico["fit_intercept"],
                "max_iter": 1000,
                "tol": 1e-2}


@_called
class Lasso(_TunableModels):
    classmodel = Lasso

    def trial(self, trial):
        return {"exp_alpha": trial.suggest_int("exp_alpha", -3, 3),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False])}

    def trial2params(self, dico):
        return {"alpha": pow(10, dico["exp_alpha"]),
                "fit_intercept": dico["fit_intercept"],
                "max_iter": 1000,
                "tol": 1e-2}
