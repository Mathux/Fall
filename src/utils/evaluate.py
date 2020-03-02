import numpy as np
import pandas
import optuna

from sklearn.model_selection import KFold
from sklearn.base import clone

from time import time
# from tqdm import tqdm

from .scoring import make_scoring


def format_score(scores):
    return "{:0.2f}".format(scores)


def format_cross(mean, std):
    formator = "{:0.2f} (+/- {:0.2f})"
    # 1.96 to have data in the 95% interval
    return formator.format(mean, std * 1.96)


def evaluate_models(models, cv, datasets,
                    dataSettings, nepochs,
                    timeout=60,
                    tablepath=None,
                    verbose=False):

    dataname = dataSettings["name"]
    is_classification = dataSettings["classification"]

    scoring = make_scoring(is_classification)
    seeds = np.arange(nepochs)
    sstats = {}
    for epoch, seed in enumerate(seeds):
        sstats[seed] = {}
        dataset = datasets(seed=seed)

        X, y = dataset["train"]["X"], dataset["train"]["Y"]
        if cv >= 2:
            kf = KFold(n_splits=cv, shuffle=False)

        for modelname, model in models.items():
            # define objective function to minimize
            def objective(trial):
                score = 0
                for i, (trainsplit, testsplit) in enumerate(kf.split(X)):
                    Xtr, ytr = X[trainsplit], y[trainsplit]
                    Xval, yval = X[testsplit], y[testsplit]
                    
                    mod = clone(model)
                    mod.set_params(**model.search_params(trial))
                    mod.fit(Xtr, ytr)
                    score += scoring(mod, Xval, yval)
                return score/i

            if cv >= 2:
                sampler = optuna.samplers.TPESampler(seed=seed)
                study = optuna.create_study(direction=scoring.direction,
                                            sampler=sampler)
                if not verbose:
                    optuna.logging.disable_default_handler()
                    
                study.optimize(objective, timeout=timeout) # n_trials=n_trials)
                best_params = model.trial2params(study.best_params)
            else:
                best_params = model.get_params()

            Xtr, ytr = dataset["train"]["X"], dataset["train"]["Y"]
            Xtest, ytest = dataset["test"]["X"], dataset["test"]["Y"]

            mod = clone(model)
            mod.set_params(**best_params)

            # fit the model
            begin = time()
            mod.fit(Xtr, ytr)
            end = time()
            training_time = end - begin

            # test on train
            train_score = scoring(mod, Xtr, ytr)

            # test on test
            begin = time()
            test_score = scoring(mod, Xtest, ytest)
            end = time()
            testing_time = end - begin
            
            # Collect results
            sstats[seed][modelname] = {"train": train_score,
                                       "test": test_score,
                                       "training_time": training_time,
                                       "testing_time": testing_time,
                                       "best_params": best_params}
            # print("Best params for testing: {}".format(best_params))
    # Averaging
    stats = {}
    for modelname, model in models.items():
        def get_mean_and_std(thing):
            mean = np.mean([sstats[seed][modelname][thing] for seed in seeds])
            std = np.std([sstats[seed][modelname][thing] for seed in seeds])
            return mean, std
        
        trM, trS = get_mean_and_std("train")
        teM, teS = get_mean_and_std("test")
        trtimeM, trtimeS = get_mean_and_std("training_time")
        tetimeM, tetimeS = get_mean_and_std("testing_time")

        stats[modelname] = {"Best params with CV": "",
                            "-> train": format_cross(trM, trS),
                            "-> test": format_cross(teM, teS),
                            "-> training time (s)": format_cross(trtimeM, trtimeS),
                            "-> testing time (s)": format_cross(tetimeM, tetimeS),
                            "": ""}

    title = "{} on the {} dataset ({}-fold cross validation) (avg of {})"
    title = title.format(scoring.name, dataname, cv, nepochs)

    dframe = pretty_print(stats, title, scoring)

    if tablepath is not None:
        print()
        make_pdf(dframe, title, tablepath)
        print("table saved in {}".format(tablepath))
        

def make_pdf(dframe, title, tablepath):
    ncols = len(dframe.keys()) + 1
    
    begin = "\\documentclass{standalone}\n\\usepackage{booktabs}\n"
    begin = begin + "\\begin{document}\n"

    latex = dframe.to_latex()

    # Adding title
    begintab = "\\begin{tabular}{" + 'l'*ncols + "}\n"
    title_add = "\\multicolumn{" + str(ncols) + "}{c}{" + title + "}\\\\\n"
    title_add = title_add + "\\multicolumn{" + str(ncols) + "}{c}{}\\\\\n"

    latex = latex.replace(begintab, begintab + title_add)

    # Better arrows
    latex = latex.replace("->", "$\\rightarrow$")

    # Change +/-
    latex = latex.replace("+/-", "$\\pm$")
    
    end = "\n\\end{document}"
    
    with open(tablepath, "w") as tex:
        tex.write(begin + latex + end)

    
def pretty_print(stats, title, scoring, sort=False):
    modelnames = list(stats.keys())
    dsets = list(stats[modelnames[0]].keys())

    results = []
    for dset in dsets:
        dresults = []
        for modelname in modelnames:
            dresults.append(stats[modelname][dset])
        results.append(dresults)

    if sort:
        restest = [float(x.split(" ")[0]) for x in results[2]]
        indsort = np.argsort(np.array(restest, dtype=float))
        indsort = indsort if scoring.first == "min" else indsort[::-1]
    else:
        indsort = np.arange(len(results[2]))
        
    print()
    print("### {}".format(title))
    dframe = pandas.DataFrame(np.array(results)[:, indsort],
                              dsets,
                              np.array(modelnames)[indsort])
    print(dframe)

    return dframe
