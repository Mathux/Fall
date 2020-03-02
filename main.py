import click
from src.data.loader import DatasetLoader

# Testing framework
from src.utils.evaluate import evaluate_models

# Models with hyper parameter space search
from tunning import KNN, KRR, Fall, Ridge, Lasso, NetworkLasso
tunableModels = {"fall": Fall,
                 "netlasso": NetworkLasso,
                 "ridge": Ridge,
                 "lasso": Lasso,
                 "knn": KNN,
                 "krr": KRR}

allModels = ["fall", "netlasso", "ridge", "lasso", "knn", "krr"]
# Remove network lasso by default because of a long running time
allModels = ["fall", "ridge", "lasso", "knn", "krr"]


@click.command()
@click.option('--dataset', default=None,
              type=click.Choice(list(DatasetLoader.datasetList),
                                case_sensitive=False))
@click.option('--model', default="all",
              type=click.Choice(["all"] + list(tunableModels), case_sensitive=False))
@click.option('--cv', default=3, type=int, help="Number of splits for the cross-validation.")
@click.option('--nepochs', default=20, type=int, help="Number of times to compute everything.")
@click.option('--timeout', default=60, type=int, help="Timeout for computing the best hyper parameters.")
@click.option('--output', default=None, help="Name of the latex output file to store the results.")
@click.option('--verbose/--no-verbose', default=False, help="Verbosity for optuna.")
def main(dataset, model, cv, nepochs, timeout, output, verbose):
    datasets, settings = DatasetLoader(dataset)
    if model == "all":
        models = {model: tunableModels[model] for model in allModels}
    else:
        models = {model: tunableModels[model]}

    evaluate_models(models, cv, datasets,
                    dataSettings=settings,
                    nepochs=nepochs, timeout=timeout,
                    tablepath=output, verbose=verbose)


if __name__ == "__main__":
    main()
