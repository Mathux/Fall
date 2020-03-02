import click

# Method
from src.models.fall import Fall

# Syntetic datasets
from src.data.simple import Simple
from src.data.moon import Moon


@click.command()
@click.option('--folder', default="figures", help="Folder to store the figures.")
@click.option('--show/--no-show', default=True, help="Show the plots during the generation.")
def main(folder, show):
    # Figures for the simple step function
    simple = Simple()
    simple.create_figures(Fall, figpath=folder, show=show)

    # Figures for the moon dataset
    moon = Moon()
    moon.create_figures(Fall, figpath=folder, show=show)


if __name__ == "__main__":
    main()
