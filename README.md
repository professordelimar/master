# Master Codes

Historical and evolving computational repository for **gravity, magnetics, potential-field processing, equivalent-layer methods, and geophysical data analysis**.

## About

This repository was originally created during the master's research of **Nelson Ribeiro-Filho** in Geophysics.

It contains Python modules, Jupyter notebooks, synthetic experiments, real-data applications, dissertation-related material, and manuscript-oriented computational tests developed around gravity and magnetic potential fields.

The repository is being preserved as a **research archive in active expansion**. Existing source codes and notebooks are intentionally retained, including legacy implementations, historical experiments, and earlier development versions. New material may be added over time without removing the original research record.

## Main scope

The repository includes computational material related to:

- gravity and magnetic fields;
- gravitational and magnetic responses of simple bodies;
- spheres and rectangular prisms;
- potential-field transformations;
- upward and downward continuation;
- horizontal and vertical derivatives;
- total gradient amplitude;
- phase transformations and reduction filters;
- Poisson relations;
- statistical analysis;
- polynomial regional-residual separation;
- cross-correlation of geophysical data;
- classical equivalent-layer techniques;
- magnetic-data processing;
- synthetic-model validation;
- real-data applications;
- geophysical visualization;
- UTM coordinate conversion;
- dissertation and manuscript experiments.

## Repository structure

### `codes/`

Core Python modules developed or collected for the computational experiments.

Main modules include:

- `auxiliars.py` — auxiliary mathematical and numerical routines;
- `constants.py` — physical and unit-conversion constants;
- `derivative.py` — horizontal and vertical derivative operations;
- `equivalentlayer.py` — classical equivalent-layer routines;
- `filtering.py` — potential-field filtering and transformations;
- `gravity.py` — gravity and reference-field calculations;
- `grids.py` — regular grids and interpolation utilities;
- `kernel.py` — kernel functions used in potential-field calculations;
- `plot.py` — plotting and geophysical visualization routines;
- `prism.py` — gravitational and magnetic calculations for rectangular prisms;
- `regres.py` — regression and regional-residual routines;
- `sphere.py` — gravitational and magnetic calculations for solid spheres;
- `statistical.py` — statistical-analysis utilities;
- `testing.py` and `testing_myfunc.py` — historical testing material.

The repository also contains a local `utm/` implementation used by part of the historical workflow.

## Main notebooks

### Potential-field processing

- `continuation.ipynb` — upward and downward continuation;
- `derivatives.ipynb` — horizontal and vertical derivatives;
- `totalgradient.ipynb` — total gradient amplitude;
- `reduction.ipynb` — phase transformations and reduction;
- `poisson.ipynb` — Poisson relations;
- `cross-correlation.ipynb` — cross-correlation applied to magnetic data.

### Forward modeling

- `sphere-grav.ipynb` — gravity attraction of a solid sphere;
- `sphere-mag.ipynb` — magnetic anomaly of a solid sphere;
- `prism_grav.ipynb` — gravitational potential and attraction of a rectangular prism;
- `prism_mag.ipynb` — total-field magnetic anomaly of a rectangular prism.

### Equivalent-layer methods

- `eqlayer_grav.ipynb` — equivalent-layer applications for gravity-related derivatives;
- `eqlayer_mag.ipynb` — equivalent-layer applications for magnetic reduction.

## Dissertation material

The `dissertation/` directory contains computational material developed for the master's research, including both **real-data** and **synthetic** experiments.

### Real-data applications

The notebooks include applications to magnetic datasets and study areas such as:

- Arraial do Cabo;
- Carajás;
- Morro do Forno;
- Pontal do Atalaia.

Associated datasets are preserved with the notebooks whenever available.

### Synthetic experiments

The `dissertation/synthetic/` directory contains tests designed to investigate, justify, validate, and identify limitations of the proposed computational approaches.

These notebooks include experiments involving total-gradient quantities, cross-correlation, synthetic magnetic sources, and validation scenarios.

## Manuscript-related material

The `manuscript_correlation_R1/` directory preserves notebooks and datasets associated with manuscript-oriented experiments involving **multi-domain cross-correlation**, synthetic tests, equivalent-source configurations, and real magnetic data.

These files are retained as part of the computational provenance of the research.

## Historical character of the repository

Some files were written with older versions of Python and scientific-computing libraries.

Consequently:

- some legacy notebooks may require an older environment;
- some imports or APIs may have changed in modern versions of NumPy, SciPy, Matplotlib, Basemap, or related packages;
- historical tests may reference software that is no longer actively maintained;
- some files may represent intermediate development stages rather than final production implementations.

These files are intentionally preserved because they document the evolution of the research.

The presence of a file in this repository does **not** imply that it has been modernized or validated against the latest software versions.

## Python environment

The core scientific stack represented in the repository includes:

- NumPy
- SciPy
- Matplotlib
- Pandas
- Jupyter
- Basemap

Install the principal dependencies with:

```bash
pip install -r requirements.txt
```

Because the repository contains legacy material, reproducing a specific historical notebook may require additional version-specific dependencies.

## Running the notebooks

Clone the repository:

```bash
git clone https://github.com/professordelimar/master.git
```

Enter the repository:

```bash
cd master
```

Start Jupyter:

```bash
jupyter notebook
```

Then open the desired notebook.

For legacy notebooks, inspect imports and local paths before execution.

## Scientific use

This repository should be interpreted as a computational research archive rather than a single modern Python package.

When using a routine, notebook, dataset, figure, or derived result in academic work:

1. identify the exact file and version used;
2. verify the numerical implementation;
3. record the computational environment;
4. cite the original mathematical or geophysical source when applicable;
5. cite the repository and any associated publication when appropriate.

## Third-party and adapted material

Some routines were historically inspired by or adapted from published geophysical formulations and external computational resources, including material associated with classical potential-field literature.

Third-party algorithms, datasets, libraries, and adapted implementations remain subject to their respective licenses, copyrights, and citation requirements.

See [REFERENCES.md](REFERENCES.md) and [LICENSE](LICENSE).

## Related repositories

More recent and specialized material is being organized separately in dedicated repositories, including:

- https://github.com/professordelimar/gravmag
- https://github.com/professordelimar/geophysics
- https://github.com/professordelimar/geodesy
- https://github.com/professordelimar/matematica
- https://github.com/professordelimar/physics

The `master` repository remains preserved as the historical computational record and may continue receiving additional legacy or research material.

## Author

**Nelson Ribeiro-Filho**

## Historical collaboration

Part of the original master's-project development was carried out with academic collaboration from **Rodrigo Bijani**.

Individual source files and notebooks should be consulted for file-specific authorship or collaboration information.

## Copyright and ownership

Original source code, documentation, notebooks, figures, and other original materials authored by Nelson Ribeiro-Filho and owned by the current rights holder belong to:

**64.200.407 NELSON DE LIMA RIBEIRO FILHO - ME**  
**CNPJ 64.200.407/0001-45**

All rights reserved unless a specific file or third-party component states otherwise.

See [LICENSE](LICENSE) for details.

## Social links

- Instagram: https://www.instagram.com/professordelimar/
- YouTube: https://www.youtube.com/@professordelimar
