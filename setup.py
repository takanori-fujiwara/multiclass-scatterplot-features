from distutils.core import setup

setup(
    name="multiclass-scatterplot-features",
    version=0.1,
    packages=["multiclass_scatterplot_features"],
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "alphashape",
        "shapely",
        "pymfe",
    ],
)
