from pathlib import Path


BASEPATH = Path(__file__).resolve().parent.parent


# Target number of vertices for the protein surface mesh.
#
# Default: 1000 vertices for reduced computational time and memory usage.
# Any positive integer can be specified, subject to mesh generation
# constraints and available computational resources.
#
# Example configurations:
#   1000 - Default; lower computational cost.
#   2000 - Intermediate mesh resolution.
#   3000 - Higher resolution used in the main benchmarks of the
#          published IFACE paper.

TARGET_VERTEX_COUNT = 1000








# Empirically fitted scaling exponents used to transfer distance
# normalization ranges across different mesh resolutions.

P_EXPONENTS = {
    "structural": 0.9010529033,
    "mean_curvature": 1.3072033390,
    "hbond": 1.3476586452,
    "hphob": 1.2530723022,
    "charge": 1.2394529308,
}

# Reference minimum and maximum values at a mesh resolution
# of 3000 vertices.

REF_MIN = {
    "structural": 3398.487255,
    "mean_curvature": 490.213150,
    "hbond": 225.427645,
    "hphob": 3367.141933,
    "charge": 1219.687797,
}

REF_MAX = {
    "structural": 104856.329234,
    "mean_curvature": 2038.293494,
    "hbond": 2008.707551,
    "hphob": 35990.127628,
    "charge": 125866.009619,
}

