# TendonForces
This repository is a validated forcing module that can be used in the open-source simulation software PyElastica. This forcing module applies the external forcing effects of tendon actuation on a rod. The validation of this forcing module was done through physical experimentation, results which can be seen in **TendonForce_Explanation.pdf** included in the repo. The PDF also contains information regarding the mathematical procedure taken in the external forcing module.


To use this forcing module, you must first be sure to have installed PyElastica properly (follow the procedure in their repo https://github.com/GazzolaLab/PyElastica)

Now, when building the simulator, the TendonForces forcing module must be called and supplied with the tendon configuration parameters. This can be seen in the example provided.

The tendon configuration parameters (arguments for the initialization of TendonForces) are the following:

#### vertebra_height: (float)
        Height at which the tendon contacts the vertebra. It should be the highest point on the tendon-vertebra space.
        
#### num_vertebrae: (int)
        Amount of vertebrae to be used in the system.
        
#### first_vertebra_node: (int)
        The first node to have a vertebra, from the base of the rod to the tip.
        
#### final_vertebra_node: (int)
        The last node to have a vertebra, from the base of the rod to the tip.
        
#### vertebra_mass: (float)
        Total mass of a single vertebra.
        
#### tension: (float)
        Tension applied to the tendon in the system.
        
#### vertebra_height_orientation: (numpy.ndarray)
        1D (dim) numpy array. Describes the orientatation of the vertebrae in the system.

#### n_elements: (int)
        Total amount of nodes in the rod system. This value is set in the simulator and is copied to this class for later use.

The structure of TendonForces is such that it is allowed to use several different tendon configurations simultaneously in one simulation, and because tension is one of the arguments provided, the application of the tendon actuation is always done in an open-loop.


## Installation Instructions

To utilize the TendonForces external forcing module for PyElastica, please follow the steps below:

### Prerequisites

Ensure you have [PyElastica](https://github.com/GazzolaLab/PyElastica) installed. If you haven't installed it yet, please refer to its documentation for installation instructions (https://github.com/GazzolaLab/PyElastica).

### Using Git
#### Cloning the Repository

1. First, clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/gabotuzl/TendonForces.git
   ```

2. Navigate to the cloned directory:
   ```bash
   cd TendonForces
   ```

#### Setting Up the Module

To correctly setup and use the `TendonForces.py` module:

1. **Add to Python Path** (optional): If you encounter issues with Python not finding the module, you might need to add the repository to your Python path. You can do this by using the following command in your terminal:
   ```bash
   export PYTHONPATH="$PYTHONPATH:$(pwd)"
   ```
   This command temporarily adds the current directory to your Python path for the current terminal session.

2. **Import the Module**: In your PyElastica simulation script, you can now import the `TendonForces` module with:
   ```python
   import TendonForces
   ```
### Manual installation options
1. Manually download the TendonForces.py script and place it inside the current workspace directory so that python can find it easily. Afterwards, be sure to include the following line in the PyElastica simulation script:
   ```python
   import TendonForces
   ```
3. Edit the external_forces.py script, which contains all of PyElastica's default forcing modules, by copying and pasting the TendonForces class, from TendonForces.py, into external_forces.py. Afterwards, be sure to include the following line in the PyElastica simulation script:
   ```python
   from elastica.external_forces import TendonForces
   ```

**NOTE: the examples provided in this repository assume that the TendonForces class is defined inside external_forces.py.**
## Running Examples

To see how to use the module, you can run the provided examples:

- For visualization: **example_with_visualization.py**

- Without visualization: **example_no_visualization.py**

## Dependencies

Make sure to install the dependencies required by this forcing module:
- [PyElastica](https://github.com/GazzolaLab/PyElastica)
- [numpy](https://numpy.org/)
- [numba](https://numba.pydata.org/)

For the visualization example:
- [matplotlib](https://matplotlib.org/)
- [moviepy](https://zulko.github.io/moviepy/)


# squirrel_gripper
