# Navigation MDP

A bare-bones Python3 interface for specifying navigation mdp. Designed to provide more natural syntax to specify MDP while also providing some extensibility required for experimentation. 

## Markov Decision Process (MDP)

MDP is defined by states S, dynamics T, actions A, and rewards R. The design philosophy of this library is that each entity in the MDP is a separate object. State is the central entity. Everything else is optional, and can be defined and attached to the states as per the need.

## Installation

To install current release with [pip](https://pypi.python.org/pypi/pip):

    pip install navigation-mdp


To install from source:

    python setup.py install



## Usage
If you're curious what it can do:

### 1. View examples in the notebook
[Navigation MDP](https://github.com/yrevar/navigation_mdp/blob/master/navigation_mdp.ipynb)

### 2. Play with it on Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yrevar/navigation_mdp/blob/master/navigation_mdp.ipynb)
    

### 3. Play with it on MyBinder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yrevar/navigation_mdp/master?urlpath=https%3A%2F%2Fgithub.com%2Fyrevar%2Fnavigation_mdp%2Fblob%2Fmaster%2Fnavigation_mdp.ipynb)

## Dependency
For visualizations: https://github.com/yrevar/navigation_vis

# Acknowledgements
- Thanks to Michael Littman, Lucas Lehnert, and David Abel for all the discussions which were very helpful in developing concepts.       
- State class is inspired from David Abel's Simple RL framework: https://github.com/david-abel/simple_rl 


