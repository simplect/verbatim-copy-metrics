from oct2py import octave

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from verbatim_metrics.data import df, get_simulation_maps
training_image, sim_image, index_map = get_simulation_maps('stone',
                                                           df.sample().iloc[0].at['simulation_parameters'])
#octave.eval('pkg install -forge image')
octave.eval('pkg load image')
octave.addpath('/Users/merijn/s/3. Studie/36._thesis/36.03_code/spatialstats-utilitary-matlab')
octave.validationMetric(np.arange(100).reshape((10,10)))i
octave.validationMetric(np.arange(100).reshape((10,10)))
