### Importing useful packages
import os
import glob
import pandas as pd
import numpy as np
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import csv
import pims
import trackpy as tp
import glob
import argparse


### Importing useful scripts
from utils import *

### Main functioning of script
def main(pathToInput):

	## Create output folder
	if not os.path.exists('output/'):
		os.makedirs('output/')
	# os.mkdir('output/' + pathToInput)

	## Convert all YOLOv5 with angles label files to a single pandas file
	# YOLOv5_to_pandas_file_all(pathToInput)

	## Link pandas csv with trackpy link
	# link_pandas(pathToInput, maxDisp=10, maxMem=25, fMin=10)

	## Organize the linked csv into better, faster txt storage
	better_storage(pathToInput)

	## End of function
	return


### Call if script is run directly
if __name__ == '__main__':

	## Input directory inside 'input/'
	pathToInput = 'r1/'
	
	## Call main
	main(pathToInput)