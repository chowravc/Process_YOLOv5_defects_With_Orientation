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
from PIL import Image


### Importing useful scripts
from utils import *

### Trackpy data analysis and storage portion of scipt
def tp_analysis(pathToInput):

	## Create output folder
	if not os.path.exists('output/'):
		os.makedirs('output/')
	# os.mkdir('output/' + pathToInput)

	## Convert all YOLOv5 with angles label files to a single pandas file
	# YOLOv5_to_pandas_file_all(pathToInput)

	## Save YOLOv5 files as JSON
	YOLOv5_as_JSON(pathToInput)

	## Link pandas csv with trackpy link
	# link_pandas(pathToInput, maxDisp=10, maxMem=25, fMin=10)

	## Remove particles created after frame cutoff and cutoff the rest to start at frame cutoff
	# fix_linking(pathToInput, cutoff=100)

	## Organize the linked csv into better, faster txt storage
	# better_storage(pathToInput)

	## Get and store binary classifier data to be classified
	# binary_data(pathToInput, minPoints=4)

	## End of function
	return



### Plot a single defect on its relevant frame
def plot_defect(pathToInput, defectNum, frame):

	## Path to defect txt
	pathTxt = 'output/' + pathToInput + 'particlewise_data/particle_' + str(defectNum).zfill(4) + '.txt'

	## Path to frame
	pathFrame = 'input/' + pathToInput + 'frames/' + pathToInput[:-1] + '_' + str(frame).zfill(4) + '.tif'

	## Read defect txt as numpy array
	defectMat = np.loadtxt(pathTxt)

	## Get transpose of numpy array
	defectMatTrans = defectMat.T

	## Get list of frames
	frames = defectMatTrans[0]

	## Check if the desired frame is one of the frames
	if float(frame) in frames:

		# Indicate that the frame has been found
		print('Found frame in defect lifetime.')

		# Get index of the frame
		fIndex = np.where(frames == float(frame))[0][0]

		# Get the x position of the defect in the frame
		xPos = defectMatTrans[1][fIndex]

		# Get the y position of the defect in the frame
		yPos = defectMatTrans[2][fIndex]

		# Print location in frame
		print('Defect ' + str(defectNum) + ' in frame ' + str(frame) + ': (' + str(xPos) + ', ' + str(yPos) + ').')

		# Load the frame as PIL image
		frameIm = Image.open(pathFrame)

		# Clear matplotlib plots
		plt.clf()

		# Plot the image
		plt.imshow(frameIm)

		# Show the defect
		plt.scatter([xPos], [yPos], color='r', s=2)

		# Show the plot
		plt.show()

		# Clear plots
		plt.clf()

	## If the desired frame is not there
	else:

		# Indicate that the frame does not exist in the lifetime
		print('Requested frame not found in defect lifetime.')

		# End function
		return



### Plot a frame with all its defects
def plot_frames(pathToInput, frame):

	## Path to defect txt
	pathTxt = 'output/' + pathToInput + 'framewise_data/frame_' + str(frame).zfill(4) + '.txt'

	## Path to frame
	pathFrame = 'input/' + pathToInput + 'frames/' + pathToInput[:-1] + '_' + str(frame).zfill(4) + '.tif'

	## Read frame txt as numpy array
	frameMat = np.loadtxt(pathTxt)

	## Get transpose of numpy array
	frameMatTrans = frameMat.T

	## Get list of particles
	particles = frameMatTrans[0]

	## Get list of xs
	xs = frameMatTrans[1]

	## Get list of ys
	ys = frameMatTrans[2]

	## Load the frame as PIL image
	frameIm = Image.open(pathFrame)

	## Clear matplotlib plots
	plt.clf()

	## Plot the image
	plt.imshow(frameIm)

	## Show the defect
	plt.scatter(xs, ys, color='r', s=2)

	## Go through every defect
	for i in range(len(xs)):

		# Plot every particle's id
		plt.text(xs[i]+1, ys[i]-1, str(int(particles[i])))

	# Show the plot
	plt.show()

	# Clear plots
	plt.clf()



### Plot the trajectory of some defects on a frame
def plot_traj(pathToInput, frame, defectNums):

	## Path to frame
	pathFrame = 'input/' + pathToInput + 'frames/' + pathToInput[:-1] + '_' + str(frame).zfill(4) + '.tif'

	# Load the frame as PIL image
	frameIm = Image.open(pathFrame)

	# Clear matplotlib plots
	plt.clf()

	# Plot the image
	plt.imshow(frameIm, cmap='gray', zorder=0)

	## Go through every particle and read list of xs and ys
	for defectNum in defectNums:

		## Path to defect txt
		pathTxt = 'output/' + pathToInput + 'particlewise_data/particle_' + str(defectNum).zfill(4) + '.txt'

		## Read defect txt as numpy array
		defectMat = np.loadtxt(pathTxt)

		## Transpose of defect matrix
		defectMatTrans = defectMat.T

		## Plot the x-coordinates
		plt.plot(defectMatTrans[1][0::5], defectMatTrans[2][0::5], color='b', zorder=5)

		interval = int(len(defectMatTrans[1])/6)

		## List of point to plot crosses
		x_cross = defectMatTrans[1][0::interval]

		y_cross = defectMatTrans[2][0::interval]

		plt.scatter(x_cross, y_cross, marker='x', color='r', zorder=10)

	## Show plot
	plt.show()

	return



### Get defect polarities from binary classifier output
def binary_to_polarity(pathToInput):

	## Read label outputs from binary classifier
	labelPaths = glob.glob('output/' + pathToInput + 'binary_classifier_data/labels/*.txt')

	## Display number of detected defects
	print('Detected ' + str(len(labelPaths)) + ' classified defects.')

	## Path to output txt
	outPath = 'output/' + pathToInput + 'polarities.txt'

	## Use to store completed defect numbers
	dNums = []

	## Use to store associated polarities
	polarities = []

	## Go through every classification
	for labelPath in labelPaths:

		# Extract defect number
		dNum = int(labelPath.split('\\')[-1].split('.')[0])
		
		# Add defect number to array of dNums
		dNums.append(dNum)

		# Open labelfile as a numpy array
		classMat = np.loadtxt(labelPath)

		# Find out which classification was higher
		# If the first element is larger
		if classMat[0] >= classMat[1]:

			# Add positive defect to the polarities array
			polarities.append(1)

		# If the second element is larger
		if classMat[0] < classMat[1]:

			# Add negative defect to the polarities array
			polarities.append(-1)

	## Convert both arrays to numpy
	dNums = np.asarray(dNums)
	polarities = np.asarray(polarities)

	## Add them to a common array
	outMat = [dNums, polarities]

	## Now convert this to numpy array and take a transpose
	outMat = np.asarray(outMat).T

	## Finally save the output
	np.savetxt(outPath, outMat)


### Call if script is run directly
if __name__ == '__main__':

	## Input directory inside 'input/'
	pathToInput = 'r4/'
	
	## Call trackpy analysis
	tp_analysis(pathToInput)

	## Call binary classified label to polarity function
	# binary_to_polarity(pathToInput)

	# defectNum = 869

	# frame = int(input('Frame: '))

	# plot_defect(pathToInput, defectNum, frame)
	# plot_frames(pathToInput, frame)

	# defectNumsR2 = [304, 450, 680, 230, 607, 600, 268, 485, 869, 659, 727, 503, 769, 777, 246, 401, 411]
	# defectNumsR3 = [642, 774, 248, 765, 611, 566, 84, 697, 513, 313, 645, 256, 64]
	# defectNumsR4 = []

	# framwise = np.loadtxt('output/' + pathToInput + 'framewise_data/frame_' + str(frame).zfill(4) + '.txt')
	# parts = framwise.T
	# parts = parts[0].astype(np.uint8)

	# print(parts)
	# plot_traj(pathToInput, frame, defectNumsR4)