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
import json


### Importing useful scripts



### Convert YOLOv5 detection results to pandas format
def YOLOv5_to_pandas_file(finalDict, yoloFile, frame):

	## Converting txt file

	dims = (1104, 800) # Image dimensions

	print('Adding file ' + yoloFile)

	## Read YOLO file
	readFile = open(yoloFile, 'r')

	## Read lines from YOLO file
	lines = readFile.read().split('\n')

	## For every line
	for line in lines:

		# If the line is not empty
		if len(line) != 0:

			# vector in the YOLOv5 format: class x_center y_center width height confidence
			v = line.split(' ')

			# Inputs in YOLOv5 format
			inClass = v[0] # Class number
			inXC = float(v[1]) # Center of the bounding box X-coord
			inYC = float(v[2]) # Center of the bounding box Y-coord
			inW = float(v[3]) # Width of the bounding box
			inH = float(v[4]) # Height of the bounding bo
			inConf = float(v[5]) # Confidence of detection
			inAngle = float(v[6]) # Orientation angle of detection

			# Outputs in pandas format for Trackpy: x y mass size ecc signal raw_mass ep frame
			finalDict['x'].append(dims[0]*inXC)
			finalDict['y'].append(dims[1]*inYC)
			finalDict['mass'].append(200.0) # arbitrary
			finalDict['size'].append(3.0) # arbitrary
			finalDict['ecc'].append(0.0) # arbitrary, eccentricity?
			finalDict['signal'].append(10.0) # arbitrary
			finalDict['raw_mass'].append(10000.0) # arbitrary
			finalDict['ep'].append(0.1) # arbitrary
			finalDict['frame'].append(frame)
			finalDict['angle'].append(inAngle) # Use this to store angle for now

			# http://www.physics.emory.edu/faculty/weeks//idl/download.html#extra
			# http://web.mit.edu/savin/Public/.Tutorial_v1.2/
			# http://www.physics.emory.edu/faculty/weeks//idl/tracking.html

	## Return the dictionary
	return finalDict



### Convert every YOLO label file to dictionary and save to single csv
def YOLOv5_to_pandas_file_all(runDir):

	## Detecting files
	files = glob.glob('input/' + runDir + 'YOLOv5_labels/*.txt')
	print('Detected ' + str(len(files)) + ' files.')

	## Dictionary to store the csv elements
	dictionary = {
		'x': [],
		'y': [],
		'mass': [],
		'size': [],
		'ecc': [],
		'signal': [],
		'raw_mass': [],
		'ep': [],
		'frame': [],
		'angle': []
	}

	## Go through every YOLO label file in the run
	for file in files:

		# Get the current frame number
		frameNum = int(file.split('.')[-2].split('_')[-1])

		# Add the detection results to the final dictionary
		dictionary = YOLOv5_to_pandas_file(dictionary, file, frameNum)

	## Create pandas dataframe from the dictionary
	pandasOutput = pd.DataFrame.from_dict(dictionary)

	## Get file path to output	
	outputFile = 'output/' + runDir + 'labels_pandas.csv'

	## Save pandas csv
	pandasOutput.to_csv(outputFile, index=False)

	print('Done. Output saved to ' + outputFile)



### Convert and save YOLOv5 files as JSON
def YOLOv5_as_JSON(runDir):

	## Create output JSON directory
	os.mkdir('output/' + runDir + 'JSON/')

	## Converting txt file
	dims = (1104, 800) # Image dimensions

	## yoloFiles
	yoloFiles = glob.glob('input/' + runDir + 'YOLOv5_labels/*.txt')

	## Go through every YOLO file
	for yoloFile in yoloFiles:

		print('Adding file ' + yoloFile)

		# Read YOLO file
		readFile = open(yoloFile, 'r')

		# Read lines from YOLO file
		lines = readFile.read().split('\n')

		# Use this to store the line of defects
		defects = []

		# For every line
		for line in lines:

			# If the line is not empty
			if len(line) != 0:

				# vector in the YOLOv5 format: class x_center y_center width height confidence
				v = line.split(' ')

				# Inputs in YOLOv5 format
				inClass = v[0] # Class number
				inXC = float(v[1])*dims[0] # Center of the bounding box X-coord
				inYC = float(v[2])*dims[1] # Center of the bounding box Y-coord
				inW = float(v[3])*dims[0] # Width of the bounding box
				inH = float(v[4])*dims[1] # Height of the bounding bo
				inConf = float(v[5]) # Confidence of detection
				inAngle = float(v[6]) # Orientation angle of detection

				# Find pixel coordinates of the top left pixel of bounding box
				top_left_x = inXC - inW/2
				top_left_y = inYC - inH/2

				# Find pixel coordinates of the bottom right pixel of bounding box
				bottom_right_x = inXC + inW/2
				bottom_right_y = inYC + inH/2

				# Add the defect to the list
				defects.append({"label": "class", "confidence": inConf, "topleft": {"x": top_left_x, "y": top_left_y}, "bottomright": {"x": bottom_right_x, "y": bottom_right_y}})

		## Path to output JSON
		outJSON = yoloFile.replace('YOLOv5_labels', 'JSON').replace('input', 'output').replace('txt', 'json')

		## Open the output JSON
		with open(outJSON, 'w') as f:

			# Dump all the data
			json.dump(defects, f)



### Link pandas csv file to get defect ids down
def link_pandas(runDir, maxDisp=10, maxMem=25, fMin=10):

	# maxDisp = max displacement of defect in pixels between frames
	# maxMem = max number of frames a defect can disappear and still be linked
	# fMin = min existence length in frames

	## Get file path to input csv
	inputFile = 'output/' + runDir + 'labels_pandas.csv'

	## Read input data
	inputData = pd.read_csv(inputFile)

	## Get file path to output csv
	outputFile = 'output/' + runDir + 'labels_pandas_linked.csv'

	print('\nLinking defects.')

	## Link the data using input parameters
	linkedData = tp.link(inputData, maxDisp, memory=maxMem)

	print('\nFiltering stubs.')
	print('Before filter:', linkedData['particle'].nunique())

	## Filter out any short lasting defect detections
	linkedData = tp.filter_stubs(linkedData, fMin)
	print('After filter:', linkedData['particle'].nunique())

	## Save linked output file
	linkedData.to_csv(outputFile, index=False)



### Remove particles created after frame cutoff and cutoff the rest to start at frame cutoff
def fix_linking(runDir, cutoff=100):

	## Get file path to input csv
	inputFile = 'output/' + runDir + 'labels_pandas_linked.csv'

	## Read input data
	inputData = pd.read_csv(inputFile)

	## Get file path to output csv
	outputFile = 'output/' + runDir + 'labels_pandas_linked_cutoff.csv'

	print('\nFixing linked defects.')

	## Get the number of rows in the csv
	rows = len(inputData['frame'])

	## Dictionary to store output data
	dictOut = {
		'x': [],
		'y': [],
		'mass': [],
		'size': [],
		'ecc': [],
		'signal': [],
		'raw_mass': [],
		'ep': [],
		'frame': [],
		'angle': [],
		'particle': []
	}

	## Store particles that have already been checked to be created before cutoff
	passingParticles = []

	## Store particles that were created after cutoff
	failingParticles = []

	## Go through every row
	for i in range(rows):

		# Display progress every 1000 rows
		if i == 0 or i%1000 == 0:

			print(str(100*i/rows)[:5] + '%')

		# Extract all information in current row
		x_par = inputData['x'][i]
		y_par = inputData['y'][i]
		mass_par = inputData['mass'][i]
		size_par = inputData['size'][i]
		ecc_par = inputData['ecc'][i]
		signal_par = inputData['signal'][i]
		raw_mass_par = inputData['raw_mass'][i]
		ep_par = inputData['ep'][i]
		frame_par = int(inputData['frame'][i])
		angle_par = inputData['angle'][i]
		particle_par = int(inputData['particle'][i])

		# If this particle had passed
		if particle_par in passingParticles:

			# Check if the current frame is past cutoff
			if frame_par >= cutoff:

				# Add the current row information to output
				dictOut['x'].append(x_par)
				dictOut['y'].append(y_par)
				dictOut['mass'].append(mass_par)
				dictOut['size'].append(size_par)
				dictOut['ecc'].append(ecc_par)
				dictOut['signal'].append(signal_par)
				dictOut['raw_mass'].append(raw_mass_par)
				dictOut['ep'].append(ep_par)
				dictOut['frame'].append(frame_par)
				dictOut['angle'].append(angle_par)
				dictOut['particle'].append(particle_par)

		# If this particle had failed
		elif particle_par in failingParticles:

			# Do nothing and go to the next line
			continue

		# If this particle is brand new and untested
		else:

			# If the first frame is before cutoff
			if frame_par <= cutoff:

				# Add the particle to passing particles
				passingParticles.append(particle_par)

				# Check if the current frame is past cutoff
				if frame_par >= cutoff:

					# Add the current row information to output
					dictOut['x'].append(x_par)
					dictOut['y'].append(y_par)
					dictOut['mass'].append(mass_par)
					dictOut['size'].append(size_par)
					dictOut['ecc'].append(ecc_par)
					dictOut['signal'].append(signal_par)
					dictOut['raw_mass'].append(raw_mass_par)
					dictOut['ep'].append(ep_par)
					dictOut['frame'].append(frame_par)
					dictOut['angle'].append(angle_par)
					dictOut['particle'].append(particle_par)

			# If the first frame is after cutoff
			if frame_par > cutoff:

				# Blacklist the particle
				failingParticles.append(particle_par)

	## Convert output dictionary to pandas dataframe
	dictOut = DataFrame.from_dict(dictOut)

	print('\nShowing filter results (particle number).')
	print('Before filter:', inputData['particle'].nunique())
	print('After filter:', dictOut['particle'].nunique())

	## Save output file with cutoff
	dictOut.to_csv(outputFile, index=False)

	## Clear plots
	plt.clf()

	## Plot trajectories of the defects
	tp.plot_traj(dictOut);

	## Clear plots
	plt.clf()



### Save framewise data
def save_framewise(runDir, data):

	## Create the output directory
	os.mkdir('output/' + runDir + 'framewise_data/')

	## Get the number of rows in the csv
	rows = len(data['frame'])

	## Use to store the current frame
	currentFrame = 0

	## Use to store the last(previous) frame
	lastFrame = 0

	## Go through every row
	for i in range(rows):

		# If on the first row
		if i == 0:

			# Read current frame
			currentFrame = data['frame'][i]

			# Make current frame the last frame
			lastFrame = data['frame'][i]

			print('Saving frame number: ' + str(currentFrame))

		# If not on the first two
		else:

			# Get the next frame and make it current
			currentFrame = data['frame'][i]

			# If we are not on a new frame
			if currentFrame == lastFrame:

				# Don't do anything
				pass

			# If it is a new frame
			else:
				print('Saving frame number: ' + str(currentFrame))

				# Update current frame
				lastFrame = currentFrame

		# Choose the output file name using current frame name
		fileName = 'output/' + runDir + 'framewise_data/' + 'frame_' + str(currentFrame).zfill(4) + '.txt'

		# Open the file
		savefile = open(fileName, "a")

		# Read the x-position
		x_val = data['x'][i]

		# Read the y-position
		y_val = data['y'][i]

		# Read the orientation angle
		angle = data['angle'][i]

		# Read the defect number
		particle = data['particle'][i]

		# Add all these details to the line
		outLine = str(particle) + ' ' + str(x_val) + ' ' + str(y_val) +  ' ' + str(angle) + '\n'

		# Write the line
		savefile.write(outLine)

		# Close the file
		savefile.close()



### Save particlewise (defectwise) data
def save_particlewise(runDir, data):

	## Create the output directory
	os.mkdir('output/' + runDir + 'particlewise_data/')

	## Figure out the number of rows
	rows = len(data['particle'])

	## Use to keep track of total particles (defect)
	particles = 0

	## Go through every row
	for i in range(rows):

		# Read current particle
		currentParticle = data['particle'][i]

		# If we are on a new particle
		if currentParticle > particles:

			# Update the current particle count
			particles = currentParticle

	## Display how many particles there are in total
	print('Number of particles: '+str(particles))

	## Use to store current particle
	currentParticle = 0

	## Use to store previous particle
	lastParticle = 0

	## Go through every particle
	for i in range(particles):

		print('Saving particle: ' + str(i))

		# Path to file to store the particle data
		fileName = 'output/' + runDir + 'particlewise_data/' + 'particle_' + str(i).zfill(4) + '.txt'

		# Open the file
		savefile = open(fileName, "w")

		# For every row in the csv
		for j in range(rows):

			# If we are on the current particle
			if data['particle'][j] == i:

				# Read x-position
				x_val = data['x'][j]

				# Read y-position
				y_val = data['y'][j]

				# Read momentum
				angle = data['angle'][j]

				# Read current frame
				frame = data['frame'][j]

				# Create the line to store outputs
				outLine = str(frame) + ' ' + str(x_val) + ' ' + str(y_val) + ' ' + str(angle) + '\n'

				# Write the outputs to the file
				savefile.write(outLine)

		# Close the file
		savefile.close()



### Store the linked pandas csv to a better format
def better_storage(runDir):

	## Get file path to input csv linked
	inputFile = 'output/' + runDir + 'labels_pandas_linked_cutoff.csv'

	## Starting from stored linked data
	print('\nReading csv linked datafile.')
	linkedData = pd.read_csv(inputFile)

	print('\nSaving framewise data.')
	## Save framewise data from linked
	save_framewise(runDir, linkedData)

	print('\nSaving particlewise data..')
	## Save particlewise data from linked
	save_particlewise(runDir, linkedData)




### Detect and remove discontinuities
def make_cont(angles, show=False):

	outAngles = []
	outAngles.append(angles[0])

	iterations = 0

	for i, angleB in enumerate(angles):

		if i >= 1:

			# For the very first angle, use the fixed version
			angleA = outAngles[i-1]

			delta = abs(angleA - angleB)
			sign = angleA < angleB

			# General Case: No discontinouity
			# delta < 20
			# Sign is True or False

			# Case 1: The angle goes over 180
			# Angle A is just below 180
			# Angle B is just above 0

			# So, delta = ~180
			# Sign is negative
			# Add 180 to angle B

			# Case 2: The angle goes below 0
			# Angle A is just above 0
			# Angle B is just below 180

			# So, angle A - angle B = ~-180

			iterations = 0

			# Subtract 180 from angle B
			if delta < 50:

				outAngles.append(angleB)

			else:

				if not sign: # Case 1

					while delta > 50 and iterations < 15:

						iterations = iterations + 1

						if iterations == 15:
							return np.asarray(outAngles)

						angleB = angleB + 180

						delta = abs(angleA - angleB)

				else: # Case 2

					while delta > 50 and iterations < 15:

						iterations = iterations + 1

						if iterations == 15:
							return np.asarray(outAngles)

						angleB = angleB - 180

						delta = abs(angleA - angleB)

				outAngles.append(angleB)

	outAngles = np.asarray(outAngles)

	if show:

		plt.clf()

		plt.scatter(np.linspace(0, len(angles)-1, len(angles)), angles - 500, s=1)
		plt.scatter(np.linspace(0, len(angles)-1, len(angles)), outAngles, s=1)
		plt.legend(['original (500 subtracted)', 'fixed'])
		plt.show()
		plt.clf()

	if iterations >= 5:
		print('Iter:', iterations)

	return outAngles



### Helper to handle indices and logical indices of NaNs
def nan_helper(y):
	"""Helper to handle indices and logical indices of NaNs.

	Input:
		- y, 1d numpy array with possible NaNs
	Output:
		- nans, logical indices of NaNs
		- index, a function, with signature indices= index(logical_indices),
		  to convert logical indices of NaNs to 'equivalent' indices
	Example:
		>>> # linear interpolation of NaNs
		>>> nans, x= nan_helper(y)
		>>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
	"""

	return np.isnan(y), lambda z: z.nonzero()[0]



### Replace any missing frames with angle NaN
def replace_missing(frames, angles):

	# Convert the frames from float to integer
	frames = np.asarray(frames).astype(int)

	# List to store frames
	outFrames = []

	# List to store angles
	outAngles = []

	# Go through every frame that SHOULD be there
	for frame in range(frames[0], frames[-1]+1):

		# If the frame is actually there
		if frame in frames:

			# Append it to the output list
			outFrames.append(frame)

			# Append the corresponding angle to the output list
			outAngles.append(angles[np.where(frames == frame)[0][0]])

		# If the frame was not there
		else:

			# Add the frame
			outFrames.append(frame)

			# Use a NaN value as placeholder angle
			outAngles.append(np.nan)

	# Return the lists to the output
	return [np.asarray(outFrames), np.asarray(outAngles)]



### Get and store binary classifier data for each defect
def binary_data(runDir, minPoints=4):

	## Create output directory
	os.mkdir('output/' + runDir + 'binary_classifier_data/')

	## Read all input defects
	filePaths = glob.glob('output/' + runDir + 'particlewise_data/*.txt')

	## Go through every filepath
	for filePath in filePaths:

		# Particle number
		particle = int(filePath.split('.')[0].split('_')[-1])

		# Check if the file is empty
		if os.stat(filePath).st_size == 0:

			# Ignore and continue
			continue

		# If the file is not empty
		else:

			# Read file as numpy array
			inMat = np.loadtxt(filePath)

			# Check if it is a 2d array and at least minimum points exist
			if len(inMat.shape) == 2 and len(inMat) >= minPoints:

				# Transpose the matrix
				inMatTrans = inMat.T

				# Get individual list of frames
				frames = inMatTrans[0]

				# Get individual list of angles
				angles = inMatTrans[-1]

				# Fix any missing values
				fs_and_as = replace_missing(frames, angles)

				# Fetch frames and angles arrays again
				frames, angles =  fs_and_as[0], fs_and_as[1]

				# Clear list
				fs_and_as = None

				# Use NaN helper
				nans, x = nan_helper(angles)

				# Interpolate the missing values
				angles[nans] = np.interp(x(nans), x(~nans), angles[~nans])

				# Make the list of angles continuous by fixing 0-180 jumps
				angles = make_cont(angles, show=False)

				# Defect duration (length) in frames
				fLen = frames[-1] - frames[0]

				# Mean value of defect orientation
				meanAngle = np.mean(angles)

				# Mean value of defect orientation
				stdAngle = np.std(angles)

				# Out file name
				outFile = 'output/' + runDir + 'binary_classifier_data/' + str(particle).zfill(4) + '.txt'

				# Store the output to the dataset
				np.savetxt(outFile, np.asarray([meanAngle, stdAngle, fLen]))