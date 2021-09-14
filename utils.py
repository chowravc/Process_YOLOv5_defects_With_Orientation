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



### Get and store binary classifier data for each defect
def binary_data(runDir):

	## Create output directory
	os.mkdir('output/' + runDir + 'binary_classifier_data/')

	## Read all input defects
	filePaths = glob.glob('output/' + runDir + 'particlewise_data/*.txt')

	## Go through every filepath
	for filePath in filePaths:

		# Read file
		file = open(filePath, 'r')

		# Read lines from file
		text = file.read()

		# Close file
		file.close()
