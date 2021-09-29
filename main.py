### Importing useful packages
import cv2
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
from scipy.optimize import curve_fit


### Importing useful scripts
from utils import *

### Trackpy data analysis and storage portion of scipt
def tp_analysis(pathToInput):

	## Create output folder
	if not os.path.exists('output/'):
		os.makedirs('output/')
	os.mkdir('output/' + pathToInput)

	## Convert all YOLOv5 with angles label files to a single pandas file
	YOLOv5_to_pandas_file_all(pathToInput)

	## Save YOLOv5 files as JSON
	YOLOv5_as_JSON(pathToInput)

	## Link pandas csv with trackpy link
	link_pandas(pathToInput, maxDisp=10, maxMem=25, fMin=10)

	## Remove particles created after frame cutoff and cutoff the rest to start at frame cutoff
	fix_linking(pathToInput, cutoff=100)

	## Organize the linked csv into better, faster txt storage
	better_storage(pathToInput)

	## Get and store binary classifier data to be classified
	binary_data(pathToInput, minPoints=4)

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
		print(labelPath, dNum)
		
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



### Plot polarity on a single frame
def plot_polarity_on_frame(pathToInput, fNum):

	## Path to frame
	fPath = 'input/' + pathToInput + 'frames/' + pathToInput[:-1] + '_' + str(fNum).zfill(4) + '.tif'

	## Read frame as opencv image
	frame = cv2.imread(fPath)

	## Path to frame data
	fDataPath = 'output/' + pathToInput + 'framewise_data/frame_' + str(fNum).zfill(4) + '.txt'

	## Read frame data matrix
	fData = np.loadtxt(fDataPath)

	## Get transpose of data matrix
	fDataTrans = fData.T

	## Extract the 'particle' numbers
	pNums = fDataTrans[0]

	print(np.sort(pNums))

	## Path to polarity data
	polePath = 'output/' + pathToInput + 'polarities.txt'

	## Read polarity data
	poles = np.loadtxt(polePath)

	## Convert poles to transpose
	polesTrans = poles.T

	## Get list of particles from poles
	poleParts = polesTrans[0].astype(int)

	## Get list of poles
	polesList = polesTrans[1]

	## Clear plots
	plt.clf()

	## Use to store plusses
	plusses = 0

	## Use to store minuses
	minuses = 0

	## Use to store unknowns
	unknowns = 0

	## Go through every particle
	for i, pNum in enumerate(pNums):

		# Convert to integer
		pNum = int(pNum)

		# Find index of pNum in poleParts
		pIndex = np.where(poleParts == pNum)[0]

		# If the particle was not found
		if len(pIndex) == 0:

			# Make the polarity 0
			polarity = 0

		# If it was found
		else:

			# Get index
			pIndex = pIndex[0]

			# Get polarity
			polarity = polesList[pIndex]

		# Extract the x-pos
		x_pos = fData[i][1]

		# Extract the y-pos
		y_pos = fData[i][2]

		# Decide colour based on polarity
		if polarity == +1:
			pointColor = 'red'
			plusses += 1
		elif polarity == -1:
			pointColor = 'blue'
			minuses += 1
		else:
			pointColor = 'purple'
			unknowns += 1

		# Plot the defect
		plt.scatter(x_pos, y_pos, color=pointColor, s=1)

		# Plot the defect number
		plt.text(x_pos, y_pos, str(pNum), size=7)

	## Display results
	print('Plusses:', plusses)
	print('Minuses:', minuses)
	print('Unknowns:', unknowns)

	## Plot frame
	plt.imshow(frame)

	## Display plot
	plt.show()

	## Clear plots
	plt.clf()



### Store framewise nearest polar neighbour distance
def nn_polar_fwise(pathToInput):

	## Read paths to framewise data
	fWisePaths = glob.glob('output/' + pathToInput + 'framewise_data/*.txt')

	## Make new directory
	os.mkdir('output/' + pathToInput + 'nn_by_frame_defectwise/')

	## Path to polarity data
	polePath = 'output/' + pathToInput + 'polarities.txt'

	## Read polarity data
	poles = np.loadtxt(polePath)

	## Convert poles to transpose
	polesTrans = poles.T

	## Get list of particles from poles
	poleParts = polesTrans[0].astype(int)

	## Get list of poles
	polesList = polesTrans[1]

	## Go through every framewise data discovered
	for fDataPath in fWisePaths:

		print('\nNow on frame ' + fDataPath)

		## Path to output txt
		outTxt = 'output/' + pathToInput + 'nn_by_frame_defectwise/' + fDataPath.split('\\')[-1]

		## Store the output matrix
		outMat = []

		## Read frame data matrix
		fData = np.loadtxt(fDataPath)

		## Get transpose of data matrix
		fDataTrans = fData.T

		## Extract the 'particle' numbers
		pNums = fDataTrans[0]

		## Use to store completed particles
		completed = []

		## Store the output matrix
		outMat = []

		if len(fDataTrans.shape) < 2:
			continue

		## Go through every particle
		for i, pNumi in enumerate(pNums):

			## Store whether a line should be added
			addLine = False

			## Store the least distance discovered
			leastDist = 1e7

			## Store the current pair
			currentPair = 0

			## Check if the current particle was completed
			if int(pNumi) in completed:

				# Continue to next
				continue

			## Go through every pair particle
			for j, pNumj in enumerate(pNums):

				# If this pair was completed
				if int(pNumj) in completed:

					continue

				# Make sure it's not the same
				if i != j:

					# Convert to integer
					pNumi, pNumj = int(pNumi), int(pNumj)

					# Find index of pNumi in poleParts
					pIndexi = np.where(poleParts == pNumi)[0]

					# Find index of pNumj in poleParts
					pIndexj = np.where(poleParts == pNumj)[0]

					# If the particle i was not found
					if len(pIndexi) == 0:

						# Make the polarity 0
						polarityi = 0

					# If it was found
					else:

						# Get index
						pIndexi = pIndexi[0]

						# Get polarity
						polarityi = polesList[pIndexi]

					# If the particle j was not found
					if len(pIndexj) == 0:

						# Make the polarity 0
						polarityj = 0

					# If it was found
					else:

						# Get index
						pIndexj = pIndexj[0]

						# Get polarity
						polarityj = polesList[pIndexj]

					# If they are of the same polarity
					if polarityi*polarityj > 0:

						# This is wrong, so just continue
						continue

					# If the are not of same polarity (even if zero)
					else:

						# Extract the x-pos-i
						x_pos_i = fData[i][1]

						# Extract the y-pos-i
						y_pos_i = fData[i][2]

						# Extract the x-pos-j
						x_pos_j = fData[j][1]

						# Extract the y-pos-j
						y_pos_j = fData[j][2]

						# Calculate distance
						dist = np.sqrt((x_pos_i - x_pos_j)**2 + (y_pos_i - y_pos_j)**2)

						# If this is lesser than the last least distance
						if dist < leastDist:

							# Make this the new least distance
							leastDist = dist

							# Store the current j-particle to the pair
							currentPair = pNumj

			## If a pair was not found
			if leastDist == 1e7:

				print(pNumi, 'pair not found.')

			## If a pair was found
			else:

				# Add the pair to completed
				completed.append(pNumi)
				completed.append(currentPair)

				# Make a numpy array of the pair and their dist
				line = np.array([pNumi, currentPair, leastDist])

				# Add the line to the output matrix
				outMat.append(line)

		## Convert the outmat to numpy array
		outMat = np.asarray(outMat)

		## Save output matrix to numpy array
		np.savetxt(outTxt, outMat)



### Store average nearest neighbour distances
def nn_avg(pathToInput):

	## Output file path
	outPath = 'output/' + pathToInput + 'nn_avg.txt'

	## Read nn txt files
	nnTxtPaths = glob.glob('output/' + pathToInput + 'nn_by_frame_defectwise/*.txt')

	## Store fNums
	fNums = []

	## Store nn avg
	avgs = []

	## Go through every txt path
	for nnTxtPath in nnTxtPaths:

		# Load text transpose
		nnTxtTrans = np.loadtxt(nnTxtPath).T

		# Check if it has something inside
		if len(nnTxtTrans) > 0:

			# Read frame number
			fNum = int(nnTxtPath.split('\\')[-1].split('.')[0][6:])

			# Add to list of frames
			fNums.append(fNum)

			# Load nearest neighbours list
			nn = nnTxtTrans[-1]

			# Add the average to the list of average if there is more than 1
			if len(nnTxtTrans.shape) == 2:
				avgs.append(np.mean(nn))
			else:
				avgs.append(nn)

	## Convert fNums to numpy array
	fNums = np.asarray(fNums)

	## Convert nn avgs to numpy array
	avgs = np.asarray(avgs)

	## Add both to a numpy array
	outMat = np.asarray([fNums, avgs])

	plt.scatter(fNums, avgs, s=5)
	plt.loglog(basex=10, basey=10)
	plt.show()

	## Save it
	np.savetxt(outPath, outMat)



### Sqrt fit function
def sqrt_fit(x, k):

	return k*np.sqrt(x)



### Yurke fit function
def yurke_fit(y, k):

	return k*(y**2)*np.log(y)



### Plot single defect pair interdefect distance vs time
def interdefect(pathToInput, defect1, defect2, cutoff):

	## Scale bar
	umPerpx = 7/10/644*10**3

	## Read defect 1 txt
	defectMat1 = np.loadtxt('output/' + pathToInput + 'particlewise_data/particle_' + str(defect1).zfill(4) + '.txt')

	## If only single particle
	if len(defectMat1.shape) == 1:

		# Add an extra dimension
		defectMat1 = np.asarray([defectMat1])

	## Get transpose of defectMat1
	defectMatTrans1 = defectMat1.T

	## Get frames in defectMat1
	frames1 = defectMatTrans1[0].astype(int)

	## Read defect 2 txt
	defectMat2 = np.loadtxt('output/' + pathToInput + 'particlewise_data/particle_' + str(defect2).zfill(4) + '.txt')

	## If only single particle
	if len(defectMat2.shape) == 1:

		# Add an extra dimension
		defectMat2 = np.asarray([defectMat2])

	## Get transpose of defectMat1
	defectMatTrans2 = defectMat2.T

	## Get frames in defectMat2
	frames2 = defectMatTrans2[0].astype(int)

	## Use to store frame numbers
	frames = []

	## Use to store interdefect distance
	interdefectDist = []

	## Go through every frame in video
	for frame in range(100, 6078):

		# If the frame was recorded for both defects
		if frame in frames1 and frame in frames2:

			# Add it to the list of frames
			frames.append(frame)

			# Get index of frame in defect1 mat
			index1 = np.where(frames1 == frame)[0][0]

			# Get index of frame in defect1 mat
			index2 = np.where(frames2 == frame)[0][0]

			# Get distance between defects and add to list
			dist = np.sqrt((defectMat1[index1][1] - defectMat2[index2][1])**2 + (defectMat1[index1][2] - defectMat2[index2][2])**2)
			interdefectDist.append(dist)

	## Turn both frames and interdefect distance to numpy arrays
	frames = np.asarray(frames)
	interdefectDist = np.asarray(interdefectDist)

	## Subtract the frames from the last frame to reverse time
	frames = frames[-1] - frames

	## Convert to seconds
	seconds = frames/500

	## Convert pixels to micrometers
	interdefectDist = umPerpx*interdefectDist

	## Cut off the values at some starting time
	seconds = seconds[cutoff:]
	interdefectDist = interdefectDist[cutoff:]

	## Perform square root fit
	popt_sqrt, pcov_sqrt = curve_fit(sqrt_fit, seconds, interdefectDist)

	## Perform yurke fit (done in reverse)
	popt_yurke, pcov_yurke = curve_fit(yurke_fit, interdefectDist, seconds)

	## Plot results
	plt.plot(seconds, interdefectDist, color='k', label='Experimental Data')

	## Plot square root fit
	plt.plot(seconds, sqrt_fit(seconds, *popt_sqrt), 'r--', label='Sqrt fit: D = %5.3f\u221a(t)' % tuple(popt_sqrt))

	# print(type(popt_yurke[0]))

	## Plot square root fit
	plt.plot(yurke_fit(interdefectDist, *popt_yurke), interdefectDist, 'b--', label='Yurke fit: D\u00b2log(D) = t/' + str(1/popt_yurke[0])[:6])

	plt.title('Interdefect Separation vs. Time')

	plt.legend()

	plt.xlabel('time to annihilation (s)')
	plt.ylabel('D (interdefect distance) (\u03bcm)')

	plt.savefig('separation_vs_time_' + pathToInput[:-1] + '_defects_' + str(defect1) + '_' + str(defect2) + '.png', dpi=600)

	plt.show()



### Plot defect trajectory
def trajectory(pathToInput, defect1, defect2, cutoff, w=7):

	## Scale bar
	umPerpx = 7/10/644*10**3

	## Read defect 1 txt
	defectMat1 = np.loadtxt('output/' + pathToInput + 'particlewise_data/particle_' + str(defect1).zfill(4) + '.txt')

	## If only single particle
	if len(defectMat1.shape) == 1:

		# Add an extra dimension
		defectMat1 = np.asarray([defectMat1])

	## Get transpose of defectMat1
	defectMatTrans1 = defectMat1.T

	## Get xs in defectMat1
	xs1 = np.convolve(defectMatTrans1[1][:], np.ones(w), 'valid')/w

	## Get ys in defectMat1
	ys1 = np.convolve(defectMatTrans1[2][:], np.ones(w), 'valid')/w

	## Read defect 2 txt
	defectMat2 = np.loadtxt('output/' + pathToInput + 'particlewise_data/particle_' + str(defect2).zfill(4) + '.txt')

	## If only single particle
	if len(defectMat2.shape) == 1:

		# Add an extra dimension
		defectMat2 = np.asarray([defectMat2])

	## Get transpose of defectMat1
	defectMatTrans2 = defectMat2.T

	## Get xs in defectMat2
	xs2 = np.convolve(defectMatTrans2[1][:], np.ones(w), 'valid')/w

	## Get ys in defectMat1
	ys2 = np.convolve(defectMatTrans2[2][:], np.ones(w), 'valid')/w

	fig, axs = plt.subplots()

	axs.plot(xs1*umPerpx, ys1*umPerpx, color='b', label='Negative Defect')

	axs.plot(xs2*umPerpx, ys2*umPerpx, color='r', label='Positive Defect')

	plt.xlabel('x-position (\u03bcm)')
	plt.ylabel('y-position (\u03bcm)')

	plt.title('Annihilating defects trajectory')

	axs.set_aspect('equal', 'box')

	axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.savefig('trajectory_' + pathToInput[:-1] + '_defects_' + str(defect1) + '_' + str(defect2) + '.png', dpi=600)

	plt.show()




### Call if script is run directly
if __name__ == '__main__':

	## Input directory inside 'input/'
	pathToInput = 'r4/'
	
	## Call trackpy analysis
	# tp_analysis(pathToInput)

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

	## Plot all polarities on a single frame
	fNum = 2000

	# plot_polarity_on_frame(pathToInput, fNum)

	## Store nearest neighbour distance with particles
	# nn_polar_fwise(pathToInput)

	## Store average nearest neighbour distances
	# nn_avg(pathToInput)

	## Good interedefect pairs (minus, plus)
	# r1:
	#

	# r2: 
	# Good, but does not annihilate and actually goes apart first (600, 680)
	# Good, and annihilates but may have to cut off start (214, 335), cutoff 3000

	# r3:
	# This pair annihilated only after the pair before it has annihilated (84, 697), cutoff 3500
	# This pair stops the ones after it from annihilating until they have annihilated (172, 433), cutoff 700
	
	# r4:
	# This pair annihilates quite quickly. It has a strange starting angle where the  (259, 847), cutoff 0

	interdefect(pathToInput, 259, 847, 000)
	trajectory(pathToInput, 259, 847, 0)