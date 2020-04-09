import numpy
import matplotlib.pyplot as plt

from kalman import predict, update


''' signal parameters '''

samplingFreq = 1000
numOfSamples = 500

signalFreq = 2      # frequency of real signal in Hz

stdev1 = 0.05       # standard deviation of sensor 1
stdev2 = 0.07      # standard deviation of sensor 2
stdev3 = 0.055      # standard deviation of sensor 3

''' data init '''
dt = 1 / samplingFreq

timeArray = [0] * numOfSamples

realSignalInput = [0] * numOfSamples
inputArrSensor1 = [0] * numOfSamples
inputArrSensor2 = [0] * numOfSamples
inputArrSensor3 = [0] * numOfSamples

filterOutput = [0] * numOfSamples

for n in range(numOfSamples):
    timeArray[n] = n * dt
    realSignalInput[n] = numpy.sin(2*numpy.pi*signalFreq*timeArray[n])
    inputArrSensor1[n] = realSignalInput[n] + numpy.random.normal(0, stdev1)
    inputArrSensor2[n] = realSignalInput[n] + numpy.random.normal(0, stdev2)
    inputArrSensor3[n] = realSignalInput[n] + numpy.random.normal(0, stdev3)

''' Kalman filter init '''

p = 0
f = 1
x = 0
h = 1
r1 = stdev1**2
r2 = stdev2**2
r3 = stdev3**2

q = 0.0012*dt

''' Filtering '''

for n in range(1, numOfSamples):
    x, p = predict(x, p, f, q)
    x, p = update(x, p, h, r1, inputArrSensor1[n])
    x, p = update(x, p, h, r2, inputArrSensor2[n])
    x, p = update(x, p, h, r3, inputArrSensor3[n])

    filterOutput[n] = x

plt.plot(timeArray, realSignalInput, color='k', label='real signal')
plt.plot(timeArray, inputArrSensor1, color='b',
         linestyle='--', label='sensor 1')
plt.plot(timeArray, inputArrSensor2, color='g',
         linestyle='--', label='sensor 2')
plt.plot(timeArray, inputArrSensor3, color='c',
         linestyle='--', label='sensor 3')
plt.plot(timeArray, filterOutput, color='r', label='Kalman')

plt.legend()
plt.xlabel('Time [s]')
axes = plt.gca()
axes.set_xlim([0, timeArray[-1]])

plt.show()

''' Visualization '''
