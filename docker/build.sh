#!/bin/sh
# Copy binaries
mkdir binaries
echo Copying binaries, scenarios, models and libraries
cp ../src/traffic_controller_qot.py binaries/
cp ../src/DQNmodels.py binaries/
cp ../src/ring_buffer.py binaries/
cp ../src/lib/qot_coreapi.py binaries/
cp -r ../Scenarios .
cp -r ../ScenarioModels .

# Build the image
echo Building the Dev Image
docker build -t traffic-controller .
