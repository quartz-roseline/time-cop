FROM ubuntu:16.04

# Install dependencies
RUN apt-get update -y && apt-get install -y software-properties-common && add-apt-repository -y ppa:sumo/stable \
    && apt-get update -y && apt-get install -y locales \
    && apt-get install -y python3-pip python3-dev \
    && apt-get install -y sumo sumo-tools \
    && pip3 install asyncio-nats-client \
    && pip3 install paho-mqtt \
    && pip3 install protobuf \
    && pip3 install pause \
    && pip3 install keras \
    && pip3 install tensorflow \
    && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

# Copy the helloworld-app
COPY binaries/traffic_controller_qot.py /usr/local/bin

# Copy the Library files
COPY binaries/qot_coreapi.py /usr/local/lib
COPY binaries/DQNmodels.py /usr/local/lib
COPY binaries/ring_buffer.py /usr/local/lib

# COPY the MQTT certificates
COPY certificates /

# Copy the Scenarios & Models
COPY Scenarios /usr/local/bin/Scenarios
COPY ScenarioModels /usr/local/bin/ScenarioModels

# Set the SUMO-HOME Environment Variable
ENV SUMO_HOME /usr/share/sumo

# Set to bin bash for interactive use, tail keeps the container running
#CMD ["/bin/bash"]
CMD tail -f /dev/null