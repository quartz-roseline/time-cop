#!/bin/bash
python "$SUMO_HOME/tools/randomTrips.py" -n osm.net.xml --seed 42 --fringe-factor 5 -p 0.798773 -r osm.passenger.rou.xml -o osm.passenger.trips.xml -e 7200 --vehicle-class passenger --vclass passenger --prefix veh --min-distance 300 --trip-attributes 'speedDev="0.1" departLane="best"' --validate
