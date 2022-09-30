#!/bin/bash
mv config.properties config.properties.bak
cd ../
git pull
./build.sh
cd ./src
