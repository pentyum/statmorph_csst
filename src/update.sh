#!/bin/bash
CONFIG_FILE=config.properties
mv $CONFIG_FILE "${CONFIG_FILE}.bak"
cd ../
git pull
./build.sh
cd ./src
if test -f "$CONFIG_FILE"; then
  echo "配置文件已更新"
else
  mv "${CONFIG_FILE}.bak" $CONFIG_FILE
  echo "配置文件未更新"
fi
