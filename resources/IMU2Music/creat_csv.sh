#! /bin/bash

HEADER="vid,start_time,duration"

array=($(find ../../data/IMU2Music/audio/ -iname "*.wav"))

num_elements=${#array[@]}

echo $HEADER | tee train.csv val.csv test.csv > /dev/null

for (( i=0; i <$num_elements; i++ ))
do

	file_name=$(basename ${array[$i]} .wav)

	if [ ! -f "../../data/IMU2Music/midi/${file_name}.midi" ] || [ ! -s "../../data/IMU2Music/imu/${file_name}_acceleration_x1.csv" ]
	then
		continue
	fi

	duration=$(soxi -D ${array[$i]})
	line="$file_name,0.0,$duration"


	if [ $i -lt 53 ]
	then
		echo $line >> train.csv
	elif [ $i -lt 64 ]
	then
		echo $line >> val.csv
	else
		echo $line >> test.csv
	fi

done
