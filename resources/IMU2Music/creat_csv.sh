#! /bin/bash

HEADER="vid,start_time,duration"

MIN_LEN=6

array=($(find ../../data/IMU2Music/audio/ -iname "*.wav"))

new_array=()
TOTAL_LEN=0

for i in "${array[@]}"
do
        FILE_LEN=$(soxi -D $i)
        
        if (( $(echo "$FILE_LEN < $MIN_LEN" |bc -l) ))
	then
		continue
	fi
	

	new_array+=($i)
	TOTAL_LEN=$(echo "$TOTAL_LEN + $FILE_LEN" | bc -l)

done

TRAIN_LIMIT=$(echo "$TOTAL_LEN*0.8" | bc -l)
VAL_LIMIT=$(echo "$TRAIN_LIMIT + $TOTAL_LEN*0.1" | bc -l)

num_elements=${#new_array[@]}

echo $HEADER | tee train.csv val.csv test.csv > /dev/null

ACC_LEN=0

for (( i=0; i <$num_elements; i++ ))
do

	file_name=$(basename ${new_array[$i]} .wav)

	duration=$(soxi -D ${new_array[$i]})
	line="$file_name,0.0,$duration"
	
	ACC_LEN=$(echo "$ACC_LEN + $duration" | bc -l)


	if (( $(echo "$ACC_LEN < $TRAIN_LIMIT" |bc -l) ))
	then
		echo $line >> train.csv
	elif (( $(echo "$ACC_LEN < $VAL_LIMIT" |bc -l) ))
	then
		echo $line >> val.csv
	else
		echo $line >> test.csv
	fi

done


