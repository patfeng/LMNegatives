FOLDER=DATA/RP

mkdir -p $FOLDER
for number in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 # sample with 40 processes
do
   python sample/sample.py --vocab_file sample/vocab.txt --output_file $FOLDER/prop_examples_$number.txt --min_pred_num 5 --max_pred_num 30 --algo RP --example_num 1000 --balance_by_depth --max_depth 6 & 
done
wait
<<<<<<< HEAD
>>>>>>> 9ecafc6bce621edf4d76a439ec0f541ec1eeb10e
