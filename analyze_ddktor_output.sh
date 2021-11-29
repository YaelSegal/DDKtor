#!/bin/sh

#creating dirs
mkdir ./data/out_summary

# echo "============"
echo "Running analysis01_get-master-csv-from-ddktor-output.praat"
# echo "============"

if [[ $OSTYPE == 'darwin'* ]]; then
    /Applications/Praat.app/Contents/MacOS/Praat --run "analyze_ddktor_output/analysis01_get-master-csv-from-ddktor-output.praat"
else
    "C:\Program Files\Praat.exe" --run "analyze_ddktor_output/analysis01_get-master-csv-from-ddktor-output.praat"
fi

echo "Running analysis02_process-master-csv.R"

cd analyze_ddktor_output
Rscript "analysis02_process-master-csv.R"

echo "Running analysis03_calculate-final-measures.R"

Rscript "analysis03_calculate-final-measures.R"
cd ..

# rm ./data/out_summary/raw.csv
rm ./data/out_summary/ddkrates.csv
rm ./data/out_summary/segments.csv
rm ./data/out_summary/syllables.csv
rm ./data/out_summary/raw.csv

echo "Finished: final data summaries can be found at : ./data/out_summary/"


# echo "============"
# echo "Step 0: Installing dependencies in a virtual environment(It doesn't change your settings)"
# echo "============"

# python3 -m pip install --user virtualenv
# python3 -m venv env
# source env/bin/activate
# pip install -r requirements.txt


# if [ "$2" = true ]; then
#     echo "============"
#     echo "Step 1: Preparing the data - with noise-reduction"
#     echo "============"
#     python ./process_data/prepare_wav_dir.py --input_dir ./data/raw --output_dir ./data/raw/all_files --use_textgrid --clean_noise
# else
#     echo "============"
#     echo "Step 1: Preparing the data - without noise-reduction"
#     echo "============"
#     python ./process_data/prepare_wav_dir.py --input_dir ./data/raw --output_dir ./data/raw/all_files --use_textgrid 
# fi


# if [ $? -eq 1 ]; then
#     echo "Failed to collect the data, check run_log.txt"
#     exit 1
# fi

# echo "============"
# echo "Step 2: Processing sound files...(may take a while - approx. 1 sec per file)"
# echo "============"
# python prepare_data_textgrid.py --input_dir ./data/raw/all_files --output_dir ./data/processed --windows_tier $1
# if [ $? -eq 1 ]; then
#     echo "Failed to process the data, check run_window_log.txt"
#     exit 1
# fi

# echo "============"
# echo "Step 3: Running DDKtor"
# echo "============"

# python predict.py --data ./data/processed/ --out_dir ./data/out_tg/tmp_parts --cuda
# if [ $? -eq 1 ]; then
#     echo "Failed to run DDKtor predict, check run_window_log.txt"
#     exit 1
# fi


# echo "============"
# echo "Step 4: Final process"
# echo "============"


# python merge_windows_textgrids.py --input_dir ./data/out_tg/tmp_parts --output_dir ./data/out_tg --pred_tier preds  --durations ./data/raw/all_files/voice_starts.txt --basic_hierarchy_file ./data/raw/all_files/files.txt --use_prev_textgrid
# if [ $? -eq 1 ]; then
#     echo "Failed to run DDKtor merge , check run_log.txt"
#     exit 1
# fi


# echo "============"
# echo "============"
# echo "============"
# echo "Finished: final predictions can be found at : ./data/out_tg/"
# echo "============"
# echo "============"
# echo "============"


# F_cleanup

