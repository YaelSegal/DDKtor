
from boltons import fileutils
import os
from shutil import copyfile,SameFileError
import sys
import argparse
import soundfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.utilities import *
from helpers.textgrid import *
__author__ = 'YosiShrem'

files_dict_fname="files.txt"
SR = 16000
def main(args):
    try:
        parser = argparse.ArgumentParser(description='copy all wav files from all sub dirs to out_dir')
        parser.add_argument('--input_dir', type=str, help='Path of wavs dir', default="./data/raw")
        parser.add_argument('--use_textgrid', action='store_true', help='copy textgrid in order to use it windows')
        parser.add_argument('--output_dir', type=str, help='Path to output dir', default="./data/raw/all_files")
        args = parser.parse_args()


        assert os.path.exists(args.input_dir),f"Invalid Path, couldn't find [{args.input_dir}]"
        assert os.path.exists(args.output_dir),f"Invalid Path, couldn't find [{args.output_dir}]"

        wav_files = list(fileutils.iter_find_files(args.input_dir, "*.wav"))+list(fileutils.iter_find_files(args.input_dir, "*.WAV"))

        counter=0
        files_dict={}
        if len(wav_files) == 0:
            raise Exception("No wavs in {}".format(args.input_dir))
        has_files = False
        for filename in wav_files:
            has_textgrid = False
            if args.use_textgrid:
                textgrid_filename = filename.replace(".wav", ".TextGrid")
                if is_textgrid(textgrid_filename):
                    new_textgrid_name = os.path.join(args.output_dir,f"{counter}.TextGrid")
                    copyfile(textgrid_filename,new_textgrid_name)
                    has_textgrid = True         
            if (args.use_textgrid and has_textgrid) or not args.use_textgrid:
                has_files = True
                files_dict[counter] = filename
                if os.path.exists(os.path.join(args.output_dir,f"{counter}.wav")):
                    os.remove(os.path.join(args.output_dir,f"{counter}.wav"))
                new_name = os.path.join(args.output_dir,f"{counter}.wav")
                y, sr = soundfile.read(filename)
                if sr != SR:      
                    cmd = "sox -v 0.9 %s  -r 16000 -b 16 %s" % (filename, new_name)
                    easy_call(cmd)
                    print("convert {} from {} samples rate to {} samples rate".format(filename, sr, SR))
                else:
                    copyfile(filename,new_name)
                counter+=1
        if not has_files:
            raise Exception("No textgrid-wav pairs in {} ".format(args.input_dir))
        
        print(f"Finished to copy '*.wav' files to {args.output_dir}")
        with open(os.path.join(args.output_dir,files_dict_fname),'w') as f:
            f.write(f"input_dir : {args.input_dir}\n")
            f.write(f"output_dir : {args.output_dir}\n")
            for k,v in files_dict.items():
                f.write(f"{k}:{v}\n")
        print(f"Finished to write the files dictionary to {os.path.join(args.output_dir,files_dict_fname)}")


    except Exception as e:
        print(f"Failed to process the data, error {e}")
        exit(1) #FAIL




if __name__ == '__main__':
    main(sys.argv[1:])