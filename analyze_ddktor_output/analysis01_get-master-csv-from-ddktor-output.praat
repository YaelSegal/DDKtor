## Clear anything that's been done before
clearinfo

## Get main textgrid output directory
out_tg_directory$ = "../data/out_tg/"

##  Get a list of all directories in textgrid_directory
dir_list = Create Strings as directory list: "dir_list", "'out_tg_directory$'"

## Get a list of all the sound files directly in textgrid_directory
list = Create Strings as file list: "list", "'out_tg_directory$'/*.TextGrid"
number = Get number of strings
if number > 0
	@makeOutputCSV: "../data/out_summary/raw.csv", out_tg_directory$, list
endif
removeObject: list

## Loop through subdirectories in textgrid_directory and make CSV files for each of them
selectObject: dir_list
n_subdir = Get number of strings
for i to n_subdir
	selectObject: dir_list
	subdir$ = Get string... i
	subdir_fullpath$ = out_tg_directory$ + subdir$ + "/"
	list = Create Strings as file list: "list", "'subdir_fullpath$'/*.TextGrid"
	number = Get number of strings
	if number > 0
		@makeOutputCSV:  "../data/out_summary/" + subdir$ + "_raw.csv", subdir_fullpath$, list
	endif
	removeObject: list
endfor

removeObject: dir_list

## Function that takes a list of files and an output directory and makes a raw data CSV
procedure makeOutputCSV: .outputFileName$, .textgrid_directory$, .fileList

	# If the corresponding .csv file exists, start over
	deleteFile(.outputFileName$)
	resultfile$ = .outputFileName$
	resultline$ = "file,seg_type,seg_start,seg_end,seg_duration,window_type,window_start,window_end 'newline$'"
	fileappend "'resultfile$'" 'resultline$'

	selectObject: .fileList
	number = Get number of strings

	# Loop through textgrids in textgrid_directory
	for i to number
		selectObject: .fileList
		filename$ = Get string... i
		name$ = filename$ - ".TextGrid"

		# Read this textgrid file
		tg_file = Read from file: .textgrid_directory$ + filename$

		# Get the last tier
		lastTier = Get number of tiers

		# Get how many intervals are on the last tier of this textgrid file
		nInterval = Get number of intervals: lastTier

		# Loop through intervals
		for j to nInterval

			# Get the label and only do stuff if we're looking at a labeled interval
			label$ = Get label of interval: lastTier,j
			if label$ <> ""

				# Get start time, end time, and label of VOT
				.start = Get start time of interval: lastTier,j
				.end = Get end time of interval: lastTier,j
				.dur = .end - .start
				windowInt = Get interval at time: 1, (.start + .end)/2
				window$ = Get label of interval: 1, windowInt
				.window_start = Get start time of interval: 1, windowInt
				.window_end = Get end time of interval: 1, windowInt

				resultline$ = "'name$', 'label$', '.start', '.end', '.dur', 'window$', '.window_start', '.window_end' 'newline$'"
				fileappend "'resultfile$'" 'resultline$'

			endif
		endfor

	removeObject: tg_file
	endfor
endproc