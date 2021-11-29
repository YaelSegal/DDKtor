# Read in output of DeepDDK
df <- read.csv(paste0("../data/out_summary/", "raw.csv"))
datasource <- 'asd-nr'

############# FUNCTIONS ###############
# Function to check if seg_type is a VOT or not
is_vot <- function(seg_type) {
  return(grepl('Vot', seg_type, fixed = TRUE) || grepl('VOT', seg_type, fixed = TRUE))
}

# Function that outputs syll_num and full_syll lists to be added to the main df
get_syll_info <- function(df) {
  expecting_vot = 1
  current_syll = 1
  syll_num <- c(NA,nrow(df))
  full_syll <- rep(NA,nrow(df))
  for (i in 1:nrow(df)) {
    if (expecting_vot == TRUE && is_vot(df[i,"seg_type"]) == TRUE) {
      # If it's the last one, update everything
      if (i == nrow(df)) {
        syll_num[i] <- current_syll
        full_syll[i] <- 0
      }
      expecting_vot <- FALSE
    } else if (expecting_vot == TRUE && is_vot(df[i,"seg_type"]) == FALSE) {
      syll_num[i] <- current_syll
      full_syll[i] <- 0
      current_syll <- current_syll + 1
    } else if (expecting_vot == FALSE && is_vot(df[i,"seg_type"]) == FALSE) {
      # The VOT ends at the same time as the Vowel starts
      if (df[i-1,"seg_end"] == df[i, "seg_start"]) {
        syll_num[i-1] <- current_syll
        syll_num[i] <- current_syll
        current_syll <- current_syll + 1
        full_syll[i-1] <- 1
        full_syll[i] <- 1
        expecting_vot <- TRUE
      } else{
        # The VOT and Vowels start/end at different times -> treat as 2 different syllables
        syll_num[i-1] <- current_syll
        full_syll[i-1] <- 0
        current_syll <- current_syll + 1
        syll_num[i] <- current_syll
        full_syll[i] <- 0
        current_syll <- current_syll + 1
        expecting_vot <- TRUE
      }
    } else if (expecting_vot == FALSE && is_vot(df[i, "seg_type"]) == TRUE) {
      syll_num[i-1] <- current_syll
      full_syll[i-1] <- 0
      current_syll <- current_syll + 1
      # If it's the last one, update everything
      if (i == nrow(df)) {
        syll_num[i] <- current_syll
        full_syll[i] <- 0
      }
    } else {
      "Something weird has happened!"
    }
  }
  return(list("syll_num" = syll_num, "full_syll" = full_syll))
}

# Function to count how many syllables are present
get_nSyll <- function(vec) {
  expecting_vot = 1
  nsyll = 0
  for (i in 1:length(vec)) {
    if (expecting_vot == 1 && grepl("VOT", toupper(vec[i]))) {
      expecting_vot = 0
    }
    else if (expecting_vot == 1 && grepl("Vowel", vec[i], fixed = TRUE)) {
      nsyll = nsyll + 1
    }
    else if (expecting_vot == 0 && grepl("Vowel", vec[i], fixed = TRUE)) {
      nsyll = nsyll + 1
      expecting_vot = 1
    }
    else if (expecting_vot == 0 && grepl("VOT", toupper(vec[i]))) {
      nsyll = nsyll + 1
    }
    else {
      print("Something weird has happened!")
    }
  }
  return(nsyll)
}

# This function adds syllables for particularly long vowels
correct_merged_syllables <- function(df, threshold) {
  vowels <- subset(df, trimws(seg_type) == 'Vowel')
  n_longvowels <- 0
  for (i in 1:length(vowels$seg_duration)) {
    if (vowels$seg_duration[i] > threshold) {
      n_longvowels <- n_longvowels + 1
    }
  }
  return(n_longvowels)
}

# Function that gets rid of the top/bottom X percent along some variable
get_cutoff <- function(var, topbottom, percent) {
  if (topbottom == 'bottom') {
    temp <- sort(var, decreasing = FALSE)
  }
  else {
    temp <- sort(var, decreasing = TRUE)
  }
  row_to_query <- floor(percent * length(var))
  return(temp[row_to_query])
}
############################

# Get syllable indices and whether they are full syllables (VOT + Vowel) or not
df$syll_num <- rep(NA, nrow(df))
df$full_syll <- rep(NA, nrow(df))

# Get a list of all of the files to analyze
files <- levels(factor(df$file))
curr_start <- 1
for (i in 1:length(files)) {
  # Loop through each file
  subfile <- subset(df, file == files[i])
  # Get a list of each window in it
  windows <- levels(factor(subfile$window_start))
  for (j in 1:length(windows)) {
    # Loop through each of the windows
    current_subset <- subset(subfile, window_start == windows[j])
    syll_info <- get_syll_info(current_subset)
    curr_end <- curr_start + nrow(current_subset) - 1
    
    df[curr_start:curr_end, "syll_num"] <- syll_info$syll_num
    df[curr_start:curr_end, "full_syll"] <- syll_info$full_syll
    curr_start <- curr_start + nrow(current_subset)
  }
}

# Get most extreme values (consonont, vowel durations) ACROSS PARTICIPANTS
vot_subset <- subset(df, grepl('Vot', seg_type, fixed = TRUE)) 
vowel_subset <- subset(df, grepl('Vowel', seg_type, fixed = TRUE))

vot_lower_cutoff <- get_cutoff(vot_subset$seg_duration, 'bottom', 0.02)
vot_upper_cutoff <- get_cutoff(vot_subset$seg_duration, 'top', 0.05)
vowel_lower_cutoff <- get_cutoff(vowel_subset$seg_duration, 'bottom', 0.02)
vowel_upper_cutoff <- get_cutoff(vowel_subset$seg_duration, 'top', 0.05)

df$extreme_value_acrossparts <- rep(0, nrow(df))

for (i in 1:nrow(df)) {
  if (grepl('Vot', df[i,'seg_type'], fixed = TRUE) == TRUE) {
    if (df[i, 'seg_duration'] < vot_lower_cutoff || df[i, 'seg_duration'] > vot_upper_cutoff) {
      df[i, 'extreme_value_acrossparts'] <- 1
    }
  } else if (grepl('Vowel', df[i,'seg_type'], fixed = TRUE) == TRUE) {
    if (df[i, 'seg_duration'] < vowel_lower_cutoff || df[i, 'seg_duration'] > vowel_upper_cutoff) {
      df[i, 'extreme_value_acrossparts'] <- 1
    }
  }
}

# Get most extreme values WITHIN participants
df$extreme_value_withinparts <- rep(0, nrow(df))

parts <- levels(factor(df$file))

vot_lower_cutoffs <- rep(NA, length(parts))
vot_upper_cutoffs <- rep(NA, length(parts))
vowel_lower_cutoffs <- rep(NA, length(parts))
vowel_upper_cutoffs <- rep(NA, length(parts))

for (i in 1:length(parts)) {
  subfile <- subset(df, file == parts[i])
  vot_subset <- subset(subfile, grepl('Vot', seg_type, fixed = TRUE)) 
  vowel_subset <- subset(subfile, grepl('Vowel', seg_type, fixed = TRUE))
  
  vot_lower_cutoffs[i] <- get_cutoff(vot_subset$seg_duration, 'bottom', 0.02)
  vot_upper_cutoffs[i] <- get_cutoff(vot_subset$seg_duration, 'top', 0.05)
  vowel_lower_cutoffs[i] <- get_cutoff(vowel_subset$seg_duration, 'bottom', 0.02)
  vowel_upper_cutoffs[i] <- get_cutoff(vowel_subset$seg_duration, 'top', 0.05)
}

curr_start <- 1
for (i in 1:length(parts)) {
  # Loop through each file
  subfile <- subset(df, file == parts[i])
  extremevals <- rep(0, nrow(subfile))
  
  vot_subset <- subset(subfile, grepl('Vot', seg_type, fixed = TRUE)) 
  vowel_subset <- subset(subfile, grepl('Vowel', seg_type, fixed = TRUE))
  
  vot_lower_cutoff <- get_cutoff(vot_subset$seg_duration, 'bottom', 0.02)
  vot_upper_cutoff <- get_cutoff(vot_subset$seg_duration, 'top', 0.05)
  vowel_lower_cutoff <- get_cutoff(vowel_subset$seg_duration, 'bottom', 0.02)
  vowel_upper_cutoff <- get_cutoff(vowel_subset$seg_duration, 'top', 0.05)
  
  for (j in 1:nrow(subfile)) {
    if (is_vot(subfile[j,'seg_type']) == TRUE) {
      if (subfile[j, 'seg_duration'] < vot_lower_cutoff || subfile[j, 'seg_duration'] > vot_upper_cutoff) {
        extremevals[j] <- 1
      }
    } else if (grepl('Vowel', subfile[j,'seg_type'], fixed = TRUE) == TRUE) {
      if (subfile[j, 'seg_duration'] < vowel_lower_cutoff || subfile[j, 'seg_duration'] > vowel_upper_cutoff) {
        extremevals[j] <- 1
      }
    }
  }
  curr_end <- curr_start + nrow(subfile) - 1
  df[curr_start:curr_end, "extreme_value_withinparts"] <- extremevals
  curr_start <- curr_start + nrow(subfile)
}

# Get syllables that are in the middle of the trial
df$middle_syll <- rep(NA, nrow(df))

# Get a list of all of the files: 001_rate_alternate, 001_rate_single, etc.
files <- levels(factor(df$file))
curr_start <- 1
for (i in 1:length(files)) {
  # Loop through each file
  subfile <- subset(df, file == files[i])
  # Get a list of each window in it
  windows <- levels(factor(subfile$window_start))
  for (j in 1:length(windows)) {
    # Loop through each of the windows
    current_subset <- subset(subfile, window_start == windows[j])
    middle_syll_vec <- rep(0, nrow(current_subset))
    n_syll <- max(current_subset$syll_num)
    if (n_syll <= 10) {
      # print(files[i])
      middle_syll_vec <- rep(1, nrow(current_subset))
    } else {
      middle_syll_n <- n_syll / 2.0
      if (floor(middle_syll_n) == ceiling(middle_syll_n)) {
        min_syll_n <- floor(middle_syll_n) - 5
        max_syll_n <- ceiling(middle_syll_n) + 4
      } else {
        min_syll_n <- floor(middle_syll_n) - 4
        max_syll_n <- ceiling(middle_syll_n) + 4
      }
      
      for (k in 1:nrow(current_subset)) {
        if (current_subset[k,"syll_num"] >= min_syll_n && current_subset[k,"syll_num"] <= max_syll_n) {
          middle_syll_vec[k] <- 1
        }
      }
    }
    
    curr_end <- curr_start + nrow(current_subset) - 1
    df[curr_start:curr_end, "middle_syll"] <- middle_syll_vec
    curr_start <- curr_start + nrow(current_subset)
  }
}

# True if annotator marked the trial with ?
df$peculiar <- grepl('?', df$window_type, fixed = TRUE)

# Get measure of whether each VOT/vowel is 2*mean(vowel duration) - vowel2long
df$vowel2long <- rep(0, nrow(df))
for (i in 1:nrow(df)) {
  subfile <- subset(df, file == df[i,"file"])
  vowel_threshold <- 2*mean(subset(subfile, trimws(seg_type) == 'Vowel')$seg_duration)
  if (trimws(df[i,"seg_type"]) == 'Vowel' & df[i,"seg_duration"] > vowel_threshold) {
    df[i,"vowel2long"] <- 1
  }
  if (trimws(df[i,"seg_type"]) == "Vot") {
    syllsub <- subset(subfile, syll_num == df[i,"syll_num"])
    winsub <- subset(syllsub, window_start == df[i,"window_start"])
    finalsub <- subset(winsub, trimws(seg_type) == 'Vowel')
    if (nrow(finalsub) > 1) {
      print("Too many vowels predicted!")
    } else if (nrow(finalsub) == 1) {
      if (finalsub[1,"seg_duration"] > vowel_threshold) {
        df[i,"vowel2long"] <- 1
      }
    }
  }
}

## Make syllables data frame that has one row for each syllable
total_new_rows <- nrow(subset(df,full_syll==1))/2 + nrow(subset(df,full_syll==0))

file <- rep("", total_new_rows)
syll_start <- rep(NA, total_new_rows)
syll_end <- rep(NA, total_new_rows)
syll_duration <- rep(NA, total_new_rows)
window_type <- rep("", total_new_rows)
window_start <- rep(NA, total_new_rows)
window_end <- rep(NA, total_new_rows)
syll_num <- rep(NA, total_new_rows)
full_syll <- rep(NA, total_new_rows)
middle_syll <- rep(NA, total_new_rows)
vowel2long <- rep(NA, total_new_rows)
extreme_value_withinparts <- rep(NA, total_new_rows)
follows_extreme_value <- rep(0, total_new_rows)

syllables <- data.frame(file, syll_start, syll_end, syll_duration, window_type, window_start, window_end, syll_num, full_syll, middle_syll, vowel2long, extreme_value_withinparts, follows_extreme_value, stringsAsFactors=FALSE)

dffile_aschar <- as.character(df$file)
dfwindow_type_aschar <- as.character(df$window_type)

i <- 1
j <- 1
while (i <= nrow(df)) {
  syllables[j,"file"] <- dffile_aschar[i] # as.character(df[i,"file"])
  syllables[j,"syll_start"] <- df[i,"seg_start"]
  syllables[j,"window_type"] <- dfwindow_type_aschar[i] # as.character(df[i,"window_type"])
  syllables[j,"window_start"] <- df[i,"window_start"]
  syllables[j,"window_end"] <- df[i,"window_end"]
  syllables[j,"syll_num"] <- df[i,"syll_num"]
  syllables[j,"full_syll"] <- df[i,"full_syll"]
  syllables[j,"middle_syll"] <- df[i,"middle_syll"]
  syllables[j,"vowel2long"] <- df[i,"vowel2long"]
  if (df[i,'full_syll'] == 1) {
    syllables[j,"syll_end"] <- df[i+1,"seg_end"]
    syllables[j,"syll_duration"] <- df[i+1,"seg_end"]-df[i,"seg_start"]
    syllables[j,"extreme_value_withinparts"] <- max(df[i,"extreme_value_withinparts"], df[i+1,"extreme_value_withinparts"])
    if (syllables[j,"extreme_value_withinparts"] == 1 & j+1 <= nrow(syllables)) {
      syllables[j+1,"follows_extreme_value"] <- 1
    }
    i <- i + 2
    j <- j + 1
  } else {
    syllables[j,"syll_end"] <- df[i,"seg_end"]
    syllables[j,"syll_duration"] <- df[i,"seg_end"]-df[i,"seg_start"]
    syllables[j,"extreme_value_withinparts"] <- df[i,"extreme_value_withinparts"]
    if (syllables[j,"extreme_value_withinparts"] == 1 & j+1 <= nrow(syllables)) {
      syllables[j+1,"follows_extreme_value"] <- 1
    }
    i <- i + 1
    j <- j + 1
  }
}

# Add intersyllable duration
syllables$intersyll_duration <- rep(NA, nrow(syllables))

# Get a list of all of the files: 001_rate_alternate, 001_rate_single, etc.
files <- levels(factor(syllables$file))
curr_start <- 1
for (i in 1:length(files)) {
  # Loop through each file
  subfile <- subset(syllables, file == files[i])
  # Get a list of each window in it
  windows <- levels(factor(subfile$window_start))
  for (j in 1:length(windows)) {
    # Loop through each of the windows
    current_subset <- subset(subfile, window_start == windows[j])
    
    intersyll <- c()
    
    for (k in 1:nrow(current_subset)) {
      if (k == 1) {
        intersyll[k] <- NA
      } else {
        intersyll[k] <- current_subset[k,"syll_start"]-current_subset[k-1,"syll_end"]
      }
    }
    curr_end <- curr_start + nrow(current_subset) - 1
    syllables[curr_start:curr_end, "intersyll_duration"] <- intersyll
    curr_start <- curr_start + nrow(current_subset)
  }
}


## Make DDK rate data frame
file <- c()
win_start <- c()
win_end <- c()

files <- levels(factor(df$file))
for (i in 1:length(files)) {
  # Loop through each file
  subfile <- subset(df, file == files[i])
  # Get a list of each window in it
  window_starts <- levels(factor(subfile$window_start))
  window_ends <- levels(factor(subfile$window_end))
  
  if (length(window_starts) != length(window_ends)) {
    print("Windows do not line up")
    print(files[i])
  }
  
  file <- c(file, rep(files[i], length(window_starts)))
  win_start <- c(win_start, window_starts)
  win_end <- c(win_end, window_ends)
  
  if (length(unique(length(file), length(win_start), length(win_end))) != 1) {
    print("Windows do not line up!")
    print(files[i])
  }
}

win_type <- rep(NA, length(win_start))
nsyll <- rep(NA, length(win_start))
speech_start <- rep(NA, length(win_start))
speech_end <- rep(NA, length(win_start))
speech_dur <- rep(NA, length(win_start))
ddk_rate <- rep(NA, length(start))
mean_vot_dur <- rep(NA, length(start))
mean_vowel_dur <- rep(NA, length(start))

windows <- data.frame(file, win_start, win_end, win_type, nsyll, speech_start, speech_end, speech_dur, ddk_rate)
amr_list <- c('pa', 'ta', 'ka', 'p', 't', 'k', 'ba', 'da', 'ga', 'b', 'd', 'g')

for (i in 1:nrow(windows)) {
  sub <- subset(df, df$file == windows[i,'file'] & df$seg_end > as.numeric(as.character(windows[i,'win_start'])) & df$seg_start < as.numeric(as.character(windows[i, 'win_end'])))
  sub <- sub[order(sub['seg_start']),]
  windows[i, 'win_type'] <- trimws(sub[1,'window_type'])
  windows[i, 'nsyll'] <- max(sub$syll_num)
  
  # Correct nsyll to account for particularly long vowels
  part_vowels <- subset(df, df$file == windows[i,'file'] & trimws(df$seg_type) == 'Vowel')
  windows[i, 'nsyll'] <- windows[i, 'nsyll'] + correct_merged_syllables(sub, 2*mean(part_vowels$seg_duration))
  
  windows[i, 'speech_start'] <- sub[1, 'seg_start']
  windows[i, 'speech_end'] <- sub[nrow(sub), 'seg_end']
  windows[i, 'speech_dur'] <- windows[i, 'speech_end'] - windows[i, 'speech_start']
  windows[i, 'ddk_rate'] <- windows[i, 'nsyll'] / windows[i, 'speech_dur']
  
  windows[i, 'mean_vot_dur'] <- mean(subset(sub, trimws(seg_type) == 'Vot')$seg_duration)
  windows[i, 'mean_vowel_dur']<- mean(subset(sub, trimws(seg_type) == 'Vowel')$seg_duration)
  # TODO: check whether there is punctuation
  windows[i, 'subtask'] <- ifelse(windows[i,'win_type'] %in% amr_list, 'amr', 'smr')

  percent_threshold <- 0.20
  if (windows[i, 'subtask'] == 'smr') {
    windows[i, 'ddktor_count_close'] <- ifelse(abs(windows[i,'nsyll'] - 30) < percent_threshold * 30,1,0)
  } else {
    windows[i, 'ddktor_count_close'] <- ifelse(abs(windows[i,'nsyll'] - 15) < percent_threshold * 15,1,0)
  }
}


voiced <- c()

for (i in 1:nrow(windows)) {
    windows[i,"voiced"] <- ifelse(substring(windows[i,"win_type"],1,1) %in% c('b', 'd', 'g', 'ba', 'da', 'ga'), 1, 0)
}

write.csv(windows, paste0("../data/out_summary/ddkrates.csv"), quote = FALSE, row.names = FALSE)


for (i in 1:nrow(df)) {
  relevant_window <- subset(windows, file == df[i,'file'] & speech_start < df[i,'seg_end'] & speech_end > df[i,'seg_start'])
  if (nrow(relevant_window) != 1) {
    print("Wrong number of rows detected!")
  }
  if (relevant_window$ddktor_count_close == 1) {
    df[i,'ddktor_count_close'] <- 1
  } else {
    df[i,'ddktor_count_close'] <- 0
  }
  if (relevant_window$voiced == 1) {
    df[i,'voiced'] <- 1
  } else {
    df[i,'voiced'] <- 0
  }
  df[i,'subtask'] <- relevant_window$subtask
  df[i,'nsyll_in_trial'] <- relevant_window$nsyll
}

for (i in 1:nrow(syllables)) {
  relevant_window <- subset(windows, file == syllables[i,'file'] & win_start == syllables[i,'window_start'])
  if (nrow(relevant_window) != 1) {
    View(relevant_window)
    print(i)
    print("Wrong number of rows detected!")
  }
  if (relevant_window$ddktor_count_close == 1) {
    syllables[i,'ddktor_count_close'] <- 1
  } else {
    syllables[i,'ddktor_count_close'] <- 0
  }
  if (relevant_window$voiced == 1) {
    syllables[i,'voiced'] <- 1
  } else {
    syllables[i,'voiced'] <- 0
  }
  syllables[i,'subtask'] <- relevant_window$subtask
  syllables[i,'nsyll_in_trial'] <- relevant_window$nsyll
}

write.csv(df, paste0("../data/out_summary/segments.csv"), quote = FALSE, row.names = FALSE)
write.csv(syllables, paste0("../data/out_summary/syllables.csv"), quote = FALSE, row.names = FALSE)

