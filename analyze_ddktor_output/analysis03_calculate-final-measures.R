df <- read.csv(paste0("../data/out_summary/segments.csv"))
syllables <- read.csv(paste0("../data/out_summary/syllables.csv"))
windows <- read.csv(paste0("../data/out_summary/ddkrates.csv"))

# Set up summary data frame
file <- rep(levels(factor(df$file)), 2)
subtask <- c(rep('amr', 0.5*length(file)), rep('smr', 0.5*length(file)))

avg_cons_duration <- rep(NA, length(file))
var_cons_duration <- rep(NA, length(file))
avg_vowel_duration <- rep(NA, length(file))
var_vowel_duration <- rep(NA, length(file))
avg_syll_duration <- rep(NA, length(file))
var_syll_duration <- rep(NA, length(file))
coeffvar_syll_duration <- rep(NA, length(file))
avg_intersyll_duration <- rep(NA, length(file))
var_intersyll_duration <- rep(NA, length(file))
coeffvar_intersyll_duration <- rep(NA, length(file))
avg_ddk_rate <- rep(NA, length(file))
var_ddk_rate <- rep(NA, length(file))
overall_ddk_rate <- rep(NA, length(file))

data_summary <- data.frame(file, subtask, avg_cons_duration, var_cons_duration, avg_vowel_duration, var_vowel_duration, avg_syll_duration, var_syll_duration, coeffvar_syll_duration, avg_intersyll_duration, var_intersyll_duration, coeffvar_intersyll_duration, avg_ddk_rate, var_ddk_rate, overall_ddk_rate)

# Calculate segmental info
for (i in 1:nrow(data_summary)) {
  part_subset <- subset(df, file == data_summary[i,'file'] & subtask == data_summary[i,'subtask'])
  part_subset <- subset(part_subset, extreme_value_withinparts == 0)
  # part_subset <- subset(part_subset, ddktor_count_close == 1)
  part_subset <- subset(part_subset, voiced == 0)
  part_subset <- subset(part_subset, vowel2long == 0)
  part_subset <- subset(part_subset, nsyll_in_trial > 9)
  part_vot_subset <- subset(part_subset, grepl('Vot', seg_type, fixed = TRUE))
  data_summary[i,'avg_cons_duration'] <- mean(part_vot_subset$seg_duration)
  data_summary[i,'var_cons_duration'] <- var(part_vot_subset$seg_duration)
  
  part_vowel_subset <- subset(part_subset, grepl('Vowel', seg_type, fixed = TRUE))
  data_summary[i,'avg_vowel_duration']<- mean(part_vowel_subset$seg_duration)
  data_summary[i,'var_vowel_duration']<- var(part_vowel_subset$seg_duration)
  
  if (nrow(part_subset) == 0) {
    data_summary[i,'avg_cons_duration'] <- NA
    data_summary[i,'var_cons_duration'] <- NA
    data_summary[i,'avg_vowel_duration']<- NA
    data_summary[i,'var_vowel_duration']<- NA
  }
}

# Calculate syllable info
for (i in 1:nrow(data_summary)) {
  part_subset <- subset(syllables, file == data_summary[i,'file'] & subtask == data_summary[i,'subtask'])
  # part_subset <- subset(part_subset, middle_syll == 1)
  # part_subset <- subset(part_subset, full_syll == 1)
  # part_subset <- subset(part_subset, ddktor_count_close == 1)
  part_subset <- subset(part_subset, extreme_value_withinparts == 0)
  part_subset <- subset(part_subset, voiced == 0)
  part_subset <- subset(part_subset, vowel2long == 0)
  part_subset <- subset(part_subset, nsyll_in_trial > 9)
  data_summary[i,'avg_syll_duration'] <- mean(part_subset$syll_duration)
  data_summary[i,'var_syll_duration'] <- var(part_subset$syll_duration)
  data_summary[i,'coeffvar_syll_duration'] <- sd(part_subset$syll_duration) / mean(part_subset$syll_duration)
  if (nrow(part_subset) == 0) {
    data_summary[i,'avg_syll_duration'] <- NA
    data_summary[i,'var_syll_duration'] <- NA
    data_summary[i,'coeffvar_syll_duration'] <- NA
  }
}

# Calculate average/variance intersyllable duration
for (i in 1:nrow(data_summary)) {
  part_subset <- subset(syllables, file == data_summary[i,'file'] & subtask == data_summary[i,'subtask'])
  # part_subset <- subset(part_subset, middle_syll == 1)
  # part_subset <- subset(part_subset, ddktor_count_close == 1)
  part_subset <- subset(part_subset, extreme_value_withinparts == 0)
  part_subset <- subset(part_subset, follows_extreme_value == 0)
  part_subset <- subset(part_subset, voiced == 0)
  part_subset <- subset(part_subset, vowel2long == 0)
  part_subset <- subset(part_subset, nsyll_in_trial > 9)
  data_summary[i,'avg_intersyll_duration'] <- mean(part_subset$intersyll_duration, na.rm = TRUE)
  data_summary[i,'var_intersyll_duration'] <- var(part_subset$intersyll_duration, na.rm = TRUE)
  data_summary[i,'coeffvar_intersyll_duration'] <- sd(part_subset$intersyll_duration, na.rm = TRUE) / data_summary[i,'avg_intersyll_duration']
  if (nrow(part_subset) == 0) {
    data_summary[i,'avg_intersyll_duration'] <- NA
    data_summary[i,'var_intersyll_duration'] <- NA
    data_summary[i,'coeffvar_intersyll_duration'] <- NA
  }
}


# Calculate DDK rate info
for (i in 1:nrow(data_summary)) {
  part_subset <- subset(windows, file == data_summary[i,'file'] & subtask == data_summary[i,'subtask'])
  # part_subset <- subset(part_subset, middle_syll == 1)
  # part_subset <- subset(part_subset, full_syll == 1)
  # part_subset <- subset(part_subset, ddktor_count_close == 1)
  part_subset <- subset(part_subset, voiced == 0)
  part_subset <- subset(part_subset, nsyll > 9)
  data_summary[i,'avg_ddk_rate'] <- mean(part_subset$ddk_rate)
  data_summary[i,'var_ddk_rate'] <- var(part_subset$ddk_rate)
  data_summary[i,'overall_ddk_rate'] <- sum(part_subset$nsyll, na.rm = TRUE) / sum(part_subset$speech_dur, na.rm = TRUE)
  if (nrow(part_subset) == 0) {
    data_summary[i,'avg_ddk_rate'] <- NA
    data_summary[i,'var_ddk_rate'] <- NA
    data_summary[i,'overall_ddk_rate'] <- NA
  }
}

names(data_summary)[names(data_summary) == 'avg_cons_duration'] <- 'avg_vot_duration'
names(data_summary)[names(data_summary) == 'var_cons_duration'] <- 'var_vot_duration'
write.csv(data_summary, paste0("../data/out_summary/bypart-measures.csv"), quote = FALSE, row.names = FALSE)


# Get info by trial
window_number <- c()
for (i in 1:length(levels(factor(windows$file)))) {
  sub <- subset(windows, file == levels(factor(windows$file))[i])
  window_number <- c(window_number, seq(1,nrow(sub)))
}
windows$win_number <- window_number

trial_summary <- data.frame(windows$file, windows$subtask, windows$win_number, windows$win_type, windows$win_start, windows$nsyll, windows$ddk_rate, windows$mean_vot_dur, windows$mean_vowel_dur)
names(trial_summary) <- c('file', 'subtask', 'window_number', 'window_type', 'window_start', 'nsyll', 'ddk_rate', 'mean_vot_duration', 'mean_vowel_duration')
write.csv(trial_summary, paste0("../data/out_summary/bytrial-measures.csv"), quote = FALSE, row.names = FALSE)

