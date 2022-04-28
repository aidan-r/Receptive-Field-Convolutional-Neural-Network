The files in the directory "2D_noise_natural/Stimulus_files" are visual stimulus files.  When you load the file into matlab (using the standard 'load' command), the 'mov' variable contains the stimulus as a 3-dimensional matrix in y, x, t.

'Equalpower*.mat' are stimuli with natural images.  Each frame is a small patch of natural image, but temporally they are uncorrelated.

'Randomphase*.mat' are stimuli with the same power spectra as the natural images, but randomized spatial phase.  

'WhitenedNatural*.mat' are stimuli with the same spatial phase as the natural images, but flat power spectrum.  

To know which stimulus file to load, you need to look at the 'log' file associated with each spike file, in which the stimulus file is described.

