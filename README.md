# StatisticalEvaluation
This is a Python script that reads the PKL file from the MASK R-CNN,
then calculates and displays the statistics.
(see Mask R-CNN: https://github.com/maxfrei750/SinterAnalysis.)
I usually execute this script using Spyder.

The main function starts in line 669, where you need to define:
-thresholds, 
-binwidth and 
-put in the scale (nanometer per pixel of you image) 

You need to input you SEM-images in the folder "data_input".
The results will be shown in the folder "data_output".
