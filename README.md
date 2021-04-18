# Background blur using DeepLab Model

Model gives segmented map and from this model we get foreground using bitwiseAND operation
Edge detection is also improve using Erosion Morphological operation 
### Input image
![](images/1.jpgn)


Foreground without erosion 
 








Forground Image using erosion

 
Background is extracted using AND operation with blur original image and inverse map 

 
Lastly adding both layers gives blur background image                            I
  
