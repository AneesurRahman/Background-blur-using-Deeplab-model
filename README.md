# Background blur using DeepLab Model

Model gives segmented map and from this model we get foreground using bitwiseAND operation
Edge detection is also improve using Erosion Morphological operation 
### Input image
![](images/1.JPG)


### Foreground without erosion 
forground without erosion 
![](images/fgwoe.JPG)

### Forground Image using erosion
![](images/fgwe.JPG)

### Background
Background is extracted using AND operation with blur original image and inverse map 
![](images/bg.JPG)
 
Lastly adding both layers gives blur background image                            I
  
