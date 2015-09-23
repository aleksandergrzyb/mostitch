//
//  Created by Aleksander Grzyb on 17/08/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#ifndef __Mosaic_Stitcher__AGImageBlender__
#define __Mosaic_Stitcher__AGImageBlender__

#include "AGOpenCVHelper.h"
#include "AGDataStructures.h"

#include <stdio.h>
#include <opencv2/opencv.hpp>

 /// Responsible for blending images into one plane.

class AGImageBlender {
public:
    
    /**
     *  Takes vector of transformed images and blends them into outputImage. Method used for blending is described 
     *  in master's thesis.
     *
     *  @param images      Vector of transformed images. Images have to be the same size on the same plane.
     *  @param outputImage The result of blending.
     *  @param error       Return error.
     */
    static void blendImages(std::vector<AGImage> &images, cv::Mat &outputImage, AGError &error);
    
private:
    
    /**
     *  Calculates distance transform of image. Distance transform is assigned to distanceTransform property of
     *  image object.
     *
     *  @param image Input image.
     *  @param error Return error.
     */
    static void calculateDistanceTransformOfImage(AGImage &image, AGError &error);
    
};

#endif /* defined(__Mosaic_Stitcher__AGImageBlender__) */
