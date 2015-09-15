//
//  Created by Aleksander Grzyb on 17/08/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#ifndef __Mosaic_Stitcher__AGImageBlender__
#define __Mosaic_Stitcher__AGImageBlender__

#include "../thirdparty/include/opencv2/opencv.hpp"
#include "AGOpenCVHelper.h"
#include "AGDataStructures.h"

#include <stdio.h>

class AGImageBlender {
public:
    static void blendImages(std::vector<AGImage> &images, cv::Mat &outputImage, AGError &error);
    static void calculateDistanceTransformOfImage(AGImage &image, AGError &error);
};

#endif /* defined(__Mosaic_Stitcher__AGImageBlender__) */
