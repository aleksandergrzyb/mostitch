//
//  Created by Aleksander Grzyb on 14/01/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#ifndef __Mosaic_Stitcher__AGOpenCVHelper__
#define __Mosaic_Stitcher__AGOpenCVHelper__

#include "../thirdparty/include/opencv2/core/core.hpp"
#include "AGDataStructures.h"

#include <stdio.h>

#define A_COEFF 0
#define B_COEFF 1

class AGOpenCVHelper {
public:
    // Handling image types
    static void convertImageToColor(cv::Mat &image);
    static void convertImageToGrayscale(cv::Mat &image);
    static std::string getImageTypeNameUsingImageType(const int imageType, AGError &error);
    
    // Geometric methods
    static void calculateLinearFunctionCoeffsUsingPoints(const cv::Point &pointOne, const cv::Point &pointTwo, std::vector<double> &coeffs);
    static double getDistanceBetweenPoints(const cv::Point &pointOne, const cv::Point &pointTwo);
    
    // Image creation
    static void createEmptyMatWithSize(cv::Mat &image, const cv::Size &size, const int imageType);
    static void createMaskMatWithSize(cv::Mat &image, const cv::Size &size);
    
    // Image data introspection
    static bool isPixelWhiteInImage(cv::Mat &image, const cv::Point &pixelPosition, AGError &error);
    static int getPixelValueAtPointInImage(const cv::Mat &image, const cv::Point &pixelPosition, AGError &error);
    static void setPixelValueAtPointInImage(const cv::Mat &image, const cv::Point &pixelPosition, const int pixelValue, AGError &error);
    
    // Image transformations
    static void createShiftMatrix(cv::Mat &shiftMatrix, const double dx, const double dy);
    static void rotateImage(cv::Mat &image, const double angle);
    static void insertSourceImageIntoOutputImageAtPoint(const cv::Mat &sourceImage, cv::Mat &outputImage, const cv::Point &point, AGError &error);
    static void linkTwoImagesTogether(const cv::Mat &imageOne, const cv::Mat &imageTwo, cv::Mat &outputImage, const ImageDirection &imageDirection, AGError &error);
    static void linkTwoImagesTogetherAndDrawMatches(const AGImage &imageOne, const AGImage &imageTwo, const std::vector<cv::DMatch> &matches, cv::Mat &outputImage, const ImageDirection &imageDirection, AGError &error);

    // Saving and showing images
    static void saveImage(cv::Mat &image, const std::string &name, const std::string &savePath, AGError &error);
    static void saveKeypointsToFile(const AGImage &image, const std::string &fileName, const std::string &savePath, AGError &error);
    static void saveMatchesToFile(const std::vector<cv::DMatch> &matches, const std::string &fileName, const std::string &savePath, AGError &error);
    static void showImage(const cv::Mat &image, const std::string &windowName, AGError &error);
    static std::string getDescriptionOfImage(const AGImage &image, AGError &error);
    static void loadImage(cv::Mat &image, const std::string &loadPath, AGError &error);
};

#endif /* defined(__Mosaic_Stitcher__AGOpenCVHelper__) */
