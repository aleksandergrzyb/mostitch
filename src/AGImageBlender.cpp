//
//  Created by Aleksander Grzyb on 17/08/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#include "AGImageBlender.h"

using namespace cv;
using namespace std;

void AGImageBlender::calculateDistanceTransformOfImage(AGImage &image, AGError &error)
{
    if (!image.image.data) {
        error = { true, "calculateDistanceTransformOfImage: Image has no data." }; return;
    }
    Mat labels;
    AGOpenCVHelper::createEmptyMatWithSize(labels, image.image.size(), CV_32FC1);
    AGOpenCVHelper::createEmptyMatWithSize(image.distanceTransform, image.image.size(), CV_8U);
    distanceTransform(image.mask, image.distanceTransform, labels, CV_DIST_C, CV_DIST_MASK_5);
}

void AGImageBlender::blendImages(std::vector<AGImage> &images, cv::Mat &outputImage, AGError &error)
{
    AGOpenCVHelper::createEmptyMatWithSize(outputImage, images.front().image.size(), CV_8U);
    
    for (auto &image : images) {
        AGError checkError;
        AGImageBlender::calculateDistanceTransformOfImage(image, checkError);
        if (checkError.isError) {
            error = { true, "blendImages: Error while calculating distance transform. " + checkError.description }; return;
        }
    }
    
    for (int row = 0; row < outputImage.rows; ++row) {
        for (int col = 0; col < outputImage.cols; ++col) {
            Point currentPoint = Point(col, row);
            int numeratorSum = 0, denominatorSum = 0;
            for (auto &image : images) {
                AGError checkError;
                bool isPixelWhite = AGOpenCVHelper::isPixelWhiteInImage(image.mask, currentPoint, checkError);
                if (checkError.isError) {
                    error = { true, "blendImages: Error while checking white pixel. " + checkError.description }; return;
                }
                if (isPixelWhite) {
                    AGError checkErrorDT, checkErrorI;
                    int weight = AGOpenCVHelper::getPixelValueAtPointInImage(image.distanceTransform, currentPoint, checkErrorDT);
                    if (checkErrorDT.isError) {
                        error = { true, "blendImages: Error while getting pixel value of distance transform. " + checkError.description }; return;
                    }
                    numeratorSum += weight * AGOpenCVHelper::getPixelValueAtPointInImage(image.image, currentPoint, checkErrorI);
                    if (checkErrorI.isError) {
                        error = { true, "blendImages: Error while getting pixel value of image. " + checkErrorI.description }; return;
                    }
                    denominatorSum += weight;
                }
            }
            int pixelValue = 0;
            if (denominatorSum != 0) {
                pixelValue = numeratorSum / denominatorSum;
            }
            AGError checkError;
            AGOpenCVHelper::setPixelValueAtPointInImage(outputImage, currentPoint, pixelValue, checkError);
            if (checkError.isError) {
                error = { true, "blendImages: Error while setting pixel value. " + checkError.description }; return;
            }
        }
    }
}