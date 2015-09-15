//
//  Created by Aleksander Grzyb on 01/04/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#ifndef __Mosaic_Stitcher__AGImagePreprocessing__
#define __Mosaic_Stitcher__AGImagePreprocessing__

#include "../thirdparty/include/opencv2/opencv.hpp"
#include "AGDataStructures.h"
#include "AGOpenCVHelper.h"

#include <stdio.h>

enum PathDirection {
    PathDown,
    PathUp,
    PathLeft,
    PathRight
};

enum MarkingVisitedPixelsMode {
    Row,
    Column
};

class AGImagePreprocessing {
public:
    AGImagePreprocessing(const AGParameters &parameters);
    void detectPaths(cv::vector<cv::vector<AGImage>> &imagesMatrix);
    void prepareForPathDetecting(cv::Mat &input, cv::Mat &output);
private:
    void searchForPathsInImageUsingSkeleton(AGImage &image, cv::Mat &skeleton);
    void addPixelIfWhiteInImage(std::vector<cv::Point> &pixels, cv::Point pixel, cv::Mat &image, std::vector<std::vector<bool>> &visitedPixels);
    void pushBackAdjacentPixelsToPixelInImage(std::vector<cv::Point> &pixels, cv::Point pixel, cv::Mat &image,
                                              PathDirection pathDirection, std::vector<std::vector<bool>> &visitedPixels);
    void createSkeleton(cv::Mat &input, cv::Mat &output);
    void performMorphology(cv::Mat &input, cv::Mat &output);
    void findWhitePixelsInSkeleton(cv::Mat &skeleton, std::vector<cv::Point> &whitePixels);
    void categorizeWhitePixels(std::vector<cv::Point> &whitePixels, std::vector<cv::Point> &topWhitePixels,
                               std::vector<cv::Point> &bottomWhitePixels, std::vector<cv::Point> &leftWhitePixels,
                               std::vector<cv::Point> &rightWhitePixels,
                               int imageWidth, int imageHeight);
    void removeNeighbourWhitePixels(std::vector<cv::Point> &whitePixels, PathDirection pathDirection);
    void performDFSForPathDirection(AGImage &image, cv::Mat &skeleton, std::vector<cv::Point> &whitePixels,
                                                 PathDirection pathDirection);
    void markVisitedPixels(std::vector<std::vector<bool>> &visitedPixels, MarkingVisitedPixelsMode markingVisitedPixelsMode, int rowOrColumnCor);
    void resetVisitedPixels(std::vector<std::vector<bool>> &visitedPixels, int width, int height);

    // cleaning
    void removePathPointsDuplicates(AGImage &image);

    // testing
    void testDetectedPathsInImage(AGImage &image);
    void testSelectedImage(AGImage &image);
    void testSaveCurrentDFS(std::vector<cv::Point> &pixels, cv::vector<cv::vector<bool>> &visitedPixels, cv::Mat &image);

    bool testingMode;
    AGParameters parameters;
};

#endif /* defined(__Mosaic_Stitcher__AGImagePreprocessing__) */
