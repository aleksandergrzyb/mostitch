//
//  Created by Aleksander Grzyb on 13/01/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#ifndef __Mosaic_Stitcher__AGMosaicStitcher__
#define __Mosaic_Stitcher__AGMosaicStitcher__

#include "../thirdparty/include/opencv2/stitching/stitcher.hpp"
#include "../thirdparty/include/opencv2/opencv.hpp"
#include "AGDataStructures.h"
#include "AGImagePreprocessing.h"
#include "AGOpenCVHelper.h"

#include <stdio.h>
#include <vector>

class AGMosaicStitcher {
public:
    AGMosaicStitcher(const AGParameters &parameters);
    int stitchMosaic(std::vector<std::vector<AGImage>> &imagesMatrix, cv::Mat &outputImage);
private:
    // Stitching
    void performStitching(std::vector<std::vector<AGImage>> &imagesMatrix, cv::Mat &outputImage);

    // Filtering matches
    void deleteMatchesFromMultipleKeypointsToMultiple(std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &filtredMatches);
    void filterMatchesUsingRANSAC(AGImage &imageOne, AGImage &imageTwo, std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &filtredMatches);
    void filterMatchesBasedOnSlopeAndLength(AGImage &imageOne, AGImage &imageTwo, std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &filtredMatches, ImageDirection imageDirection);
    void filterMatchesBasedOnPlacement(AGImage &imageOne,AGImage &imageTwo, std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &filtredMatches, ImageDirection imageDirection);
    void filterMatches(AGImage &imageOne, AGImage &imageTwo, std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &filtredMatches, ImageDirection imageDirection);
    void applyStitchingAlgorithm(AGImage &imageOne, AGImage &imageTwo, ImageDirection imageDirection, cv::Mat &transform);

    // Finding image features
    void findFeatures(AGImage &imageOne, AGImage &imageTwo, std::vector<cv::detail::ImageFeatures> &imagesFeatures, ImageDirection imageDirection);
    void findFeaturesWithSIFT(AGImage &inputImage, cv::detail::ImageFeatures &imageFeatures, cv::Rect &roi);
    void findMatchesWithBruteForce(std::vector<cv::detail::ImageFeatures> &imagesFeatures, cv::detail::MatchesInfo &matchesInfo);

    // Finding transform between images
    void findTransformBetweenImages(AGImage &imageOne, AGImage &imageTwo, std::vector<cv::DMatch> &matches, cv::Mat &transform, ImageDirection imageDirection);
    void findShiftTransform(AGImage &imageOne, AGImage &imageTwo, cv::Mat &transform, ImageDirection imageDirection);
    bool findTransformBasedOnPaths(AGImage &imageOne, AGImage &imageTwo, cv::Mat &transform, ImageDirection imageDirection);
    bool selectPointFromPath(AGImage &image, ImageDirection desiredPlace, cv::Point &point);

    // Helper methods
    void clusterArrayWithinRange(std::vector<double> &array, std::vector<double> &output, double rangeParameter);

    // Init transforms matrix
    void initTransformsMatrix(int xSize, int ySize);

    // Init masks in images matrix
    void initMaskInImagesMatrix(std::vector<std::vector<AGImage>> &imagesMatrix);

    // Testing
    void testStitchBetweenTwoImages(AGImage &imageOne, AGImage &imageTwo, ImageDirection imageDirection);
    void testImagePreprocessing(std::vector<std::vector<AGImage>> &imagesMatrix);

    // Private variables
    AGImagePreprocessing *imagePreprocessing;
    AGParameters parameters;
    int yShift;
    int xShift;
    bool testingMode;
    std::vector<std::vector<cv::Mat>> transformsMatrix;
};

#endif /* defined(__Mosaic_Stitcher__AGMosaicStitcher__) */
