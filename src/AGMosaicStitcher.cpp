//
//  Created by Aleksander Grzyb on 13/01/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#include "../thirdparty/include/opencv2/features2d/features2d.hpp"
#include "../thirdparty/include/opencv2/nonfree/features2d.hpp"
#include "../thirdparty/include/opencv2/video/tracking.hpp"
#include "../thirdparty/include/opencv2/calib3d/calib3d.hpp"
#include "AGMosaicStitcher.h"
#include "AGImageBlender.h"

#include <cmath>
#include <map>

using namespace cv;
using namespace std;
using namespace detail;

#pragma mark -
#pragma mark Initialization

AGMosaicStitcher::AGMosaicStitcher(const AGParameters &parameters)
{
    this->parameters = parameters;
    this->imagePreprocessing = new AGImagePreprocessing(parameters);
}

void AGMosaicStitcher::initTransformsMatrix(int xSize, int ySize)
{
    for (int x = 0; x < xSize; x++) {
        vector<Mat> nextColumn;
        this->transformsMatrix.push_back(nextColumn);
        for (int y = 0; y < ySize; y++) {
            Mat emptyTransform;
            this->transformsMatrix[x].push_back(emptyTransform);
        }
    }
}

void AGMosaicStitcher::initMaskInImagesMatrix(std::vector<std::vector<AGImage>> &imagesMatrix)
{
    for (int x = 0; x < imagesMatrix.size(); x++) {
        for (int y = 0; y < imagesMatrix.front().size(); y++) {
            imagesMatrix[x][y].mask = Mat::ones(imagesMatrix[x][y].image.rows, imagesMatrix[x][y].image.cols, CV_8UC1) * 255;
        }
    }
}

#pragma mark -
#pragma mark Starting Point

int AGMosaicStitcher::stitchMosaic(vector<vector<AGImage>> &imagesMatrix, Mat &outputImage)
{
    if (imagesMatrix.empty()) {
        return EXIT_FAILURE;
    }
    for (int x = 0; x < imagesMatrix.size(); x++) {
        for (int y = 0; y < imagesMatrix[x].size(); y++) {
            if (!imagesMatrix[x][y].image.data) {
                return EXIT_FAILURE;
            }
        }
    }
    this->testingMode = false;
    this->initTransformsMatrix((int)imagesMatrix.size(), (int)imagesMatrix.front().size());
    this->initMaskInImagesMatrix(imagesMatrix);
    this->imagePreprocessing->detectPaths(imagesMatrix);

//    this->testImagePreprocessing(imagesMatrix);

    this->performStitching(imagesMatrix, outputImage);
//    this->testStitchBetweenTwoImages(imagesMatrix[1][8], imagesMatrix[1][7], Up);
//    this->testStitchBetweenTwoImages(imagesMatrix[0][1], imagesMatrix[0][0], Up);
    return EXIT_SUCCESS;
}

void AGMosaicStitcher::performStitching(vector<vector<AGImage>> &imagesMatrix, Mat &outputImage)
{
    int imageWidth = imagesMatrix.front().front().width;
    int imageHeight = imagesMatrix.front().front().height;

    int midXCoor = floor((imagesMatrix.size() - 1) * 0.5);
    int midYCoor = floor((imagesMatrix.front().size() - 1) * 0.5);

    int offset = 0;
    int outputImageWidth = imageWidth * (int)imagesMatrix.size() + 2 * offset;
    int outputImageHeigth = imageHeight * (int)imagesMatrix.front().size() + 2 * offset;

    outputImage = Mat::zeros(outputImageHeigth, outputImageWidth, CV_8UC4);

    this->xShift = midXCoor * imageWidth + offset;
    this->yShift = midYCoor * imageHeight + offset;

    ImageDirection imageDirection;
    if (imagesMatrix.front().size() > imagesMatrix.size()) {
        for (int x = 0; x < midXCoor; x++) {
            imageDirection = Right;
            this->applyStitchingAlgorithm(imagesMatrix[x][midYCoor], imagesMatrix[x + 1][midYCoor], imageDirection, this->transformsMatrix[x][midYCoor]);
        }

        for (int x = midXCoor + 1; x < imagesMatrix.size(); x++) {
            imageDirection = Left;
            this->applyStitchingAlgorithm(imagesMatrix[x][midYCoor], imagesMatrix[x - 1][midYCoor], imageDirection, this->transformsMatrix[x][midYCoor]);
        }

        for (int x = 0; x < imagesMatrix.size(); x++) {
            for (int y = midYCoor - 1; y >= 0; y--) {
                imageDirection = Down;
                this->applyStitchingAlgorithm(imagesMatrix[x][y], imagesMatrix[x][y + 1], imageDirection, this->transformsMatrix[x][y]);
            }
            for (int y = midYCoor + 1; y < imagesMatrix.front().size(); y++) {
                imageDirection = Up;
                this->applyStitchingAlgorithm(imagesMatrix[x][y], imagesMatrix[x][y - 1], imageDirection, this->transformsMatrix[x][y]);
            }
        }
    } else {
        for (int y = 0; y < midYCoor; y++) {
            imageDirection = Down;
            this->applyStitchingAlgorithm(imagesMatrix[midXCoor][y], imagesMatrix[midXCoor][y + 1], imageDirection, this->transformsMatrix[midXCoor][y]);
        }

        for (int y = midYCoor + 1; y < imagesMatrix.front().size(); y++) {
            imageDirection = Up;
            this->applyStitchingAlgorithm(imagesMatrix[midXCoor][y], imagesMatrix[midXCoor][y - 1], imageDirection, this->transformsMatrix[midXCoor][y]);
        }

        for (int y = 0; y < imagesMatrix.front().size(); y++) {
            for (int x = midXCoor - 1; x >= 0; x--) {
                imageDirection = Right;
                this->applyStitchingAlgorithm(imagesMatrix[x][y], imagesMatrix[x + 1][y], imageDirection, this->transformsMatrix[x][y]);
            }
            for (int x = midXCoor + 1; x < imagesMatrix.size(); x++) {
                imageDirection = Left;
                this->applyStitchingAlgorithm(imagesMatrix[x][y], imagesMatrix[x - 1][y], imageDirection, this->transformsMatrix[x][y]);
            }
        }
    }

    // All images need to be shifted to the place where reference image is located
    Mat baseShiftTransform;
    AGOpenCVHelper::createShiftMatrix(baseShiftTransform, this->xShift, this->yShift);
    for (int x = 0; x < imagesMatrix.size(); x++) {
        for (int y = 0; y < imagesMatrix.front().size(); y++) {
            cvtColor(imagesMatrix[x][y].image, imagesMatrix[x][y].image, CV_GRAY2BGRA);
            warpAffine(imagesMatrix[x][y].image, imagesMatrix[x][y].image, baseShiftTransform, outputImage.size());
            warpAffine(imagesMatrix[x][y].mask, imagesMatrix[x][y].mask, baseShiftTransform, outputImage.size());
        }
    }

    // Displaying rest of the images

    for (int x = 0; x < imagesMatrix.size(); x++) {
        for (int y = 0; y < imagesMatrix.front().size(); y++) {
            if (x != midXCoor || y != midYCoor) {
                Mat transformedImage = imagesMatrix[x][y].image;
                Mat mask = imagesMatrix[x][y].mask;
                if (imagesMatrix.front().size() > imagesMatrix.size()) {
                    if (y > midYCoor) {
                        for (int yTrans = y; yTrans > midYCoor; yTrans--) {
                            warpAffine(transformedImage, transformedImage, transformsMatrix[x][yTrans], outputImage.size());
                            warpAffine(mask, mask, transformsMatrix[x][yTrans], outputImage.size());
                        }
                    } else if (y < midYCoor) {
                        for (int yTrans = y; yTrans < midYCoor; yTrans++) {
                            warpAffine(transformedImage, transformedImage, transformsMatrix[x][yTrans], outputImage.size());
                            warpAffine(mask, mask, transformsMatrix[x][yTrans], outputImage.size());
                        }
                    }

                    if (x > midXCoor) {
                        for (int xTrans = x; xTrans > midXCoor; xTrans--) {
                            warpAffine(transformedImage, transformedImage, transformsMatrix[xTrans][midYCoor], outputImage.size());
                            warpAffine(mask, mask, transformsMatrix[xTrans][midYCoor], outputImage.size());
                        }
                    } else if (x < midXCoor) {
                        for (int xTrans = x; xTrans < midXCoor; xTrans++) {
                            warpAffine(transformedImage, transformedImage, transformsMatrix[xTrans][midYCoor], outputImage.size());
                            warpAffine(mask, mask, transformsMatrix[xTrans][midYCoor], outputImage.size());
                        }
                    }
                } else {
                    if (x > midXCoor) {
                        for (int xTrans = x; xTrans > midXCoor; xTrans--) {
                            warpAffine(transformedImage, transformedImage, transformsMatrix[xTrans][y], outputImage.size());
                            warpAffine(mask, mask, transformsMatrix[xTrans][y], outputImage.size());
                        }
                    } else if (x < midXCoor) {
                        for (int xTrans = x; xTrans < midXCoor; xTrans++) {
                            warpAffine(transformedImage, transformedImage, transformsMatrix[xTrans][y], outputImage.size());
                            warpAffine(mask, mask, transformsMatrix[xTrans][y], outputImage.size());
                        }
                    }

                    if (y > midYCoor) {
                        for (int yTrans = y; yTrans > midYCoor; yTrans--) {
                            warpAffine(transformedImage, transformedImage, transformsMatrix[midXCoor][yTrans], outputImage.size());
                            warpAffine(mask, mask, transformsMatrix[midXCoor][yTrans], outputImage.size());
                        }
                    } else if (y < midYCoor) {
                        for (int yTrans = y; yTrans < midYCoor; yTrans++) {
                            warpAffine(transformedImage, transformedImage, transformsMatrix[midXCoor][yTrans], outputImage.size());
                            warpAffine(mask, mask, transformsMatrix[midXCoor][yTrans], outputImage.size());
                        }
                    }
                }
//                transformedImage.copyTo(outputImage, mask);
//                addWeighted(transformedImage, 1.0, outputImage, 1.0, 0.0, outputImage);
            }
        }
    }
    //  Displaying reference (unchanged) image on the final grid
//    imagesMatrix[midXCoor][midYCoor].image.copyTo(outputImage, imagesMatrix[midXCoor][midYCoor].mask);
//    addWeighted(imagesMatrix[midXCoor][midYCoor].image, 1.0, outputImage, 1.0, 0.0, outputImage);
    
    vector<AGImage> imagesToBlend;
    for (auto &imageRow : imagesMatrix) {
        for (auto &image : imageRow) {
            imagesToBlend.push_back(image);
        }
    }
    
    AGError error;
    AGImageBlender::blendImages(imagesToBlend, outputImage, error);
    if (error.isError) {
        cout << error.description << endl;
    }
}

void AGMosaicStitcher::applyStitchingAlgorithm(AGImage &imageOne, AGImage &imageTwo, ImageDirection imageDirection, Mat &transform)
{
    vector<DMatch> filtredMatches;
    if (this->parameters.isAdHoc) {
        Rect firstHalfImageOneROI;
        Rect secondHalfImageOneROI;

        Rect firstHalfImageTwoROI;
        Rect secondHalfImageTwoROI;

        if (imageDirection == Up || imageDirection == Down) {
            firstHalfImageOneROI = Rect(0, 0, imageOne.width * 0.5, imageOne.height);
            secondHalfImageOneROI = Rect(imageOne.width * 0.5, 0, imageOne.width * 0.5, imageOne.height);

            firstHalfImageTwoROI = Rect(0, 0, imageTwo.width * 0.5, imageTwo.height);
            secondHalfImageTwoROI = Rect(imageTwo.width * 0.5, 0, imageTwo.width * 0.5, imageTwo.height);
        }
        else {
            firstHalfImageOneROI = Rect(0, 0, imageOne.width, imageOne.height * 0.5);
            secondHalfImageOneROI = Rect(0, imageOne.height * 0.5, imageOne.width, imageOne.height * 0.5);

            firstHalfImageTwoROI = Rect(0, 0, imageTwo.width, imageTwo.height * 0.5);
            secondHalfImageTwoROI = Rect(0, imageTwo.height * 0.5, imageTwo.width, imageTwo.height * 0.5);
        }

        Mat firstHalfImageOneMat;
        Mat secondHalfImageOneMat;

        Mat firstHalfImageTwoMat;
        Mat secondHalfImageTwoMat;

        imageOne.image(firstHalfImageOneROI).copyTo(firstHalfImageOneMat);
        imageOne.image(secondHalfImageOneROI).copyTo(secondHalfImageOneMat);

        imageTwo.image(firstHalfImageTwoROI).copyTo(firstHalfImageTwoMat);
        imageTwo.image(secondHalfImageTwoROI).copyTo(secondHalfImageTwoMat);

        string firstHalfImageOneName = imageOne.name;
        firstHalfImageOneName += "_first_half";
        AGImage firstHalfImageOne(firstHalfImageOneMat, imageOne.xCoordinate, imageOne.yCoordinate, firstHalfImageOneROI.width, firstHalfImageOneROI.height, firstHalfImageOneName);

        string secondHalfImageOneName = imageOne.name;
        secondHalfImageOneName += "_second_half";
        AGImage secondHalfImageOne(secondHalfImageOneMat, imageOne.xCoordinate, imageOne.yCoordinate, secondHalfImageOneROI.width, secondHalfImageOneROI.height, secondHalfImageOneName);

        string firstHalfImageTwoName = imageTwo.name;
        firstHalfImageTwoName += "_first_half";
        AGImage firstHalfImageTwo(firstHalfImageTwoMat, imageTwo.xCoordinate, imageTwo.yCoordinate, firstHalfImageTwoROI.width, firstHalfImageTwoROI.height, firstHalfImageTwoName);

        string secondHalfImageTwoName = imageTwo.name;
        secondHalfImageTwoName += "_second_half";
        AGImage secondHalfImageTwo(secondHalfImageTwoMat, imageTwo.xCoordinate, imageTwo.yCoordinate, secondHalfImageTwoROI.width, secondHalfImageTwoROI.height, secondHalfImageTwoName);

        vector<ImageFeatures> firstHalfImagesFeatures;
        vector<ImageFeatures> secondHalfImagesFeatures;

        this->findFeatures(firstHalfImageOne, firstHalfImageTwo, firstHalfImagesFeatures, imageDirection);

        firstHalfImageOne.keypoints = firstHalfImagesFeatures[0].keypoints;
        firstHalfImageTwo.keypoints = firstHalfImagesFeatures[1].keypoints;

        MatchesInfo firstHalfMatchesInfo;
        this->findMatchesWithBruteForce(firstHalfImagesFeatures, firstHalfMatchesInfo);

        this->findFeatures(secondHalfImageOne, secondHalfImageTwo, secondHalfImagesFeatures, imageDirection);

        secondHalfImageOne.keypoints = secondHalfImagesFeatures[0].keypoints;
        secondHalfImageTwo.keypoints = secondHalfImagesFeatures[1].keypoints;

        MatchesInfo secondHalfMatchesInfo;
        this->findMatchesWithBruteForce(secondHalfImagesFeatures, secondHalfMatchesInfo);

        vector<DMatch> firstHalfFiltredMatches;
        this->filterMatches(firstHalfImageOne, firstHalfImageTwo, firstHalfMatchesInfo.matches, firstHalfFiltredMatches, imageDirection);

        vector<DMatch> secondHalfFiltredMatches;
        this->filterMatches(secondHalfImageOne, secondHalfImageTwo, secondHalfMatchesInfo.matches, secondHalfFiltredMatches, imageDirection);

        for (int i = 0; i < firstHalfFiltredMatches.size(); i++) {
            filtredMatches.push_back(firstHalfFiltredMatches[i]);
        }

        for (int i = 0; i < secondHalfFiltredMatches.size(); i++) {
            DMatch newMatch = secondHalfFiltredMatches[i];
            newMatch.queryIdx += firstHalfImageOne.keypoints.size();
            newMatch.trainIdx += firstHalfImageTwo.keypoints.size();
            filtredMatches.push_back(newMatch);
        }

        for (int i = 0; i < firstHalfImageOne.keypoints.size(); i++) {
            imageOne.keypoints.push_back(firstHalfImageOne.keypoints[i]);
        }

        for (int i = 0; i < secondHalfImageOne.keypoints.size(); i++) {
            KeyPoint correctedKeyPoint = secondHalfImageOne.keypoints[i];
            if (imageDirection == Up || imageDirection == Down) {
                correctedKeyPoint.pt.x += firstHalfImageOne.width;
            }
            else {
                correctedKeyPoint.pt.y += firstHalfImageOne.height;
            }
            imageOne.keypoints.push_back(correctedKeyPoint);
        }

        for (int i = 0; i < firstHalfImageTwo.keypoints.size(); i++) {
            imageTwo.keypoints.push_back(firstHalfImageTwo.keypoints[i]);
        }

        for (int i = 0; i < secondHalfImageTwo.keypoints.size(); i++) {
            KeyPoint correctedKeyPoint = secondHalfImageTwo.keypoints[i];
            if (imageDirection == Up || imageDirection == Down) {
                correctedKeyPoint.pt.x += firstHalfImageOne.width;
            }
            else {
                correctedKeyPoint.pt.y += firstHalfImageOne.height;
            }
            imageTwo.keypoints.push_back(correctedKeyPoint);
        }

//        Mat mat;
//        AGOpenCVHelper::linkTwoImagesTogetherAndDrawMatches(imageOne, imageTwo, filtredMatches, mat, imageDirection);
//        AGOpenCVHelper::saveImage(mat, "image");


        vector<KeyPoint> shiftedKeyPoints;
        for (int i = 0; i <= imageOne.keypoints.size(); i++) {
            KeyPoint shiftedKeyPoint = imageOne.keypoints[i];
            shiftedKeyPoint.pt.x += this->xShift;
            shiftedKeyPoint.pt.y += this->yShift;
            shiftedKeyPoints.push_back(shiftedKeyPoint);
        }
        imageOne.keypoints = shiftedKeyPoints;

        for (int i = 0; i <= imageTwo.keypoints.size(); i++) {
            KeyPoint shiftedKeyPoint = imageTwo.keypoints[i];
            shiftedKeyPoint.pt.x += this->xShift;
            shiftedKeyPoint.pt.y += this->yShift;
            imageTwo.keypoints[i] = shiftedKeyPoint;
        }

    }
    else {
        // Finding features
        vector<ImageFeatures> imagesFeatures;

        this->findFeatures(imageOne, imageTwo, imagesFeatures, imageDirection);
        imageOne.keypoints = imagesFeatures[0].keypoints;
        imageTwo.keypoints = imagesFeatures[1].keypoints;

        // Matching features
        MatchesInfo matchesInfo;
        this->findMatchesWithBruteForce(imagesFeatures, matchesInfo);

        // Filtering matches (eliminating outliners)
        this->filterMatches(imageOne, imageTwo, matchesInfo.matches, filtredMatches, imageDirection);

//        Mat mat;
//        AGOpenCVHelper::linkTwoImagesTogetherAndDrawMatches(imageOne, imageTwo, filtredMatches, mat, imageDirection);
//        AGOpenCVHelper::saveImage(mat, "image");
    }

    // Finding transform between images
    this->findTransformBetweenImages(imageOne, imageTwo, filtredMatches, transform, imageDirection);
}

#pragma mark -
#pragma mark Matching Features

void AGMosaicStitcher::findMatchesWithBruteForce(vector<ImageFeatures> &imagesFeatures, MatchesInfo &matchesInfo)
{
    /*
    * Brute Force Matcher is called twice, because he gives different results that depends on which image
    * is passed as train, and which as query. Probably this is because he iterates through all keypoints from query
    * image and makes 1-many relationship to keypoints from train image (we want many-many relationships). As a
    * solution we combine the result from both calls to bfMatcher and later in algorithm we are removing duplicates.
    */

    BFMatcher bfMatcherOne;
    MatchesInfo matchesInfoOne;
    bfMatcherOne.match(imagesFeatures[0].descriptors, imagesFeatures[1].descriptors, matchesInfoOne.matches);

    BFMatcher bfMatcherTwo;
    MatchesInfo matchesInfoTwo;
    bfMatcherTwo.match(imagesFeatures[1].descriptors, imagesFeatures[0].descriptors, matchesInfoTwo.matches);

    for (int i = 0; i < matchesInfoOne.matches.size(); i++) {
        DMatch match;
        match.queryIdx = matchesInfoOne.matches[i].queryIdx;
        match.trainIdx = matchesInfoOne.matches[i].trainIdx;
        match.distance = matchesInfoOne.matches[i].distance;
        matchesInfo.matches.push_back(match);
    }

    for (int i = 0; i < matchesInfoTwo.matches.size(); i++) {
        DMatch match;
        match.queryIdx = matchesInfoTwo.matches[i].trainIdx;
        match.trainIdx = matchesInfoTwo.matches[i].queryIdx;
        match.distance = matchesInfoTwo.matches[i].distance;
        matchesInfo.matches.push_back(match);
    }
}

#pragma mark -
#pragma mark Matches Filtering

void AGMosaicStitcher::filterMatches(AGImage &imageOne, AGImage &imageTwo, std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &filtredMatches, ImageDirection imageDirection)
{
    if (this->parameters.angleParameter < 0 || this->parameters.angleParameter > 90 || matches.empty() || !imageOne.image.data || !imageTwo.image.data) {
        return;
    }
    
    AGError error;
    Mat outputImage;
    if (this->testingMode) {
        AGOpenCVHelper::linkTwoImagesTogetherAndDrawMatches(imageOne, imageTwo, matches, outputImage, imageDirection, error);
        AGOpenCVHelper::saveImage(outputImage, "no_filtering", this->parameters.mosaicsSaveAbsolutePath, error);
    }
    
    vector<DMatch> matchesWithinRange;
    this->filterMatchesBasedOnPlacement(imageOne, imageTwo, matches, matchesWithinRange, imageDirection);

    if (this->testingMode) {
        AGOpenCVHelper::linkTwoImagesTogetherAndDrawMatches(imageOne, imageTwo, matchesWithinRange, outputImage, imageDirection, error);
        AGOpenCVHelper::saveImage(outputImage, "after_on_placement", this->parameters.mosaicsSaveAbsolutePath, error);
    }

    vector<DMatch> matchesWithNoRepetitions;
    this->deleteMatchesFromMultipleKeypointsToMultiple(matchesWithinRange, matchesWithNoRepetitions);

    if (this->testingMode) {
        AGOpenCVHelper::linkTwoImagesTogetherAndDrawMatches(imageOne, imageTwo, matchesWithNoRepetitions, outputImage, imageDirection, error);
        AGOpenCVHelper::saveImage(outputImage, "after_repetitions", this->parameters.mosaicsSaveAbsolutePath, error);
    }

    vector<DMatch> matchesAfterRANSAC;
    this->filterMatchesUsingRANSAC(imageOne, imageTwo, matchesWithNoRepetitions, matchesAfterRANSAC);

    if (this->testingMode) {
        AGOpenCVHelper::linkTwoImagesTogetherAndDrawMatches(imageOne, imageTwo, matchesAfterRANSAC, outputImage, imageDirection, error);
        AGOpenCVHelper::saveImage(outputImage, "after_ransac", this->parameters.mosaicsSaveAbsolutePath, error);
    }

    this->filterMatchesBasedOnSlopeAndLength(imageOne, imageTwo, matchesAfterRANSAC, filtredMatches, imageDirection);
    
    if (this->testingMode) {
        AGOpenCVHelper::linkTwoImagesTogetherAndDrawMatches(imageOne, imageTwo, filtredMatches, outputImage, imageDirection, error);
        AGOpenCVHelper::saveImage(outputImage, "after_slope_and_length", this->parameters.mosaicsSaveAbsolutePath, error);
    }
}

// Calculates angle between matches and x axis, then calculates the median angle of all matches, and deletes that ones that have angle greater than angleParameter
void AGMosaicStitcher::filterMatchesBasedOnSlopeAndLength(AGImage &imageOne, AGImage &imageTwo, vector<DMatch> &matches, vector<DMatch> &filtredMatches, ImageDirection imageDirection)
{
    if (matches.empty() || imageOne.keypoints.empty() || imageTwo.keypoints.empty()) {
        return;
    }
    vector<double> angles;
//    vector<double> lengths;
    vector<pair<int, double>> matchIndexAnglePair;
    for (int i = 0; i < matches.size(); i++) {
        Point pointOne = imageOne.keypoints[matches[i].queryIdx].pt;
        Point pointTwo = imageTwo.keypoints[matches[i].trainIdx].pt;
        if (imageDirection == Up || imageDirection == Down) {
            pointTwo.y += imageOne.height;
        }
        else {
            pointTwo.x += imageOne.width;
        }
        pointTwo.y *= -1;
        pointOne.y *= -1;

        vector<double> coeff;
        AGOpenCVHelper::calculateLinearFunctionCoeffsUsingPoints(pointOne, pointTwo, coeff);
//        lengths.push_back(AGOpenCVHelper::getDistanceBetweenPoints(pointOne, pointTwo));
        double angle = atan(coeff[A_COEFF]) * 180 / M_PI;
        if (angle < 0) {
            angle += 180;
        }
        else if (angle == 0) {
            angle = 90;
        }
        angles.push_back(angle);
    }

    if (this->parameters.clustering) {
        vector<double> lengthsCluster;
        vector<double> anglesCluster;
//        this->clusterArrayWithinRange(lengths, lengthsCluster, this->parameters.lengthParameter);
        this->clusterArrayWithinRange(angles, anglesCluster, this->parameters.angleParameter);

        //        cout << "Angle Cluster" << endl;
        //        for (int i = 0; i < anglesCluster.size(); i++) {
        //            cout << i << ": " << anglesCluster[i] << endl;
        //        }
        //        cout << "-------" << endl;
        //        cout << "Length Cluster" << endl;
        //        for (int i = 0; i < lengthsCluster.size(); i++) {
        //            cout << i << ": " << lengthsCluster[i] << endl;
        //        }
        //        cout << "-------" << endl;

        for (int i = 0; i < matches.size(); i++) {
            for (int a = 0; a < anglesCluster.size(); a++) {
                if (angles[i] == anglesCluster[a]) {
                    for (int b = 0; b < lengthsCluster.size(); b++) {
//                        if (lengths[i] == lengthsCluster[b]) {
                            //                            cout << "Angle " << angles[i] << " Length " << lengths[i] << endl;
                        filtredMatches.push_back(matches[i]);
//                        }
                    }
                }
            }
        }
    }
    else {
        vector<double> anglesCopy = angles;
        sort(anglesCopy.begin(), anglesCopy.end());
        double medianAngle = anglesCopy[floor(angles.size() * 0.5)];

//        vector<double> lengthCopy = lengths;
//        sort(lengthCopy.begin(), lengthCopy.end());
//        double medianLength = lengthCopy[floor(lengths.size() * 0.5)];

        //        cout << "Median both" << endl;
        for (int i = 0; i < matches.size(); i++) {
            if (angles[i] > medianAngle - this->parameters.angleParameter && angles[i] < medianAngle + this->parameters.angleParameter) {
//                if (lengths[i] > medianLength - this->parameters.lengthParameter && lengths[i] < medianLength + this->parameters.lengthParameter) {
                    //                    cout << "Angle " << angles[i] << " Length " << lengths[i] << endl;
                    filtredMatches.push_back(matches[i]);
//                }
            }
        }
        //        cout << "Median angle: " << medianAngle << endl;
        //        cout << "Median length: " << medianLength << endl;
    }

    //    for (int i = 0; i < lengths.size(); i++) {
    //        cout << i << ": " << lengths[i] << ";" << angles[i] << endl;
    //    }
}

// Filtering matches based on coordinates x or y. The main idea is that images cannot be shifted too far from each other. You can control the allowed shift by error paramter (expressed in percentage)
void AGMosaicStitcher::filterMatchesBasedOnPlacement(AGImage &imageOne,AGImage &imageTwo, std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &filtredMatches, ImageDirection imageDirection)
{
    if (matches.empty() || !imageOne.image.data || !imageTwo.image.data || this->parameters.shiftParameter > 1.0 || this->parameters.shiftParameter < 0.0) {
        return;
    }
    for (int i = 0; i < matches.size(); i++) {
        Point pointOne = imageOne.keypoints[matches[i].queryIdx].pt;
        Point pointTwo = imageTwo.keypoints[matches[i].trainIdx].pt;
        bool inRange = false;
        if (imageDirection == Up || imageDirection == Down) {
            if (abs(pointOne.x - pointTwo.x) < imageOne.width * this->parameters.shiftParameter) {
                inRange = true;
            }
        }
        else {
            if (abs(pointOne.y - pointTwo.y) < imageOne.height * this->parameters.shiftParameter) {
                inRange = true;
            }
        }
        if (inRange) {
            double distance = 0.0;
            switch (imageDirection) {
                case Up:
                    distance = sqrt(pow(abs(pointTwo.x - pointOne.x), 2.0) + pow(abs((pointOne.y + imageTwo.height) -pointTwo.y), 2.0));
                    break;

                case Down:
                    distance = sqrt(pow(abs(pointTwo.x - pointOne.x), 2.0) + pow(abs((pointTwo.y + imageOne.height) - pointOne.y), 2.0));
                    break;

                case Left:
                    distance = sqrt(pow(abs((pointOne.x + imageTwo.width) - pointTwo.x), 2.0) + pow(abs(pointTwo.y - pointOne.y), 2.0));
                    break;

                case Right:
                    distance = sqrt(pow(abs((pointTwo.x + imageOne.width) - pointOne.x), 2.0) + pow(abs(pointTwo.y - pointOne.y), 2.0));
                    break;
            }
            if (distance < imageOne.height * this->parameters.percentOverlap * 2.5) {
                filtredMatches.push_back(matches[i]);
            }
        }
    }
}

// Finding features that were matched to more than one feature and chosing the one with the minimum distance
void AGMosaicStitcher::deleteMatchesFromMultipleKeypointsToMultiple(vector<DMatch> &matches, vector<DMatch> &filtredMatches)
{
    if (matches.empty()) {
        return;
    }

    // Firstly we need to delete multiple matches from keypoints from first image and then we
    // need to delete multiple matches from keypoints from second image.
    vector<DMatch> matchesAfterFirstDeletion;

    // First deletion
    map<int, vector<DMatch>> repetitions;
    for (int i = 0; i < matches.size(); i++) {
        repetitions[matches[i].trainIdx].push_back(matches[i]);
    }
    map<int, vector<DMatch>>::iterator it = repetitions.begin();
    for (it = repetitions.begin(); it != repetitions.end(); ++it) {
        if (it->second.size() > 1) {
            float minDistance = 1000;
            int minIndex = 0;
            for (int i = 0; i < it->second.size(); i++) {
                if (minDistance < it->second[i].distance) {
                    minDistance = it->second[i].distance;
                    minIndex = i;
                }
            }
            matchesAfterFirstDeletion.push_back(it->second[minIndex]);
        }
        else {
            matchesAfterFirstDeletion.push_back(it->second.front());
        }
    }

    // Second deletion
    repetitions.clear();
    for (int i = 0; i < matchesAfterFirstDeletion.size(); i++) {
        repetitions[matchesAfterFirstDeletion[i].queryIdx].push_back(matchesAfterFirstDeletion[i]);
    }
    it = repetitions.begin();
    for (it = repetitions.begin(); it != repetitions.end(); ++it) {
        if (it->second.size() > 1) {
            float minDistance = 1000;
            int minIndex = 0;
            for (int i = 0; i < it->second.size(); i++) {
                if (minDistance < it->second[i].distance) {
                    minDistance = it->second[i].distance;
                    minIndex = i;
                }
            }
            filtredMatches.push_back(it->second[minIndex]);
        }
        else {
            filtredMatches.push_back(it->second.front());
        }
    }
}

void AGMosaicStitcher::filterMatchesUsingRANSAC(AGImage &imageOne, AGImage &imageTwo, vector<DMatch> &matches, vector<DMatch> &filtredMatches)
{
    if (matches.empty()) {
//        cerr << "Couldn't perform RANSAC. Matches vector is empty." << endl;
        return;
    }
    vector<Point2f> imageOneKeypoints;
    vector<Point2f> imageTwoKeypoints;
    for (int i = 0; i < matches.size(); i++) {
        Point2f pointOne = imageOne.keypoints[matches[i].queryIdx].pt;
        imageOneKeypoints.push_back(pointOne);
        Point2f pointTwo = imageTwo.keypoints[matches[i].trainIdx].pt;
        imageTwoKeypoints.push_back(pointTwo);
    }
    Mat mask;
    if (imageTwoKeypoints.size() >= 4 && imageOneKeypoints.size() >= 4) {
        Mat homography = findHomography(imageTwoKeypoints, imageOneKeypoints, CV_RANSAC, 3, mask);
        for (int i = 0; i < matches.size(); i++) {
            if((unsigned int)mask.at<uchar>(i)) {
                filtredMatches.push_back(matches[i]);
            }
        }
    }
    else {
        for (int i = 0; i < matches.size(); i++) {
            filtredMatches.push_back(matches[i]);
        }
    }
}

#pragma mark -
#pragma mark Features Extraction

void AGMosaicStitcher::findFeatures(AGImage &imageOne, AGImage &imageTwo, vector<ImageFeatures> &imagesFeatures, ImageDirection imageDirection)
{
    Rect roiImageOne, roiImageTwo;
    ImageFeatures imageOneFeatures, imageTwoFeatures;
    imagesFeatures.clear();

    switch (imageDirection) {
        case Up:
            roiImageOne = Rect(0, 0, imageOne.width, (double)imageOne.height * this->parameters.percentOverlap);
            roiImageTwo = Rect(0, (double)imageTwo.height * (1.0 - this->parameters.percentOverlap), imageTwo.width, (double)imageTwo.height * this->parameters.percentOverlap);
            break;

        case Down:
            roiImageOne = Rect(0, (double)imageOne.height * (1.0 - this->parameters.percentOverlap), imageOne.width, imageOne.height * this->parameters.percentOverlap);
            roiImageTwo = Rect(0, 0, imageTwo.width, (double)imageTwo.height * this->parameters.percentOverlap);
            break;

        case Left:
            roiImageOne = Rect(0, 0, (double)imageTwo.width * this->parameters.percentOverlap, imageTwo.height);
            roiImageTwo = Rect((double)imageTwo.width * (1.0 - this->parameters.percentOverlap), 0, (double)imageTwo.width * this->parameters.percentOverlap, imageTwo.height);
            break;

        case Right:
            roiImageOne = Rect((double)imageTwo.width * (1.0 - this->parameters.percentOverlap), 0, (double)imageOne.width * this->parameters.percentOverlap, imageOne.height);
            roiImageTwo = Rect(0, 0, (double)imageTwo.width * this->parameters.percentOverlap, imageTwo.height);
            break;
    }

    this->findFeaturesWithSIFT(imageOne, imageOneFeatures, roiImageOne);
    this->findFeaturesWithSIFT(imageTwo, imageTwoFeatures, roiImageTwo);

    if (!this->parameters.isAdHoc) {
        for (int i = 0; i <= imageOneFeatures.keypoints.size(); i++) {
            KeyPoint keyPoint = imageOneFeatures.keypoints[i];
            keyPoint.pt.x = keyPoint.pt.x + this->xShift;
            keyPoint.pt.y = keyPoint.pt.y + this->yShift;
            imageOneFeatures.keypoints[i] = keyPoint;
        }

        for (int i = 0; i <= imageTwoFeatures.keypoints.size(); i++) {
            KeyPoint keyPoint = imageTwoFeatures.keypoints[i];
            keyPoint.pt.x = keyPoint.pt.x + this->xShift;
            keyPoint.pt.y = keyPoint.pt.y + this->yShift;
            imageTwoFeatures.keypoints[i] = keyPoint;
        }
    }

    imagesFeatures.push_back(imageOneFeatures);
    imagesFeatures.push_back(imageTwoFeatures);
}

void AGMosaicStitcher::findFeaturesWithSIFT(AGImage &inputImage, ImageFeatures &imageFeatures, Rect &roi)
{
    SIFT featuresFinder = SIFT::SIFT(0, 3, 0.02, 10, 1.5);
//    SiftFeatureDetector featuresFinder = SiftFeatureDetector(0, 3, 0.04, 10, 1.5);
    Mat mask = Mat::zeros(inputImage.image.size(), CV_8UC1);
    Mat roin(mask, roi);
    roin = Scalar(255, 255, 255);
    featuresFinder(inputImage.image, mask, imageFeatures.keypoints, imageFeatures.descriptors);
}

#pragma mark -
#pragma mark Computing Transform

void AGMosaicStitcher::findTransformBetweenImages(AGImage &imageOne, AGImage &imageTwo, std::vector<cv::DMatch> &matches, cv::Mat &transform, ImageDirection imageDirection)
{
    vector<Point2f> imageOneSelectedKeypoints;
    vector<Point2f> imageTwoSelectedKeypoints;
    for (int i = 0; i < matches.size(); i++) {
        Point2f pointOne = imageOne.keypoints[matches[i].queryIdx].pt;
        imageOneSelectedKeypoints.push_back(pointOne);

        Point2f pointTwo = imageTwo.keypoints[matches[i].trainIdx].pt;
        imageTwoSelectedKeypoints.push_back(pointTwo);
    }

    // finding transform between images based on detected blood vessels paths
    if (this->parameters.usePaths && this->findTransformBasedOnPaths(imageOne, imageTwo, transform, imageDirection)) {
        AGError error;
        cout << "Path based transform between images: " << endl;
        cout << "1. " << AGOpenCVHelper::getDescriptionOfImage(imageOne, error) << endl;
        cout << "2. " << AGOpenCVHelper::getDescriptionOfImage(imageTwo, error) << endl << endl;
        return;
    }

    // simply shifting image when there is no keypoints detected
    if ((imageOneSelectedKeypoints.empty() || imageTwoSelectedKeypoints.empty())) {
        AGError error;
        cout << "Lack of keypoints in one of the images. Shifting images:" << endl;
        cout << "1. " << AGOpenCVHelper::getDescriptionOfImage(imageOne, error) << endl;
        cout << "2. " << AGOpenCVHelper::getDescriptionOfImage(imageTwo, error) << endl << endl;
        this->findShiftTransform(imageOne, imageTwo, transform, imageDirection);
        return;
    }

    // there are two ways of finding right transform, the first one is to use OpenCV method called
    // estimateRigitTransform(), and the second one just transform image based on mean Y difference and
    // mean X difference
    if (this->parameters.simplerTransform) {
        double meanXDiff = 0.0, meanYDiff = 0.0;
        double sumXDiff = 0.0, sumYDiff = 0.0;
        for (int i = 0; i < matches.size(); i++) {
            sumXDiff += imageTwoSelectedKeypoints[i].x - imageOneSelectedKeypoints[i].x;
            sumYDiff += imageTwoSelectedKeypoints[i].y - imageOneSelectedKeypoints[i].y;
        }
        meanXDiff = sumXDiff / matches.size();
        meanYDiff = sumYDiff / matches.size();
        AGOpenCVHelper::createShiftMatrix(transform, meanXDiff, meanYDiff);
        AGError error;
        cout << "Found simpler transform. Transforming images: " << endl;
        cout << "1. " << AGOpenCVHelper::getDescriptionOfImage(imageOne, error) << endl;
        cout << "2. " << AGOpenCVHelper::getDescriptionOfImage(imageTwo, error) << endl << endl;
        return;
    }
    else if (this->parameters.rigidTransform) {
        transform = estimateRigidTransform(imageOneSelectedKeypoints, imageTwoSelectedKeypoints, false);
    }

    if (transform.cols != 3 || transform.rows != 2) {
        AGError error;
        cout << "Found transform is invalid. Shifting images:" << endl;
        cout << "1. " << AGOpenCVHelper::getDescriptionOfImage(imageOne, error) << endl;
        cout << "2. " << AGOpenCVHelper::getDescriptionOfImage(imageTwo, error) << endl << endl;
        this->findShiftTransform(imageOne, imageTwo, transform, imageDirection);
        return;
    }

    AGError error;
    cout << "Found rigid transform. Transforming images:" << endl;
    cout << "1. " << AGOpenCVHelper::getDescriptionOfImage(imageOne, error) << endl;
    cout << "2. " << AGOpenCVHelper::getDescriptionOfImage(imageTwo, error) << endl << endl;
}

bool AGMosaicStitcher::findTransformBasedOnPaths(AGImage &imageOne, AGImage &imageTwo, Mat &transform, ImageDirection imageDirection)
{
    bool transformFound = false;
    if (!imageOne.pathPoints.empty() && !imageTwo.pathPoints.empty()) {
        Point imageOnePoint, imageTwoPoint;
        double deltaX, deltaY;
        switch (imageDirection) {
            case Up:
                if (!selectPointFromPath(imageTwo, Down, imageTwoPoint)
                    || !selectPointFromPath(imageOne, Up, imageOnePoint)) {
                    return false;
                }
                if(abs(double(imageOnePoint.x) - double(imageTwoPoint.x)) > PATH_RANGE) {
                    return false;
                }
                deltaX = -(double(imageOnePoint.x) - double(imageTwoPoint.x));
                deltaY = imageOne.height * (1 - this->parameters.percentOverlap);
                break;

            case Down:
                if (!selectPointFromPath(imageTwo, Up, imageTwoPoint)
                    || !selectPointFromPath(imageOne, Down, imageOnePoint)) {
                    return false;
                }
                if(abs(double(imageOnePoint.x) - double(imageTwoPoint.x)) > PATH_RANGE) {
                    return false;
                }
                deltaX = -(double(imageOnePoint.x) - double(imageTwoPoint.x));
                deltaY = -imageOne.height * (1 - this->parameters.percentOverlap);
                break;

            case Left:
                if (!selectPointFromPath(imageOne, Left, imageOnePoint)
                    || !selectPointFromPath(imageTwo, Right, imageTwoPoint)) {
                    return false;
                }
                if (abs(double(imageOnePoint.y) - double(imageTwoPoint.y)) > PATH_RANGE) {
                    return false;
                }
                deltaX = imageOne.width * (1 - this->parameters.percentOverlap);
                deltaY = -(double(imageOnePoint.y) - double(imageTwoPoint.y));
                break;

            case Right:
                if (!selectPointFromPath(imageOne, Right, imageOnePoint)
                    || !selectPointFromPath(imageTwo, Left, imageTwoPoint)) {
                    return false;
                }
                if (abs(double(imageOnePoint.y) - double(imageTwoPoint.y)) > PATH_RANGE) {
                    return false;
                }
                deltaX = -imageOne.width * (1 - this->parameters.percentOverlap);
                deltaY = -(double(imageOnePoint.y) - double(imageTwoPoint.y));
                break;
        }
        AGOpenCVHelper::createShiftMatrix(transform, deltaX, deltaY);
        transformFound = true;
    }
    return transformFound;
}

bool AGMosaicStitcher::selectPointFromPath(AGImage &image, ImageDirection desiredPlace, Point &point)
{
    bool pointSelected = false;
    point.x = image.width;
    point.y = image.height;
    for (int i = 0; i < image.pathPoints.size(); ++i) {
        switch (desiredPlace) {
            case Up:
                if (image.pathPoints[i].y == 0 && image.pathPoints[i].x < point.x) {
                    point = image.pathPoints[i];
                    pointSelected = true;
                }
                break;

            case Down:
                if (image.pathPoints[i].y == image.height - 1 && image.pathPoints[i].x < point.x) {
                    point = image.pathPoints[i];
                    pointSelected = true;
                }
                break;

            case Left:
                if (image.pathPoints[i].x == 0 && image.pathPoints[i].y < point.y) {
                    point = image.pathPoints[i];
                    pointSelected = true;
                }
                break;

            case Right:
                if (image.pathPoints[i].x == image.width - 1 && image.pathPoints[i].y < point.y) {
                    point = image.pathPoints[i];
                    pointSelected = true;
                }
                break;
        }
    }
    return pointSelected;
}

void AGMosaicStitcher::findShiftTransform(AGImage &imageOne, AGImage &imageTwo, Mat &transform, ImageDirection imageDirection)
{
    switch (imageDirection) {
        case Up:
            AGOpenCVHelper::createShiftMatrix(transform, 0.0, imageOne.height * (1 - this->parameters.percentOverlap));
            break;
        case Down:
            AGOpenCVHelper::createShiftMatrix(transform, 0.0, -imageOne.height * (1 - this->parameters.percentOverlap));
            break;
        case Left:
            AGOpenCVHelper::createShiftMatrix(transform, imageOne.width * (1 - this->parameters.percentOverlap), 0.0);
            break;
        case Right:
            AGOpenCVHelper::createShiftMatrix(transform, -imageOne.width * (1 - this->parameters.percentOverlap), 0.0);
            break;
    }
}

#pragma mark -
#pragma mark Helper Methods

void AGMosaicStitcher::clusterArrayWithinRange(std::vector<double> &array, std::vector<double> &output, double rangeParameter)
{
    vector<double> sortedArray = array;
    sort(sortedArray.begin(), sortedArray.end());
    vector<double> actualRange;
    int rangeStartIndex = 0, rangeEndIndex = 0;
    int startIndex = 0, endIndex = 1;

    if (sortedArray.size() == 1) {
        output.push_back(sortedArray[0]);
    }
    while (endIndex < sortedArray.size() && sortedArray.size() > 1) {
        actualRange.clear();
        for (int i = startIndex; i <= endIndex; i++) {
            actualRange.push_back(sortedArray[i]);
        }
        double maximumElement = 0;
        double minimumElement = 180;

        for (int i = 0; i < actualRange.size(); i++) {
            if (actualRange[i] >= maximumElement) {
                maximumElement = actualRange[i];
            }
        }

        for (int i = 0; i < actualRange.size(); i++) {
            if (actualRange[i] <= minimumElement) {
                minimumElement = actualRange[i];
            }
        }

        if (maximumElement - minimumElement <= rangeParameter) {
            if (endIndex - startIndex > rangeEndIndex - rangeStartIndex) {
                rangeStartIndex = startIndex;
                rangeEndIndex = endIndex;
            }
            endIndex++;
        }
        else {
            startIndex++;
            endIndex = startIndex + 1;
        }
    }

    for (int i = startIndex; i < endIndex; i++) {
        output.push_back(sortedArray[i]);
    }
}

#pragma mark -
#pragma mark Testing

void AGMosaicStitcher::testStitchBetweenTwoImages(AGImage &imageOne, AGImage &imageTwo, ImageDirection imageDirection)
{
    this->xShift = 0;
    this->yShift = 0;
    this->testingMode = false;

    Mat output, transform;
    this->applyStitchingAlgorithm(imageOne, imageTwo, imageDirection, transform);

    switch (imageDirection) {
        case Up:
        {
            output = Mat::zeros(imageOne.image.rows * 2, imageOne.image.cols, imageOne.image.type());
            warpAffine(imageOne.image, output, transform, output.size());
            imageTwo.image.copyTo(output(Rect(0, 0, imageTwo.image.cols, imageTwo.image.rows)));
        }
        break;
        case Down:
        {
            output = Mat::zeros(imageOne.image.rows * 2, imageOne.image.cols, imageOne.image.type());
            int rows = imageOne.image.rows;
            Mat baseShiftTransform;
            AGOpenCVHelper::createShiftMatrix(baseShiftTransform, 0, imageOne.height);
            warpAffine(imageOne.image, imageOne.image, baseShiftTransform, output.size());
            warpAffine(imageOne.image, output, transform, output.size());
            imageTwo.image.copyTo(output(Rect(0, rows, imageTwo.image.cols, imageTwo.image.rows)));
        }
        break;
        case Left:
        {
            output = Mat::zeros(imageOne.image.rows, imageOne.image.cols * 2, imageOne.image.type());
            warpAffine(imageOne.image, output, transform, output.size());
            imageTwo.image.copyTo(output(Rect(0, 0, imageTwo.image.cols, imageTwo.image.rows)));
        }
        break;
        case Right:
        {
            output = Mat::zeros(imageOne.image.rows, imageOne.image.cols * 2, imageOne.image.type());
            int cols = imageOne.image.cols;
            Mat baseShiftTransform;
            AGOpenCVHelper::createShiftMatrix(baseShiftTransform, imageOne.width, 0);
            warpAffine(imageOne.image, imageOne.image, baseShiftTransform, output.size());
            warpAffine(imageOne.image, output, transform, output.size());
            imageTwo.image.copyTo(output(Rect(cols, 0, imageTwo.image.cols, imageTwo.image.rows)));
        }
        break;
    }
    string windowName = "Stitch testing. Transformed ";
    windowName += imageOne.name;
    windowName += " reference ";
    windowName += imageTwo.name;
    AGError error;
    AGOpenCVHelper::saveImage(output, windowName.c_str(), this->parameters.mosaicsSaveAbsolutePath, error);
}

void AGMosaicStitcher::testImagePreprocessing(vector<vector<AGImage>> &imagesMatrix)
{
    for (int x = 0; x < imagesMatrix.size(); x++) {
        for (int y = 0; y < imagesMatrix.front().size(); y++) {
            Mat outputImage;
            this->imagePreprocessing->prepareForPathDetecting(imagesMatrix[x][y].image, outputImage);
            string imageName = to_string(x) + to_string(y);
            AGError error;
            AGOpenCVHelper::saveImage(outputImage, imageName, this->parameters.mosaicsSaveAbsolutePath, error);
        }
    }
}

