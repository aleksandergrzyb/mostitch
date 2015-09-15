//
//  Created by Aleksander Grzyb on 01/04/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#include "AGImagePreprocessing.h"

#include <algorithm>

using namespace std;
using namespace cv;

#pragma mark -
#pragma mark Initialization

AGImagePreprocessing::AGImagePreprocessing(const AGParameters &parameters)
{
    this->parameters = parameters;
}

#pragma mark -
#pragma mark Helper Methods

bool sortFunction(int i, int j)
{
    return i > j;
}

bool pathSortFunction(Point point1, Point point2)
{
    return point1.x > point2.x;
}

#pragma mark -
#pragma mark Starting Point

void AGImagePreprocessing::detectPaths(cv::vector<cv::vector<AGImage>> &imagesMatrix)
{
    this->testingMode = false;
//    this->testSelectedImage(imagesMatrix[1][0]);
//    this->testSelectedImage(imagesMatrix[1][0]);
    for (int x = 0; x < imagesMatrix.size(); x++) {
        for (int y = 0; y < imagesMatrix.front().size(); y++) {
            Mat skeleton;
            this->prepareForPathDetecting(imagesMatrix[x][y].image, skeleton);
            this->searchForPathsInImageUsingSkeleton(imagesMatrix[x][y], skeleton);
            this->testDetectedPathsInImage(imagesMatrix[x][y]);
        }
    }
}

void AGImagePreprocessing::prepareForPathDetecting(Mat &input, Mat &output)
{
    this->performMorphology(input, output);
    this->createSkeleton(output, output);
}

#pragma mark -
#pragma mark Morphology

void AGImagePreprocessing::performMorphology(cv::Mat &input, cv::Mat &output)
{
    threshold(input, output, 180, 255, THRESH_BINARY);
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(output, output, MORPH_CLOSE, element);
    element = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(output, output, MORPH_OPEN, element);
}

#pragma mark -
#pragma mark Finding Skeleton

void AGImagePreprocessing::createSkeleton(cv::Mat &input, cv::Mat &output)
{
    Mat skeleton(input.size(), CV_8UC1, Scalar(0));
    Mat temp(input.size(), CV_8UC1);
    Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
    bool done = false;
    int iterator = 0;
    while (!done) {
        AGError error;
//        AGOpenCVHelper::saveImage(input, to_string(iterator) + "_input_before", this->parameters.mosaicsSaveAbsolutePath, error);
        morphologyEx(input, temp, MORPH_OPEN, element);
//        AGOpenCVHelper::saveImage(temp, to_string(iterator) + "_temp_after_open", this->parameters.mosaicsSaveAbsolutePath, error);
        bitwise_not(temp, temp);
//        AGOpenCVHelper::saveImage(temp, to_string(iterator) + "_temp_after_not", this->parameters.mosaicsSaveAbsolutePath, error);
        bitwise_and(input, temp, temp);
//        AGOpenCVHelper::saveImage(temp, to_string(iterator) + "_temp_after_and", this->parameters.mosaicsSaveAbsolutePath, error);
        bitwise_or(skeleton, temp, skeleton);
//        AGOpenCVHelper::saveImage(skeleton, to_string(iterator) + "_skeleton_after_or", this->parameters.mosaicsSaveAbsolutePath, error);
        erode(input, input, element);
        double max;
//        AGOpenCVHelper::saveImage(input, to_string(iterator) + "_input_after_erode", this->parameters.mosaicsSaveAbsolutePath, error);
        minMaxLoc(input, 0, &max);
        done = (max == 0);
        ++iterator;
    };
    output = skeleton;
}

#pragma mark -
#pragma mark Paths Searching

void AGImagePreprocessing::searchForPathsInImageUsingSkeleton(AGImage &image, cv::Mat &skeleton)
{
    // Finding all whites pixels in every edge of skeleton
    vector<Point> whitePixels;
    this->findWhitePixelsInSkeleton(skeleton, whitePixels);

    vector<Point> topWhitePixels, bottomWhitePixels, leftWhitePixels, rightWhitePixels;
    this->categorizeWhitePixels(whitePixels, topWhitePixels, bottomWhitePixels, leftWhitePixels, rightWhitePixels, skeleton.cols, skeleton.rows);

    // Removing pixels that are next to each other
    if (!whitePixels.empty()) {
        this->removeNeighbourWhitePixels(topWhitePixels, PathDown);
        this->removeNeighbourWhitePixels(bottomWhitePixels, PathUp);
        this->removeNeighbourWhitePixels(leftWhitePixels, PathRight);
        this->removeNeighbourWhitePixels(rightWhitePixels, PathLeft);

        // Searching for paths from 4 directions
        this->performDFSForPathDirection(image, skeleton, topWhitePixels, PathDown);
        this->performDFSForPathDirection(image, skeleton, bottomWhitePixels, PathUp);
        this->performDFSForPathDirection(image, skeleton, leftWhitePixels, PathRight);
        this->performDFSForPathDirection(image, skeleton, rightWhitePixels, PathLeft);
    }

    this->removePathPointsDuplicates(image);
}

void AGImagePreprocessing::findWhitePixelsInSkeleton(Mat &skeleton, vector<Point> &whitePixels)
{
    uchar *pixel = skeleton.ptr<uchar>(FIRST_ROW);
    uchar pixelValue;
    for (int y = 0; y < skeleton.rows; y++) {
        for (int x = 0; x < skeleton.cols; x++) {
            pixelValue = *pixel++;
            if (x == 0 || x == skeleton.cols - 1 || y == 0 || y == skeleton.rows - 1) {
                if (pixelValue == WHITE_PIXEL) {
                    Point whitePixel(x, y);
                    whitePixels.push_back(whitePixel);
                }
            }
        }
    }
}

void AGImagePreprocessing::pushBackAdjacentPixelsToPixelInImage(vector<Point> &pixels, Point pixel, Mat &image,
                                                                PathDirection pathDirection, vector<vector<bool>> &visitedPixels)
{
    vector<pair<int, int>> offsetPoints;

    // below are cases for path Up
    int numberOfCases = 12;
    offsetPoints.push_back(pair<int, int>(0, -1));
    offsetPoints.push_back(pair<int, int>(-1, -1));
    offsetPoints.push_back(pair<int, int>(1, -1));
    offsetPoints.push_back(pair<int, int>(1, 0));
    offsetPoints.push_back(pair<int, int>(-1, 0));
    offsetPoints.push_back(pair<int, int>(-2, 0));
    offsetPoints.push_back(pair<int, int>(2, 0));
    offsetPoints.push_back(pair<int, int>(-2, -1));
    offsetPoints.push_back(pair<int, int>(2, -1));
    offsetPoints.push_back(pair<int, int>(-1, -2));
    offsetPoints.push_back(pair<int, int>(0, -2));
    offsetPoints.push_back(pair<int, int>(1, -2));

    for (int i = 0; i < numberOfCases; i++) {
        switch (pathDirection) {
            case Down:
                this->addPixelIfWhiteInImage(pixels, Point(pixel.x + offsetPoints[i].first, pixel.y - offsetPoints[i].second), image, visitedPixels);
                break;
            case Up:
                this->addPixelIfWhiteInImage(pixels, Point(pixel.x + offsetPoints[i].first, pixel.y + offsetPoints[i].second), image, visitedPixels);
                break;
            case Right:
                this->addPixelIfWhiteInImage(pixels, Point(pixel.x - offsetPoints[i].second, pixel.y + offsetPoints[i].first), image, visitedPixels);
                break;
            case Left:
                this->addPixelIfWhiteInImage(pixels, Point(pixel.x + offsetPoints[i].second, pixel.y + offsetPoints[i].first), image, visitedPixels);
                break;
        }
    }
}

#pragma mark -
#pragma mark Cleaning

void AGImagePreprocessing::removePathPointsDuplicates(AGImage &image)
{
    if (image.pathPoints.size() < 2) {
        return;
    }
    Point currentPoint = image.pathPoints.front();
    vector<int> indexesToDelete;
    sort(image.pathPoints.begin(), image.pathPoints.end(), pathSortFunction);
    for (int i = 1; i < image.pathPoints.size(); i++) {
        if (image.pathPoints[i] == currentPoint) {
            indexesToDelete.push_back(i);
        }
        else {
            currentPoint = image.pathPoints[i];
        }
    }
    for (int i = int(indexesToDelete.size()) - 1; i >= 0; i--) {
        image.pathPoints.erase(image.pathPoints.begin() + indexesToDelete[i]);
    }
}

#pragma mark -
#pragma mark White Pixels

void AGImagePreprocessing::addPixelIfWhiteInImage(vector<Point> &pixels, Point pixel, Mat &image, vector<vector<bool>> &visitedPixels)
{
    if (pixel.x >= 0 && pixel.x < image.cols && pixel.y >= 0 && pixel.y < image.rows) {
        AGError error;
        if (AGOpenCVHelper::isPixelWhiteInImage(image, pixel, error) && !visitedPixels[pixel.x][pixel.y]) {
            pixels.push_back(pixel);
        }
    }
}

void AGImagePreprocessing::categorizeWhitePixels(vector<Point> &whitePixels, vector<Point> &topWhitePixels, vector<Point> &bottomWhitePixels, vector<Point> &leftWhitePixels, vector<Point> &rightWhitePixels, int imageWidth, int imageHeight)
{
    for (int i = 0; i < whitePixels.size(); i++) {
        Point whitePixel = whitePixels[i];
        if (whitePixel.y == 0) {
            topWhitePixels.push_back(whitePixel);
        }
        else if (whitePixel.y == imageHeight - 1) {
            bottomWhitePixels.push_back(whitePixel);
        }
        else if (whitePixel.x == 0) {
            leftWhitePixels.push_back(whitePixel);
        }
        else if (whitePixel.x == imageWidth - 1) {
            rightWhitePixels.push_back(whitePixel);
        }
    }
}

void AGImagePreprocessing::removeNeighbourWhitePixels(vector<Point> &whitePixels, PathDirection pathDirection)
{
    if (whitePixels.empty()) {
        return;
    }

    vector<int> indexesToRemove;
    for (int i = 0; i < whitePixels.size() - 1; i++) {
        switch (pathDirection) {
            case PathDown:
            case PathUp:
                if (whitePixels[i + 1].x - 1 == whitePixels[i].x) {
                    indexesToRemove.push_back(i + 1);
                }
                break;
            case PathRight:
            case PathLeft:
                if (whitePixels[i + 1].y - 1 == whitePixels[i].y) {
                    indexesToRemove.push_back(i + 1);
                }
                break;
        }

    }
    sort(indexesToRemove.begin(), indexesToRemove.end(), sortFunction);
    for (int i = 0; i < indexesToRemove.size(); i++) {
        whitePixels.erase(whitePixels.begin() + indexesToRemove[i]);
    }
}

#pragma mark -
#pragma mark Depth First Search

void AGImagePreprocessing::performDFSForPathDirection(AGImage &image, Mat &skeleton, vector<Point> &whitePixels, PathDirection pathDirection)
{
    if (whitePixels.empty()) {
        return;
    }

    vector<vector<bool>> visitedPixels;
    for (int i = 0; i < whitePixels.size(); i++) {
        vector<Point> pixelStack;
        this->resetVisitedPixels(visitedPixels, image.width, image.height);
        bool foundPoint = false;
        pixelStack.push_back(whitePixels[i]);
        while (!pixelStack.empty()) {
            Point pixel = pixelStack.back();
            if (pixel.x != whitePixels[i].x || pixel.y != whitePixels[i].y) {
                double distance = sqrt(pow(whitePixels[i].x - pixel.x, 2.0) + pow(whitePixels[i].y - pixel.y, 2.0));
                switch (pathDirection) {
                    case PathUp:
                    case PathDown:
                        if (distance > image.height * 0.5 && abs(pixel.y - whitePixels[i].y) > POINT_Y_OFFSET * image.height) {
                            image.pathPoints.push_back(whitePixels[i]);
                            foundPoint = true;
                        }
                        break;

                    case PathRight:
                    case PathLeft:
                        if (distance > image.width * 0.5 && abs(pixel.x - whitePixels[i].x) > POINT_X_OFFSET * image.width) {
                            image.pathPoints.push_back(whitePixels[i]);
                            foundPoint = true;
                        }
                        break;
                }
            }
            if (!foundPoint) {
                pixelStack.pop_back();
                visitedPixels[pixel.x][pixel.y] = true;
                this->pushBackAdjacentPixelsToPixelInImage(pixelStack, pixel, skeleton, pathDirection, visitedPixels);
            }
            if (this->testingMode) {
                this->testSaveCurrentDFS(pixelStack, visitedPixels, skeleton);
            }
            if (foundPoint) {
                pixelStack.clear();
            }
        }
    }
}

void AGImagePreprocessing::resetVisitedPixels(std::vector<std::vector<bool> > &visitedPixels, int width, int height)
{
    visitedPixels.clear();
    for (int x = 0; x < width; x++) {
        vector<bool> heightPixels;
        for (int y = 0; y < height; y++) {
            heightPixels.push_back(false);
        }
        visitedPixels.push_back(heightPixels);
    }
}

void AGImagePreprocessing::markVisitedPixels(vector<vector<bool>> &visitedPixels, MarkingVisitedPixelsMode markingVisitedPixelsMode, int rowOrColumnCor)
{
    if (rowOrColumnCor < 0 || rowOrColumnCor > visitedPixels.size() || rowOrColumnCor > visitedPixels.front().size()) {
        return;
    }

    switch (markingVisitedPixelsMode) {
        case Column:
            for (int y = 0; y < visitedPixels.front().size(); y++) {
                visitedPixels[rowOrColumnCor][y] = true;
            }
            break;

        case Row:
            for (int x = 0; x < visitedPixels.size(); x++) {
                visitedPixels[x][rowOrColumnCor] = true;
            }
            break;
    }
}

#pragma mark -
#pragma mark Testing

void AGImagePreprocessing::testDetectedPathsInImage(AGImage &image)
{
    static int iterator = 0;
    Mat testImage = image.image.clone();
    AGOpenCVHelper::convertImageToColor(testImage);
    for (int i = 0; i < image.pathPoints.size(); i++) {
        Rect testImageRect(Point(), testImage.size());
        for (int x = -2; x < 3; ++x) {
            for (int y = -2; y < 3; ++y) {
                Point currentPoint(image.pathPoints[i].y + y, image.pathPoints[i].x + x);
                if (testImageRect.contains(currentPoint)) {
                    testImage.at<cv::Vec4b>(currentPoint.x, currentPoint.y)[0] = 0;
                    testImage.at<cv::Vec4b>(currentPoint.x, currentPoint.y)[1] = 0;
                    testImage.at<cv::Vec4b>(currentPoint.x, currentPoint.y)[2] = 255;
                    testImage.at<cv::Vec4b>(currentPoint.x, currentPoint.y)[3] = 0;

//                    testImage.at<cv::Vec4b>(image.pathPoints[i].y, image.pathPoints[i].x)[0] = 0;
//                    testImage.at<cv::Vec4b>(image.pathPoints[i].y, image.pathPoints[i].x)[1] = 0;
//                    testImage.at<cv::Vec4b>(image.pathPoints[i].y, image.pathPoints[i].x)[2] = 255;
//                    testImage.at<cv::Vec4b>(image.pathPoints[i].y, image.pathPoints[i].x)[3] = 0;
                }
            }
        }
    }
    string imageName = "detected_path_points_" + to_string(iterator);
    iterator++;
    AGError error;
//    AGOpenCVHelper::saveImage(testImage, imageName, this->parameters.mosaicsSaveAbsolutePath, error);
}

void AGImagePreprocessing::testSaveCurrentDFS(vector<Point> &pixels, vector<vector<bool>> &visitedPixels, Mat &image)
{
    Mat testImage = image.clone();
    AGOpenCVHelper::convertImageToColor(testImage);
    for (int i = 0; i < pixels.size(); i++) {
        testImage.at<cv::Vec4b>(pixels[i].y, pixels[i].x)[0] = 255;
        testImage.at<cv::Vec4b>(pixels[i].y, pixels[i].x)[1] = 0;
        testImage.at<cv::Vec4b>(pixels[i].y, pixels[i].x)[2] = 0;
        testImage.at<cv::Vec4b>(pixels[i].y, pixels[i].x)[3] = 0;
    }

    for (int x = 0; x < visitedPixels.size(); x++) {
        for (int y = 0; y < visitedPixels.front().size(); y++) {
            if (visitedPixels[x][y]) {
                testImage.at<cv::Vec4b>(y, x)[0] = 0;
                testImage.at<cv::Vec4b>(y, x)[1] = 0;
                testImage.at<cv::Vec4b>(y, x)[2] = 255;
                testImage.at<cv::Vec4b>(y, x)[3] = 0;
            }
        }
    }
    AGError error;
    AGOpenCVHelper::showImage(testImage, "DFS", error);
    AGOpenCVHelper::saveImage(testImage, "DFS", this->parameters.mosaicsSaveAbsolutePath, error);
}

void AGImagePreprocessing::testSelectedImage(AGImage &image)
{
    Mat skeleton;
    this->prepareForPathDetecting(image.image, skeleton);
    AGError error;
    AGOpenCVHelper::saveImage(skeleton, "skeleton", this->parameters.mosaicsSaveAbsolutePath, error);
    this->searchForPathsInImageUsingSkeleton(image, skeleton);
    this->testDetectedPathsInImage(image);
    this->testingMode = false;
}









