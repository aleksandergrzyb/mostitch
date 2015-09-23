//
//  Created by Aleksander Grzyb on 01/04/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#ifndef __Mosaic_Stitcher__AGPathDetection__
#define __Mosaic_Stitcher__AGPathDetection__

#include "AGDataStructures.h"
#include "AGOpenCVHelper.h"

#include <stdio.h>
#include <opencv2/opencv.hpp>

/**
 *  Algorithm starts for searching paths from every edge of image.
 */
enum AGPathDirection {
    PathDown,
    PathUp,
    PathLeft,
    PathRight
};

enum AGVisitedPixelsMarkingMode {
    Row,
    Column
};

 /// Responsible for detecting blood vessels (as paths). The result of this method is a group of points indicating the existence of blood vessels at the edges of input image.
class AGPathDetection {
public:
    
    /**
     *  Constructor of AGPathDetection object.
     *
     *  @param parameters Loaded parameters from configuration file.
     */
    AGPathDetection(const AGParameters &parameters);
    
    /**
     *  Detects path in every image of imagesMatrix.
     *
     *  @param imagesMatrix Matrix of images.
     */
    void detectPaths(cv::vector<cv::vector<AGImage>> &imagesMatrix);
    
    /**
     *  Performs preprocessing of image required for better path detection results.
     *
     *  @param input  Input image.
     *  @param output Output Image after preprocessing.
     */
    void prepareForPathDetecting(cv::Mat &input, cv::Mat &output);
private:
    
    /**
     *  Entry point for searching paths.
     *
     *  @param image    Image in which paths will be searched.
     *  @param skeleton Skeleton of image.
     */
    void searchForPathsInImageUsingSkeleton(AGImage &image, cv::Mat &skeleton);
    
    /**
     *  Adds pixel (if white color) to depth first search stack data structure.
     *
     *  @param pixels        Stack of depth first search algorithm.
     *  @param pixel         Pixel that will be checked.
     *  @param image         Current image in which path detection is performing.
     *  @param visitedPixels Data structure that indicates already visited pixels by depth first search algorithm.
     */
    void addPixelIfWhiteInImage(std::vector<cv::Point> &pixels,
                                cv::Point pixel,
                                cv::Mat &image,
                                std::vector<std::vector<bool>> &visitedPixels);
    
    /**
     *  Adds adjacent pixels to pixel to depth first search stack data structure.
     *
     *  @param pixels        Stack of depth first search algorithm.
     *  @param pixel         Pixel to for which adjacent pixels will be added.
     *  @param image         Current image in which path detection is performing.
     *  @param pathDirection Direction of path searching.
     *  @param visitedPixels Data structure that indicates already visited pixels by depth first search algorithm.
     */
    void pushBackAdjacentPixelsToPixelInImage(std::vector<cv::Point> &pixels,
                                              cv::Point pixel,
                                              cv::Mat &image,
                                              AGPathDirection pathDirection,
                                              std::vector<std::vector<bool>> &visitedPixels);
    
    /**
     *  Creates skeleton of image.
     *
     *  @param input  Input image.
     *  @param output Output skeleton of input image.
     */
    void createSkeleton(cv::Mat &input, cv::Mat &output);
    
    /**
     *  Performs morphology operations on input image. This method is part of preprocessing.
     *
     *  @param input  Input image.
     *  @param output Output preprocessed input image.
     */
    void performMorphology(cv::Mat &input, cv::Mat &output);
    
    /**
     *  Adds all white pixels of skeleton to whitePixels vector.
     *
     *  @param skeleton    Image skeleton.
     *  @param whitePixels Vector of white pixels.
     */
    void findWhitePixelsInSkeleton(cv::Mat &skeleton, std::vector<cv::Point> &whitePixels);
    
    /**
     *  Sorts white pixels vector based on their position in the image.
     *
     *  @param whitePixels       White pixels vector.
     *  @param topWhitePixels    White pixels that are in the first row of image.
     *  @param bottomWhitePixels White pixels that are in the last row of image.
     *  @param leftWhitePixels   White pixels that are in the first column of image.
     *  @param rightWhitePixels  White pixels that are in the last column of image.
     *  @param imageWidth        Width of the image.
     *  @param imageHeight       Height of the image.
     */
    void categorizeWhitePixels(std::vector<cv::Point> &whitePixels,
                               std::vector<cv::Point> &topWhitePixels,
                               std::vector<cv::Point> &bottomWhitePixels,
                               std::vector<cv::Point> &leftWhitePixels,
                               std::vector<cv::Point> &rightWhitePixels,
                               int imageWidth, int imageHeight);
    
    /**
     *  Removes pixels that are close to each other.
     *
     *  @param whitePixels   Pixels vector.
     *  @param pathDirection Direction of path searching.
     */
    void removeNeighbourWhitePixels(std::vector<cv::Point> &whitePixels, AGPathDirection pathDirection);
    
    /**
     *  Performs depth first search algorithm for direction of searching.
     *
     *  @param image         Input image.
     *  @param skeleton      Skeleton of input image.
     *  @param whitePixels   White pixels found in skeleton of input image.
     *  @param pathDirection Direction of path searching.
     */
    void performDFSForPathDirection(AGImage &image, cv::Mat &skeleton,
                                    std::vector<cv::Point> &whitePixels,
                                    AGPathDirection pathDirection);
    
    /**
     *  Resets depth first search stack data structure.
     *
     *  @param visitedPixels Data structure that indicates already visited pixels by depth first search algorithm.
     *  @param width         Width of image.
     *  @param height        Height of image.
     */
    void resetVisitedPixels(std::vector<std::vector<bool>> &visitedPixels, int width, int height);

    /**
     *  Removes duplicates of path points. This is done, because algorithm is going from left to right, and from
     *  right to left detecting this way similar points.
     *
     *  @param image Input image.
     */
    void removePathPointsDuplicates(AGImage &image);

    /**
     *  Draws detected path points on the image.
     *
     *  @param image Input image.
     */
    void testDetectedPathsInImage(AGImage &image);
    
    /**
     *  Performs path detection on only one image.
     *
     *  @param image Input image.
     */
    void testSelectedImage(AGImage &image);
    
    /**
     *  Saves and shows current state of depth first search algorithm.
     *
     *  @param pixels        Stack of depth first search algorithm.
     *  @param visitedPixels Data structure that indicates already visited pixels by depth first search algorithm.
     *  @param image         Input image.
     */
    void testSaveCurrentDFS(std::vector<cv::Point> &pixels, cv::vector<cv::vector<bool>> &visitedPixels, cv::Mat &image);

    /**
     *  When is set to true algorithm goes into testing mode.
     */
    bool testingMode;
    
    /**
     *  Loaded parameters from configuration file.
     */
    AGParameters parameters;
};

#endif /* defined(__Mosaic_Stitcher__AGPathDetection__) */
