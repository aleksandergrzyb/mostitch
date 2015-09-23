//
//  Created by Aleksander Grzyb on 06/02/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#ifndef Mosaic_Stitcher_AGDataStructures_h
#define Mosaic_Stitcher_AGDataStructures_h

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>

/**
 *  Constant for pixel value.
 */
const int WHITE_PIXEL = 0xFF;

/**
 *  Constants for AGPathDetection class.
 */
const double POINT_Y_OFFSET = 0.1;
const double POINT_X_OFFSET = 0.1;
const int FIRST_ROW = 0;

/**
 *  Constants for AGMosaicStitcher class.
 */
const double PATH_RANGE = 20.0;

/**
 *  Informs about relationship between two images. For example direction 'Up' tells that first image is below second
 *  image and the first image would be transformed.
 */
enum ImageDirection {
    Down = 0,
    Up = 1,
    Left = 2,
    Right = 3
};

/// Helper for error handling. Structure is passed with most of the method calls and then used inside those methods to report errors to the caller.

struct AGError {
    
    /**
     *  Informs about existence of error.
     */
    bool isError;
    
    /**
     *  Error description.
     */
    std::string description;
};

 /// Captures all program parameters.

struct AGParameters {
    
    /**
     *  Parameter used to control filtering of matches. Explained in chapter 4.2.3 in master's thesis. Required to set
     *  in configuration file.
     */
    double angleParameter;
    
    /**
     *  Parameter used to control filtering of matches. Explained in chapter 4.2.3 in master's thesis. Required to set
     *  in configuration file.
     */
    double percentOverlap;
    
    /**
     *  Parameter used to control filtering of matches. Explained in chapter 4.2.3 in master's thesis. Required to set 
     *  in configuration file.
     */
    double shiftParameter;
    
    /**
     *  Parameter used to control number of mosaic to load from disc. Required to set in configuration file.
     */
    int numberOfMosaics;
    
    /**
     *  Parameter used to specify path for saving final mosaic to disc. Required to set in configuration file.
     */
    std::string mosaicsSaveAbsolutePath;
    
    /**
     *  Parameter used to specify path for loading images from disc. Required to set in configuration file.
     */
    std::string mosaicsDirectoryAbsolutePath;
    
    /**
     *  Indicates if program should use simpler transform. Explained in chapter 4.4.5 in master's thesis. Set
     *  by program itself (not in configuration file).
     */
    bool simplerTransform;
    
    /**
     *  Indicates if program should use rigid transform. Explained in chapter 4.4.5 in master's thesis. Set
     *  by program itself (not in configuration file).
     */
    bool rigidTransform;
    
    /**
     *  Indicates if program should use blood vessels detection. Explained in chapter 4.4.5 in master's thesis. Set
     *  by program itself (not in configuration file).
     */
    bool usePaths;
    
    /**
     *  Currently unused. Developed to experiment with filtering matches by vector lengths.
     */
    double lengthParameter;
    
    /**
     *  Currently unused. Developed to experiment with ad-hoc method (dividing images into regions).
     */
    bool isAdHoc;
    
    /**
     *  Currently unused. Developed to experiment with matches clustering.
     */
    bool clustering;
};

 /// Main structure of program. Captures all properties of image.

struct AGImage {
    /**
     *  Constructor of AGImage.
     *
     *  @param image       Image data.
     *  @param xCoordinate Coordinate of image in final mosaic (x axis).
     *  @param yCoordinate Coordinate of image in final mosaic (y axis).
     *  @param width       Width of image (same as passed image.size().width).
     *  @param height      Height of image (same as passed image.size().height).
     *  @param name        Name of the file with image data.
     *
     *  @return AGImage object.
     */
    AGImage(cv::Mat image,
            int xCoordinate,
            int yCoordinate,
            int width,
            int height,
            std::string name) :
    image(image),
    xCoordinate(xCoordinate),
    yCoordinate(yCoordinate),
    width(width),
    height(height),
    name(name) {}
    
    /**
     *  Image data.
     */
    cv::Mat image;
    
    /**
     *  Mask of the image i.e. white rectangle that is exactly the same size and in the same position as image property.
     */
    cv::Mat mask;
    
    /**
     *  Value of distance transform of image mask.
     */
    cv::Mat distanceTransform;
    
    /**
     *  Width of image (done, because during creation of final grid the image is copied and
     *  loses information about its original width and height, which is needed for matches filtering).
     */
    int width;
    
    /**
     *  Height of image (done, because during creation of final grid the image is copied and
     *  loses information about its original width and height, which is needed for matches filtering).
     */
    int height;
    
    /**
     *  Coordinate of image in final mosaic (x axis).
     */
    int xCoordinate;
    
    /**
     *  Coordinate of image in final mosaic (y axis).
     */
    int yCoordinate;
    
    /**
     *  Name of the file with image data.
     */
    std::string name;
    
    /**
     *  Keypoints detected from SIFT.
     */
    std::vector<cv::KeyPoint> keypoints;
    
    /**
     *  Points that point to the place where detected blood vessels are in the edges of image.
     */
    std::vector<cv::Point> pathPoints;
};

#endif
