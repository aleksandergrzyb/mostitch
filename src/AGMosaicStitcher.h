//
//  Created by Aleksander Grzyb on 13/01/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#ifndef __Mosaic_Stitcher__AGMosaicStitcher__
#define __Mosaic_Stitcher__AGMosaicStitcher__

#include "AGDataStructures.h"
#include "AGPathDetection.h"
#include "AGOpenCVHelper.h"

#include <stdio.h>
#include <vector>
#include <opencv2/stitching/stitcher.hpp>
#include <opencv2/opencv.hpp>

 /// Responsible for stitching and producing final mosaic.

class AGMosaicStitcher {
public:
    
    /**
     *  Constructor of AGMosaicStitcher object.
     *
     *  @params parameters Loaded parameters from configuration file.
     */
    AGMosaicStitcher(const AGParameters &parameters);
    
    /**
     *  Starting point of whole stitching process. Takes matrix of tiles and produces mosaic.
     *
     *  @param imagesMatrix Matrix of image tiles.
     *  @param outputImage  Output Mosaic.
     *
     *  @return Error code (EXIT_SUCCESS or EXIT_FAILURE).
     */
    int stitchMosaic(std::vector<std::vector<AGImage>> &imagesMatrix, cv::Mat &outputImage);
private:
    
    /**
     *  Starting point of algorithm. At this point all necessary matrices are initialized.
     *
     *  @param imagesMatrix Matrix of image tiles.
     *  @param outputImage  Output mosaic.
     */
    void performStitching(std::vector<std::vector<AGImage>> &imagesMatrix, cv::Mat &outputImage);

    /**
     *  Filters matches that the output matches are from only one keypoint to only one keypoint.
     *  The situations in which one keypoint is matched to multiple is eliminated.
     *
     *  @param matches        Input matches.
     *  @param filtredMatches Output filtered matches.
     */
    void deleteMatchesFromMultipleKeypointsToMultiple(std::vector<cv::DMatch> &matches,
                                                      std::vector<cv::DMatch> &filtredMatches);
    
    /**
     *  Filters matched using RANSAC algorithm.
     *
     *  @param imageOne       First image.
     *  @param imageTwo       Second image.
     *  @param matches        Input matches between first image and second image.
     *  @param filtredMatches Output filtered matches.
     */
    void filterMatchesUsingRANSAC(AGImage &imageOne,
                                  AGImage &imageTwo,
                                  std::vector<cv::DMatch> &matches,
                                  std::vector<cv::DMatch> &filtredMatches);
    
    /**
     *  Filters matches based on slope and length of vector that connects keypoints of the match.
     *
     *  @param imageOne       First image.
     *  @param imageTwo       Second image.
     *  @param matches        Input matches between first image and second image.
     *  @param filtredMatches Output filtered matches.
     *  @param imageDirection Stitching direction (see ImageDirection enum in AGDataStructures.h).
     */
    void filterMatchesBasedOnSlopeAndLength(AGImage &imageOne,
                                            AGImage &imageTwo,
                                            std::vector<cv::DMatch> &matches,
                                            std::vector<cv::DMatch> &filtredMatches,
                                            ImageDirection imageDirection);
    
    /**
     *  Filters matches based on distance between keypoints in one of axis (depends on imageDirection).
     *
     *  @param imageOne       First image.
     *  @param imageTwo       Second image.
     *  @param matches        Input matches between first image and second image.
     *  @param filtredMatches Output filtered matches.
     *  @param imageDirection Stitching direction (see ImageDirection enum in AGDataStructures.h).
     */
    void filterMatchesBasedOnPlacement(AGImage &imageOne,
                                       AGImage &imageTwo,
                                       std::vector<cv::DMatch> &matches,
                                       std::vector<cv::DMatch> &filtredMatches,
                                       ImageDirection imageDirection);
    
    /**
     *  Filters matches.
     *
     *  @param imageOne       First image.
     *  @param imageTwo       Second image.
     *  @param matches        Input matches between first image and second image.
     *  @param filtredMatches Output filtered matches.
     *  @param imageDirection Stitching direction (see ImageDirection enum in AGDataStructures.h).
     */
    void filterMatches(AGImage &imageOne,
                       AGImage &imageTwo,
                       std::vector<cv::DMatch> &matches,
                       std::vector<cv::DMatch> &filtredMatches,
                       ImageDirection imageDirection);
    
    /**
     *  Stitches two images. Produces transformation matrix between those images.
     *
     *  @param imageOne       First image.
     *  @param imageTwo       Second image.
     *  @param imageDirection Stitching direction (see ImageDirection enum in AGDataStructures.h).
     *  @param transform      Output transformation matrix between images (the second one is transformed).
     */
    void applyStitchingAlgorithm(AGImage &imageOne,
                                 AGImage &imageTwo,
                                 ImageDirection imageDirection,
                                 cv::Mat &transform);

    /**
     *  Extract features of two images in their region of interest (ROI) based on image direction.
     *
     *  @param imageOne       First image.
     *  @param imageTwo       Second image.
     *  @param imagesFeatures Output features extracted from two images.
     *  @param imageDirection Stitching direction (see ImageDirection enum in AGDataStructures.h).
     */
    void findFeatures(AGImage &imageOne,
                      AGImage &imageTwo,
                      std::vector<cv::detail::ImageFeatures> &imagesFeatures,
                      ImageDirection imageDirection);
    
    /**
     *  Extracts SIFT features in ROI of input image.
     *
     *  @param inputImage    Input image.
     *  @param imageFeatures Output image features.
     *  @param roi           Region of interest.
     */
    void findFeaturesWithSIFT(AGImage &inputImage, cv::detail::ImageFeatures &imageFeatures, cv::Rect &roi);
    
    /**
     *  Finds matches between keypoints using Brute Force algorithm.
     *
     *  @param imagesFeatures Input keypoints vector.
     *  @param matchesInfo    Output matches.
     */
    void findMatchesWithBruteForce(std::vector<cv::detail::ImageFeatures> &imagesFeatures,
                                   cv::detail::MatchesInfo &matchesInfo);

    /**
     *  Calculates transformation matrix between two images (second one is transformed) based on matches.
     *
     *  @param imageOne       First image.
     *  @param imageTwo       Second image.
     *  @param matches        Input matches between first image and second image.
     *  @param transform      Output transformation matrix.
     *  @param imageDirection Stitching direction (see ImageDirection enum in AGDataStructures.h).
     */
    void findTransformBetweenImages(AGImage &imageOne,
                                    AGImage &imageTwo,
                                    std::vector<cv::DMatch> &matches,
                                    cv::Mat &transform,
                                    ImageDirection imageDirection);
    
    /**
     *  Calculates shift transform matrix between two images (second one is transformed) based on percent overlap 
     *  parameter from configuration file.
     *
     *  @param imageOne       First image.
     *  @param imageTwo       Second image.
     *  @param transform      Output transformation matrix.
     *  @param imageDirection Stitching direction (see ImageDirection enum in AGDataStructures.h).
     */
    void findShiftTransform(AGImage &imageOne, AGImage &imageTwo, cv::Mat &transform, ImageDirection imageDirection);
    
    /**
     *  Calculates transformation matrix between two image (second one is transformed) based on the result from
     *  path finding algorithm.
     *
     *  @param imageOne       First image.
     *  @param imageTwo       Second image.
     *  @param transform      Output transformation matrix.
     *  @param imageDirection Stitching direction (see ImageDirection enum in AGDataStructures.h).
     *
     *  @return Indicates if transformation matrix was found.
     */
    bool findTransformBasedOnPaths(AGImage &imageOne,
                                   AGImage &imageTwo,
                                   cv::Mat &transform,
                                   ImageDirection imageDirection);
    
    /**
     *  Helper method for findTransformBasedOnPaths(...) method. Selects point from path points of image if the point
     *  is inside desired area.
     *
     *  @param image        Input image.
     *  @param desiredPlace Desired area in which point is being searched.
     *  @param point        Output point.
     *
     *  @return Indicates if point was selected.
     */
    bool selectPointFromPath(AGImage &image, ImageDirection desiredPlace, cv::Point &point);

    /**
     *  Not used. Experimental method that was clustering values in array within given range parameter.
     *
     *  @param array          Input array.
     *  @param output         Output clustered values of array.
     *  @param rangeParameter Range parameter.
     */
    void clusterArrayWithinRange(std::vector<double> &array, std::vector<double> &output, double rangeParameter);

    /**
     *  Initialize empty transformation matrices matrix with given size.
     *
     *  @param xSize Number of columns in final mosaic.
     *  @param ySize Number of rows in final mosaic.
     */
    void initTransformsMatrix(int xSize, int ySize);

    /**
     *  Initialize empty mask in images matrix.
     *
     *  @param imagesMatrix Input matrix of tile images.
     */
    void initMaskInImagesMatrix(std::vector<std::vector<AGImage>> &imagesMatrix);

    /**
     *  Performs stitching algorithm between two images (testing purpose).
     *
     *  @param imageOne       First image.
     *  @param imageTwo       Second image.
     *  @param imageDirection Stitching direction (see ImageDirection enum in AGDataStructures.h).
     */
    void testStitchBetweenTwoImages(AGImage &imageOne, AGImage &imageTwo, ImageDirection imageDirection);
    
    /**
     *  Performs path detection algorithm (testing purpose).
     *
     *  @param imagesMatrix Input tile images matrix.
     */
    void testPathDetection(std::vector<std::vector<AGImage>> &imagesMatrix);

    /**
     *  Object responsible for path detection.
     */
    AGPathDetection *pathDetection;
    
    /**
     *  Loaded parameters from configuration file.
     */
    AGParameters parameters;
    
    /**
     *  Set this value if you want all mosaic be sifted in y axis.
     */
    int yShift;
    
    /**
     *  Set this value if you want all mosaic be sifted in x axis.
     */
    int xShift;
    
    /**
     *  Testing mode.
     */
    bool testingMode;
    
    /**
     *  Matrix of transformation matrices between tile images.
     */
    std::vector<std::vector<cv::Mat>> transformsMatrix;
};

#endif /* defined(__Mosaic_Stitcher__AGMosaicStitcher__) */
