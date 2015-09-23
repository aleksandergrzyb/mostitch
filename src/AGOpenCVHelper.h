//
//  Created by Aleksander Grzyb on 14/01/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#ifndef __Mosaic_Stitcher__AGOpenCVHelper__
#define __Mosaic_Stitcher__AGOpenCVHelper__

#include "AGDataStructures.h"

#include <opencv2/opencv.hpp>
#include <stdio.h>

#define A_COEFF 0
#define B_COEFF 1

 /// Contains helper methods for whole project.

class AGOpenCVHelper {
public:
    
    /**
     *  Converts image to color.
     *
     *  @param image Input/Output image.
     */
    static void convertImageToColor(cv::Mat &image);
    
    /**
     *  Converts image to grayscale.
     *
     *  @param image Input/Output image.
     */
    static void convertImageToGrayscale(cv::Mat &image);
    
    /**
     *  Converts image type (as number) to readable string format.
     *
     *  @param imageType Image type.
     *  @param error     Error.
     *
     *  @return Image type as string.
     */
    static std::string getImageTypeNameUsingImageType(const int imageType, AGError &error);
    
    /**
     *  Calculates linear function coefficients from two points.
     *
     *  @param pointOne First point.
     *  @param pointTwo Second point.
     *  @param coeffs   Output vector of two coefficients.
     */
    static void linearFunctionCoeffsUsingPoints(const cv::Point &pointOne,
                                                         const cv::Point &pointTwo,
                                                         std::vector<double> &coeffs);
    
    /**
     *  Calculates distance between two points.
     *
     *  @param pointOne First point.
     *  @param pointTwo Second point.
     *
     *  @return Distance between two points.
     */
    static double distanceBetweenPoints(const cv::Point &pointOne, const cv::Point &pointTwo);
    
    /**
     *  Creates empty Mat object with given size and type.
     *
     *  @param image     Output image.
     *  @param size      Size of desired image.
     *  @param imageType Type of desired image.
     */
    static void createEmptyMatWithSize(cv::Mat &image, const cv::Size &size, const int imageType);
    
    /**
     *  Creates white rectangle (image mask) with given size.
     *
     *  @param image Output image.
     *  @param size  Size of desired mask.
     */
    static void createMaskMatWithSize(cv::Mat &image, const cv::Size &size);
    
    /**
     *  Checks if pixel in given pixel position is white in image.
     *
     *  @param image         Image to check.
     *  @param pixelPosition Pixel position in image.
     *  @param error         Error.
     *
     *  @return Boolean indicating if pixel is white.
     */
    static bool isPixelWhiteInImage(cv::Mat &image, const cv::Point &pixelPosition, AGError &error);
    
    /**
     *  Value of pixel in given pixel position in image.
     *
     *  @param image         Image to check.
     *  @param pixelPosition Pixel position in image.
     *  @param error         Error.
     *
     *  @return Value of pixel.
     */
    static int pixelValueAtPointInImage(const cv::Mat &image, const cv::Point &pixelPosition, AGError &error);
    
    /**
     *  Sets pixel at pixel position to a given value in image.
     *
     *  @param image         Image.
     *  @param pixelPosition Pixel position in image.
     *  @param pixelValue    Value of pixel to set.
     *  @param error         Error.
     */
    static void setPixelValueAtPointInImage(const cv::Mat &image,
                                            const cv::Point &pixelPosition,
                                            const int pixelValue,
                                            AGError &error);
    
    /**
     *  Creates shif matrix with given parameters.
     *
     *  @param shiftMatrix Output shift matrix.
     *  @param dx          Translation in x axis.
     *  @param dy          Translation in y axis.
     */
    static void createShiftMatrix(cv::Mat &shiftMatrix, const double dx, const double dy);
    
    /**
     *  Rotates the image by given angle.
     *
     *  @param image Input/Output image.
     *  @param angle Rotation angle.
     */
    static void rotateImage(cv::Mat &image, const double angle);
    
    /**
     *  Inserts source image into output image at given point (position).
     *
     *  @param sourceImage Image to be inserted.
     *  @param outputImage Image to which source image will be inserted.
     *  @param point       Postion of insertion.
     *  @param error       Error.
     */
    static void insertSourceImageIntoOutputImageAtPoint(const cv::Mat &sourceImage,
                                                        cv::Mat &outputImage,
                                                        const cv::Point &point,
                                                        AGError &error);
    
    /**
     *  Inserts two images side by side (vertically or horizontally, specified by image direction) into output image.
     *
     *  @param imageOne       First image.
     *  @param imageTwo       Second image.
     *  @param outputImage    Output image.
     *  @param imageDirection Specifies if images will be stacked vertically or horizontally.
     *  @param error          Error.
     */
    static void linkTwoImagesTogether(const cv::Mat &imageOne,
                                      const cv::Mat &imageTwo,
                                      cv::Mat &outputImage,
                                      const ImageDirection &imageDirection,
                                      AGError &error);
    
    /**
     *  Inserts two images side by side (vertically or horizontally, specified by image direction) into output image
     *  and draw given matches between them.
     *
     *  @param imageOne       First image.
     *  @param imageTwo       Second image.
     *  @param matches        Matches to draw between images.
     *  @param outputImage    Output image.
     *  @param imageDirection Specifies if images will be stacked vertically or horizontally.
     *  @param error          Error.
     */
    static void linkTwoImagesTogetherAndDrawMatches(const AGImage &imageOne,
                                                    const AGImage &imageTwo,
                                                    const std::vector<cv::DMatch> &matches,
                                                    cv::Mat &outputImage,
                                                    const ImageDirection &imageDirection,
                                                    AGError &error);

    /**
     *  Saves image to disc.
     *
     *  @param image    Image to save.
     *  @param name     Name of image file.
     *  @param savePath Path to save.
     *  @param error    Error.
     */
    static void saveImage(cv::Mat &image, const std::string &name, const std::string &savePath, AGError &error);
    
    /**
     *  Saves detected keypoints vector to file.
     *
     *  @param image    Image with detected keypoints.
     *  @param fileName File name to save.
     *  @param savePath File path to save.
     *  @param error    Error.
     */
    static void saveKeypointsToFile(const AGImage &image,
                                    const std::string &fileName,
                                    const std::string &savePath,
                                    AGError &error);
    
    /**
     *  Saves detected matches vector to file.
     *
     *  @param matches  Deteceted matches vector.
     *  @param fileName File name to save.
     *  @param savePath File path to save.
     *  @param error    Error.
     */
    static void saveMatchesToFile(const std::vector<cv::DMatch> &matches,
                                  const std::string &fileName,
                                  const std::string &savePath,
                                  AGError &error);
    
    /**
     *  Shows given image in window.
     *
     *  @param image      Image to show.
     *  @param windowName Window name.
     *  @param error      Error.
     */
    static void showImage(const cv::Mat &image, const std::string &windowName, AGError &error);
    
    /**
     *  Returns full description of image.
     *
     *  @param image Image.
     *  @param error Error.
     *
     *  @return Description of image.
     */
    static std::string getDescriptionOfImage(const AGImage &image, AGError &error);
    
    /**
     *  Loads image from disc.
     *
     *  @param image    Output image.
     *  @param loadPath Image file path.
     *  @param error    Error.
     */
    static void loadImage(cv::Mat &image, const std::string &loadPath, AGError &error);
};

#endif /* defined(__Mosaic_Stitcher__AGOpenCVHelper__) */
