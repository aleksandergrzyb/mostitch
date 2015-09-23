//
//  Created by Aleksander Grzyb on 14/01/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#include "AGOpenCVHelper.h"

#include <fstream>

using namespace cv;
using namespace std;

#pragma mark - 
#pragma mark Handling image types

void AGOpenCVHelper::convertImageToColor(Mat &image)
{
    cvtColor(image, image, CV_GRAY2RGBA);
}

void AGOpenCVHelper::convertImageToGrayscale(Mat &image)
{
    cvtColor(image, image, CV_RGBA2GRAY);
}

string AGOpenCVHelper::getImageTypeNameUsingImageType(int imageType, AGError &error)
{
    int numberOfImageTypes = 35;
    int enumInts[] = {CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
        CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
        CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
        CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
        CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
        CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
        CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4};
    
    string enumStrings[] = {"CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",
        "CV_8S",  "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",
        "CV_16U", "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
        "CV_16S", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
        "CV_32S", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
        "CV_32F", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
        "CV_64F", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"};
    
    for(int i = 0; i < numberOfImageTypes; i++) {
        if(imageType == enumInts[i]) return enumStrings[i];
    }
    error = { true, "getImageTypeNameUsingImageType: Unknown image type." }; return "";
}

#pragma mark - Geometric methods

void AGOpenCVHelper::linearFunctionCoeffsUsingPoints(const cv::Point &pointOne,
                                                     const cv::Point &pointTwo,
                                                     std::vector<double> &coeffs)
{
    if (pointTwo.x == pointOne.x) {
        coeffs.push_back(0.00);
        coeffs.push_back(0.00);
        return;
    }
    double aCoeff = (double)(pointTwo.y - pointOne.y) / (double)(pointTwo.x - pointOne.x);
    double bCoeff = (double)(pointOne.y) - aCoeff * (double)(pointOne.x);
    coeffs.push_back(aCoeff);
    coeffs.push_back(bCoeff);
}

double AGOpenCVHelper::distanceBetweenPoints(const Point &pointOne, const Point &pointTwo)
{
    return sqrt(pow(pointOne.x - pointTwo.x, 2.0) + pow(pointOne.y - pointTwo.y, 2.0));
}

#pragma mark - Image creation

void AGOpenCVHelper::createEmptyMatWithSize(cv::Mat &image, const cv::Size &size, const int imageType)
{
    image = Mat::zeros(size.height, size.width, imageType);
}

void AGOpenCVHelper::createMaskMatWithSize(cv::Mat &image, const cv::Size &size)
{
    image = Mat::ones(size.height, size.width, CV_8U);
    image = Scalar(WHITE_PIXEL);
}

#pragma mark - Image data introspection

bool AGOpenCVHelper::isPixelWhiteInImage(cv::Mat &image, const cv::Point &pixelPosition, AGError &error)
{
    if (pixelPosition.x < 0 || pixelPosition.x >= image.size().width) {
        error = { true, "isPixelWhiteInImage: Invalid x position." }; return false;
    }
    if (pixelPosition.y < 0 || pixelPosition.y >= image.size().height) {
        error = { true, "isPixelWhiteInImage: Invalid y position." }; return false;
    }
    if ((int)image.data[image.step * pixelPosition.y + image.channels() * pixelPosition.x + 0] == WHITE_PIXEL) {
        return true;
    }
    return false;
}

int AGOpenCVHelper::pixelValueAtPointInImage(const cv::Mat &image, const cv::Point &pixelPosition, AGError &error)
{
    if (pixelPosition.x < 0 || pixelPosition.x >= image.size().width) {
        error = { true, "pixelValueAtPointInImage: Invalid x position." }; return 0;
    }
    if (pixelPosition.y < 0 || pixelPosition.y >= image.size().height) {
        error = { true, "pixelValueAtPointInImage: Invalid y position." }; return 0;
    }
    if (image.type() == CV_32F) {
        return static_cast<int>(image.at<float>(pixelPosition.y, pixelPosition.x));
    }
    else {
        return static_cast<int>(image.data[image.step * pixelPosition.y + image.channels() * pixelPosition.x + 0]);
    }
}

void AGOpenCVHelper::setPixelValueAtPointInImage(const cv::Mat &image,
                                                 const cv::Point &pixelPosition,
                                                 const int pixelValue,
                                                 AGError &error)
{
    if (pixelPosition.x < 0 || pixelPosition.x >= image.size().width) {
        error = { true, "setPixelValueAtPointInImage: Invalid x position." }; return;
    }
    if (pixelPosition.y < 0 || pixelPosition.y >= image.size().height) {
        error = { true, "setPixelValueAtPointInImage: Invalid y position." }; return;
    }
    if (pixelValue < 0 || pixelValue > 255) {
        error = { true, "setPixelValueAtPointInImage: Invalid pixel value." }; return;
    }
    image.data[image.step * pixelPosition.y + image.channels() * pixelPosition.x + 0] = pixelValue;
}

#pragma mark - Image transformations

void AGOpenCVHelper::createShiftMatrix(cv::Mat &shiftMatrix, const double dx, const double dy)
{
    shiftMatrix = (Mat_<double>(2,3) << 1, 0, dx, 0, 1, dy);
}

void AGOpenCVHelper::rotateImage(cv::Mat &image, const double angle)
{
    int len = max(image.cols, image.rows);
    Point2f pt(len * 0.5, len * 0.5);
    Mat r = getRotationMatrix2D(pt, angle, 1.0);
    warpAffine(image, image, r, Size(len, len));
}

void AGOpenCVHelper::insertSourceImageIntoOutputImageAtPoint(const cv::Mat &sourceImage,
                                                             cv::Mat &outputImage,
                                                             const cv::Point &point,
                                                             AGError &error)
{
    if (!sourceImage.data || !outputImage.data) {
        error = { true, "insertSourceImageIntoOutputImageAtPoint: Couldn't insert source image into output. Check if images have data." }; return;
    }
    sourceImage.copyTo(outputImage(Rect(point.x, point.y, sourceImage.cols, sourceImage.rows)));
}

void AGOpenCVHelper::linkTwoImagesTogether(const cv::Mat &imageOne,
                                           const cv::Mat &imageTwo,
                                           cv::Mat &outputImage,
                                           const ImageDirection &imageDirection,
                                           AGError &error)
{
    if (!imageOne.data || !imageTwo.data) {
        error = { true, "linkTwoImagesTogether: imageOne or imageTwo has no data." }; return;
    }
    
    int totalCols = 0;
    int totalRows = 0;

    if (imageDirection == Up || imageDirection == Down) {
        totalCols = (imageTwo.cols > imageOne.cols) ? imageTwo.cols : imageOne.cols;
        totalRows = imageOne.rows + imageTwo.rows;
    }
    else {
        totalCols = imageOne.cols + imageTwo.cols;
        totalRows = (imageTwo.rows > imageOne.rows) ? imageTwo.rows : imageOne.rows;
    }
    
    Mat totalImage(Size(totalCols, totalRows), imageOne.type());

    switch (imageDirection) {
        case Up:
            imageTwo.copyTo(totalImage(Rect(0, 0, imageTwo.cols, imageTwo.rows)));
            imageOne.copyTo(totalImage(Rect(0, imageTwo.rows, imageTwo.cols, imageTwo.rows)));
            break;

        case Down:
            imageOne.copyTo(totalImage(Rect(0, 0, imageOne.cols, imageOne.rows)));
            imageTwo.copyTo(totalImage(Rect(0, imageOne.rows, imageTwo.cols, imageTwo.rows)));
            break;

        case Left:
            imageTwo.copyTo(totalImage(Rect(0, 0, imageOne.cols, imageOne.rows)));
            imageOne.copyTo(totalImage(Rect(imageOne.cols, 0, imageTwo.cols, imageTwo.rows)));
            break;

        case Right:
            imageOne.copyTo(totalImage(Rect(0, 0, imageOne.cols, imageOne.rows)));
            imageTwo.copyTo(totalImage(Rect(imageOne.cols, 0, imageTwo.cols, imageTwo.rows)));
            break;
    }
    
    outputImage = totalImage;
}

void AGOpenCVHelper::linkTwoImagesTogetherAndDrawMatches(const AGImage &imageOne,
                                                         const AGImage &imageTwo,
                                                         const std::vector<cv::DMatch> &matches,
                                                         cv::Mat &outputImage,
                                                         const ImageDirection &imageDirection,
                                                         AGError &error)
{
    if (!imageOne.image.data || !imageTwo.image.data) {
        error = { true, "linkTwoImagesTogetherAndDrawMatches: imageOne or imageTwo has no data." }; return;
    }
    
    if (imageOne.keypoints.empty() || imageTwo.keypoints.empty()) {
        error = { true, "linkTwoImagesTogetherAndDrawMatches: imageOne.keypoints or imageTwo.keypoints are empty." };
        return;
    }
    
    Mat first = imageOne.image.clone();
    Mat second = imageTwo.image.clone();

    if (imageOne.image.type() == CV_8UC1 && imageTwo.image.type() == CV_8UC1) {
        AGOpenCVHelper::convertImageToColor(first);
        AGOpenCVHelper::convertImageToColor(second);
    }
    else if (imageOne.image.type() == CV_8UC4 && imageTwo.image.type() == CV_8UC4) {
        cvtColor(first, first, CV_RGBA2BGR);
        cvtColor(second, second, CV_RGBA2BGR);
    }

    for (int i = 0; i < imageOne.keypoints.size(); i++) {
        circle(first, imageOne.keypoints[i].pt, 3, Scalar(255, 0, 0));
    }
    for (int i = 0; i < imageTwo.keypoints.size(); i++) {
        circle(second, imageTwo.keypoints[i].pt, 3, Scalar(0, 255, 0));
    }

    AGError checkError = {false, ""};
    AGOpenCVHelper::linkTwoImagesTogether(first, second, outputImage, imageDirection, checkError);
    if (checkError.isError) {
        error = { true, "linkTwoImagesTogetherAndDrawMatches: drawMatches method failed inside with error: "
            + checkError.description };
        return;
    }
    
    switch (imageDirection) {
        case Up:
            for (int i = 0; i < matches.size(); i++) {
                Point pointTwo = imageTwo.keypoints[matches[i].trainIdx].pt;
                Point pointOne(imageOne.keypoints[matches[i].queryIdx].pt.x,
                               imageOne.keypoints[matches[i].queryIdx].pt.y + imageTwo.image.rows);
                line(outputImage, pointOne, pointTwo, Scalar(0, 0, 255));
            }
            break;

        case Down:
            for (int i = 0; i < matches.size(); i++) {
                Point pointOne = imageOne.keypoints[matches[i].queryIdx].pt;
                Point pointTwo(imageTwo.keypoints[matches[i].trainIdx].pt.x,
                               imageTwo.keypoints[matches[i].trainIdx].pt.y + imageOne.image.rows);
                line(outputImage, pointOne, pointTwo, Scalar(0, 0, 255));
            }
            break;

        case Left:
            for (int i = 0; i < matches.size(); i++) {
                Point pointTwo = imageTwo.keypoints[matches[i].trainIdx].pt;
                Point pointOne(imageOne.keypoints[matches[i].queryIdx].pt.x + imageTwo.image.cols,
                               imageOne.keypoints[matches[i].queryIdx].pt.y);
                line(outputImage, pointOne, pointTwo, Scalar(0, 0, 255));
            }
            break;

        case Right:
            for (int i = 0; i < matches.size(); i++) {
                Point pointOne = imageOne.keypoints[matches[i].queryIdx].pt;
                Point pointTwo(imageTwo.keypoints[matches[i].trainIdx].pt.x + imageOne.image.cols,
                               imageTwo.keypoints[matches[i].trainIdx].pt.y);
                line(outputImage, pointOne, pointTwo, Scalar(255, 0, 0));
            }
            break;
    }
}

#pragma mark - Saving and showing images

void AGOpenCVHelper::saveImage(cv::Mat &image,
                               const std::string &imageName,
                               const std::string &savePath,
                               AGError &error)
{
    if (imageName.empty() || savePath.empty()) {
        error = { true, "saveImage: imageName or savePath is empty." }; return;
    }
    if (!image.data) {
        error = { true, "saveImage: imageName or savePath is empty." }; return;
    }
    string path = savePath + "/" + imageName + ".png";
    if (image.type() == CV_8UC4) {
        cvtColor(image, image, CV_BGRA2BGR);
    }
    if (!imwrite(path, image)) {
        error = { true, "saveImage: Couldn't write image." }; return;
    }
}

void AGOpenCVHelper::saveKeypointsToFile(const AGImage &image,
                                         const std::string &fileName,
                                         const std::string &savePath,
                                         AGError &error)
{
    if (savePath.empty() || fileName.empty()) {
        error = { true, "saveKeypointsToFile: savePath or fileName is empty" }; return;
    }
    if (image.keypoints.empty()) {
        error = { true, "saveKeypointsToFile: image.keypoints are empty" }; return;
    }
    ofstream file;
    string path = savePath + "/" + fileName + ".txt";
    file.open(path.c_str());
    for (int i = 0; i < image.keypoints.size(); i++) {
        file << i << " x = " << to_string(image.keypoints[i].pt.x) << " y = "
        << to_string(image.keypoints[i].pt.y) << "\n";
    }
    file.close();
}

void AGOpenCVHelper::saveMatchesToFile(const std::vector<cv::DMatch> &matches,
                                       const std::string &fileName,
                                       const std::string &savePath,
                                       AGError &error)
{
    if (savePath.empty() || fileName.empty()) {
        error = { true, "saveMatchesToFile: savePath or fileName is empty" }; return;
    }
    if (matches.empty()) {
        error = { true, "saveMatchesToFile: matches are empty" }; return;
    }
    ofstream file;
    string path = savePath + "/" + fileName + ".txt";
    file.open(path.c_str());
    for (int i = 0; i < matches.size(); i++) {
        file << i << " one = " << to_string(matches[i].queryIdx) << " train = "
        << to_string(matches[i].trainIdx) << "\n";
    }
    file.close();
}

void AGOpenCVHelper::showImage(const cv::Mat &image, const std::string &windowName, AGError &error)
{
    if (windowName.empty()) {
        error = { true, "showImage: Window name is empty." }; return;
    }
    if (!image.data) {
        error = { true, "showImage: Image to show is empty." }; return;
    }
    imshow(windowName.c_str(), image);
    waitKey();
}

string AGOpenCVHelper::getDescriptionOfImage(const AGImage &image, AGError &error)
{
    if (image.name.empty()) {
        error = { true, "getDescriptionOfImage: Image name is empty." }; return "";
    }
    return "Image coordinates in mosaic x = " + to_string(image.xCoordinate) + " y = "
    + to_string(image.yCoordinate) + "; Image file name: " + image.name;
}

void AGOpenCVHelper::loadImage(Mat &image, const string &loadPath, AGError &error)
{
    if (loadPath.empty()) {
        error = { true, "loadImage: loadPath is empty." }; return;
    }
    image = imread(loadPath, CV_LOAD_IMAGE_COLOR);
    if(!image.data) {
        error = { true, "loadImage: Couldn't open or find the image." }; return;
    }
}
