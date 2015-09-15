//
//  Created by Aleksander Grzyb on 13/01/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#include "AGImageLoader.h"
#include "AGOpenCVHelper.h"
#include "../thirdparty/include/libconfig/libconfig.h++"

#include <unistd.h>
#include <sstream>

using namespace cv;
using namespace std;
using namespace libconfig;

AGImageLoader::AGImageLoader(const char *configFilePath, AGParameters &parameters, AGError &error)
{
    AGError checkError;
    loadConfigurationFile(configFilePath, checkError);
    if (checkError.isError) {
        error = { true, "AGImageLoader: Error occured while loading configuration file. Information about the error is above. " + checkError.description}; return;
    }
    parameters = this->parameters;
}

void AGImageLoader::loadConfigurationFile(const char *configFilePath, AGError &error)
{
    Config configuration;

    // Loading configuration file
    try {
        configuration.readFile(configFilePath);
    }
    catch (const FileIOException &fioex) {
        error = { true, "loadConfigurationFile: I/O error while reading configuration file." }; return;
    }
    catch(const ParseException &pex) {
        stringstream ss;
        ss << "loadConfigurationFile: Parse of configuration file error at " << pex.getFile() << ":" << pex.getLine() << " - " << pex.getError();
        error = { true, ss.str() }; return;
    }

    // Loading information from configuration file
    try {
        string mosaicsDirectoryAbsolutePath = configuration.lookup("mosaicsDirectoryAbsolutePath");
        this->parameters.mosaicsDirectoryAbsolutePath = mosaicsDirectoryAbsolutePath;
    }
    catch(const SettingNotFoundException &nfex) {
        error = { true, "loadConfigurationFile: No 'mosaicsDirectoryAbsolutePath' setting in configuration file." }; return;
    }

    try {
        string mosaicsSaveAbsolutePath = configuration.lookup("mosaicsSaveAbsolutePath");
        this->parameters.mosaicsSaveAbsolutePath = mosaicsSaveAbsolutePath;
    }
    catch(const SettingNotFoundException &nfex) {
        error = { true, "loadConfigurationFile: No 'mosaicsSaveAbsolutePath' setting in configuration file." }; return;
    }

    try {
        string tilesBaseName = configuration.lookup("tilesBaseName");
        this->tilesBaseName = tilesBaseName;
    }
    catch(const SettingNotFoundException &nfex) {
        error = { true, "loadConfigurationFile: No 'tilesBaseName' setting in configuration file." }; return;
    }
    
    try {
        this->parameters.numberOfMosaics = configuration.lookup("numberOfMosaics");
    }
    catch(const SettingNotFoundException &nfex) {
        error = { true, "loadConfigurationFile: No 'numberOfMosaics' setting in configuration file." }; return;
    }

//    try {
//        this->parameters.clustering = configuration.lookup("clustering");
//        cout << boolalpha;
//        cout << "Experimental clustering will be used: " << this->parameters.clustering << endl;
//    }
//    catch(const SettingNotFoundException &nfex) {
//        cerr << "No 'clustering' setting in configuration file." << endl;
//        return EXIT_FAILURE;
//    }
//
//    try {
//        this->parameters.isAdHoc = configuration.lookup("isAdHoc");
//        cout << boolalpha;
//        cout << "Ad-hoc will be used: " << this->parameters.isAdHoc << endl;
//    }
//    catch(const SettingNotFoundException &nfex) {
//        cerr << "No 'isAdHoc' setting in configuration file." << endl;
//        return EXIT_FAILURE;
//    }

    try {
        this->parameters.angleParameter = configuration.lookup("angleParameter");
    }
    catch(const SettingNotFoundException &nfex) {
        error = { true, "loadConfigurationFile: No 'angleParameter' setting in configuration file." }; return;
    }

//    try {
//        this->parameters.lengthParameter = configuration.lookup("lengthParameter");
//    }
//    catch(const SettingNotFoundException &nfex) {
//        error = { true, "loadConfigurationFile: No 'lengthParameter' setting in configuration file." }; return;
//    }

    try {
        this->parameters.shiftParameter = configuration.lookup("shiftParameter");
    }
    catch(const SettingNotFoundException &nfex) {
        error = { true, "loadConfigurationFile: No 'shiftParameter' setting in configuration file." }; return;
    }

    try {
        this->parameters.percentOverlap = configuration.lookup("percentOverlap");
    }
    catch(const SettingNotFoundException &nfex) {
        error = { true, "loadConfigurationFile: No 'percentOverlap' setting in configuration file." }; return;
    }
    
//
//    try {
//        this->parameters.simplerTransform = configuration.lookup("simplerTransform");
//        cout << boolalpha;
//        cout << "Simpler transform will be used: " << this->parameters.simplerTransform << endl;
//    }
//    catch(const SettingNotFoundException &nfex) {
//        cerr << "No 'simplerTransform' setting in configuration file." << endl;
//        return EXIT_FAILURE;
//    }
//    
//    try {
//        this->parameters.rigidTransform = configuration.lookup("rigidTransform");
//        cout << boolalpha;
//        cout << "Rigid transform will be used: " << this->parameters.rigidTransform << endl;
//    }
//    catch(const SettingNotFoundException &nfex) {
//        cerr << "No 'rigidTransform' setting in configuration file." << endl;
//        return EXIT_FAILURE;
//    }
//
//    try {
//        this->parameters.usePaths = configuration.lookup("usePaths");
//        cout << boolalpha;
//        cout << "Paths will be used: " << this->parameters.usePaths << endl;
//    }
//    catch(const SettingNotFoundException &nfex) {
//        cerr << "No 'usePaths' setting in configuration file." << endl;
//        return EXIT_FAILURE;
//    }
}

void AGImageLoader::loadTilesInMosaicNumber(std::vector<std::vector<AGImage>> &tiles, int mosaicNumber, AGError &error)
{
    // This method should always give images in left-right coordinate system
    // Right now images in folder are in left-bottom coordinate system
    // Additionally images need to be rotated (180 degrees) which is flipping horizontal then vertical
    
    if (mosaicNumber < 0) {
        error = { true, "loadTilesInMosaicNumber: Wrong mosaic number passed." }; return;
    }
    
    unsigned long xCoordinatePosition = this->tilesBaseName.find("_X");
    xCoordinatePosition += 2;

    for (int y = 0; ; ++y) {
        Mat image = imread(this->getTilePathAtPosition(0, y, mosaicNumber), CV_LOAD_IMAGE_GRAYSCALE);
        if (!image.data) {
            break;
        }
        else {
            vector<AGImage> nextColumn;
            tiles.push_back(nextColumn);
        }
    }
    
    for (int y = (int)tiles.size() - 1; y >= 0; --y) {
        for (int x = 0; ; ++x) {
            Mat image = imread(this->getTilePathAtPosition(x, y, mosaicNumber), CV_LOAD_IMAGE_GRAYSCALE);
            if (!image.data) {
                break;
            }
            else {
                AGOpenCVHelper::rotateImage(image, 180);
                AGImage imageInfo(image, x, (int)tiles.size() - y - 1, image.cols, image.rows, this->getTileNameAtPosition(x, y));
                tiles[x].push_back(imageInfo);
            }
        }
    }
    if (tiles.empty()) {
        error = { true, "loadTilesInMosaicNumber: There is no images to load. Check path and image name in configuration file." }; return;
    }
}

string AGImageLoader::getTilePathAtPosition(int x, int y, int mosaicNumber)
{
    if (x < 0 || y < 0) {
        return "";
    }

    string tilePath = this->parameters.mosaicsDirectoryAbsolutePath + "/" + "mosaic_" + to_string(mosaicNumber);

    tilePath += "/";
    tilePath += this->getTileNameAtPosition(x, y);

    return tilePath;
}

string AGImageLoader::getTileNameAtPosition(int x, int y)
{
    if (x < 0 || y < 0) {
        return "";
    }

    string tileName = this->tilesBaseName;

    unsigned long xCoordinatePosition = this->tilesBaseName.find("_X");
    xCoordinatePosition += 2;

    tileName.erase(xCoordinatePosition, 1);
    tileName.insert(xCoordinatePosition, to_string(x));

    unsigned long yCoordinatePosition = tileName.find("_Y");
    yCoordinatePosition += 2;

    tileName.erase(yCoordinatePosition, 1);
    tileName.insert(yCoordinatePosition, to_string(y));

    return tileName;
}





