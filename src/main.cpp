//
//  Created by Aleksander Grzyb on 13/01/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#include "../thirdparty/include/opencv2/core/core.hpp"
#include "../thirdparty/include/opencv2/highgui/highgui.hpp"
#include "AGMosaicStitcher.h"
#include "AGImageLoader.h"
#include "AGOpenCVHelper.h"

#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

void createMosaic(vector<vector<AGImage>> &imagesMatrix, const AGParameters &parameters, const string &versionName)
{
    Mat outputImage;
    AGMosaicStitcher mosaicStitcher = AGMosaicStitcher(parameters);
    mosaicStitcher.stitchMosaic(imagesMatrix, outputImage);
    if (outputImage.data) {
        AGError error;
        AGOpenCVHelper::saveImage(outputImage, versionName, parameters.mosaicsSaveAbsolutePath, error);
        if (error.isError) {
            cout << "createMosaic: " << error.description << endl;
        }
    }
}

void modifyAbsoultePathForDataSetNumber(AGParameters &parameters, int dataSetNumber)
{
//    size_t positionOfSlash = parameters.imagesDirectoryAbsolutePath.find_last_of("/");
//    string dataSetPath = parameters.imagesDirectoryAbsolutePath.substr(0, positionOfSlash) + "/" + "data_set_" + to_string(dataSetNumber);
//    parameters.imagesDirectoryAbsolutePath = dataSetPath;
}

int main(int argc, const char *argv[])
{
    bool testMode = false;
    if (argc > 1) {
        AGParameters parameters;
        AGError error;
        AGImageLoader imageLoader = AGImageLoader(argv[1], parameters, error);
        if (error.isError) {
            cout << error.description << endl; return EXIT_FAILURE;
        }
        if (testMode) {
            int testMosaic = 4;
            vector<vector<AGImage>> imagesMatrix;
            modifyAbsoultePathForDataSetNumber(parameters, testMosaic);
            imageLoader.loadTilesInMosaicNumber(imagesMatrix, testMosaic, error);
            if (error.isError) {
                cout << error.description << endl; return EXIT_FAILURE;
            }
            parameters.simplerTransform = false; parameters.rigidTransform = false; parameters.usePaths = true;
            createMosaic(imagesMatrix, parameters, "mosaic_" + to_string(testMosaic) + "_version_1");
        }
        else {
            for (int i = 1; i <= parameters.numberOfMosaics; ++i) {
                vector<vector<AGImage>> imagesMatrix;
                modifyAbsoultePathForDataSetNumber(parameters, i);
                imageLoader.loadTilesInMosaicNumber(imagesMatrix, i, error);
                if (error.isError) {
                    cout << error.description << endl; return EXIT_FAILURE;
                }
                
                // Version 1
                // simplerTransform = false;
                // rigidTransform = true;
                // usePaths = false;
                parameters.simplerTransform = false; parameters.rigidTransform = true; parameters.usePaths = false;
                createMosaic(imagesMatrix, parameters, "mosaic_" + to_string(i) + "_version_1");
                imagesMatrix.clear();
                
                // Version 2
                // simplerTransform = true;
                // rigidTransform = true;
                // usePaths = true;
                imageLoader.loadTilesInMosaicNumber(imagesMatrix, i, error);
                parameters.simplerTransform = true; parameters.rigidTransform = true; parameters.usePaths = true;
                createMosaic(imagesMatrix, parameters, "mosaic_" + to_string(i) + "_version_2");
                imagesMatrix.clear();
                
                // Version 3
                // simplerTransform = true;
                // rigidTransform = true;
                // usePaths = false;
                imageLoader.loadTilesInMosaicNumber(imagesMatrix, i, error);
                parameters.simplerTransform = true; parameters.rigidTransform = true; parameters.usePaths = false;
                createMosaic(imagesMatrix, parameters, "mosaic_" + to_string(i) + "_version_3");
                imagesMatrix.clear();
                
                // Version 4
                // simplerTransform = true;
                // rigidTransform = false;
                // usePaths = false;
                imageLoader.loadTilesInMosaicNumber(imagesMatrix, i, error);
                parameters.simplerTransform = false; parameters.rigidTransform = false; parameters.usePaths = false;
                createMosaic(imagesMatrix, parameters, "mosaic_" + to_string(i) + "_version_4");
            }
        }
    } else {
        cout << "Please provide configuration file (.cfg)." << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
