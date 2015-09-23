//
//  Created by Aleksander Grzyb on 13/01/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#ifndef __Mosaic_Stitcher__AGImageLoader__
#define __Mosaic_Stitcher__AGImageLoader__

#include "AGDataStructures.h"

#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

 /// Responsible for loading configuration file and images.

class AGImageLoader {
public:
    
    /**
     *  Constructor of AGImageLoader object.
     *
     *  @param configFilePath Path to configuration file.
     *  @param parameters Parameters from configuration file.
     *  @param error Error.
     */
    AGImageLoader(const char *configFilePath, AGParameters &parameters, AGError &error);
    
    /**
     *  Loads images to tiles matrix.
     *
     *  @param tiles        Matrix of mosaic tiles to which images will be loaded.
     *  @param mosaicNumber Identifier of currently loaded mosaic.
     *  @param error        Error.
     */
    void loadTilesInMosaicNumber(std::vector<std::vector<AGImage>> &tiles, int mosaicNumber, AGError &error);
private:
    
    /**
     *  Loads configuration file.
     *
     *  @param configFilePath Path to configuration file.
     *  @param error          Error.
     */
    void loadConfigurationFile(const char *configFilePath, AGError &error);
    
    /**
     *  Returns path of tile image.
     *
     *  @param x            Coordinate of current tile (x axis).
     *  @param y            Coordinate of current tile (y axis).
     *  @param mosaicNumber Identifier of currently loaded mosaic.
     *
     *  @return File path to tile image.
     */
    std::string tilePathAtPosition(int x, int y, int mosaicNumber);
    
    /**
     *  Return tile image file name.
     *
     *  @param x Coordinate of current tile (x axis).
     *  @param y Coordinate of current tile (y axis).
     *
     *  @return File name of tile image.
     */
    std::string tileNameAtPosition(int x, int y);
    
    /**
     *  Base name of tiles from configuration file.
     */
    std::string tilesBaseName;
    
    /**
     *  Loaded parameters from configuration file.
     */
    AGParameters parameters;
};

#endif /* defined(__Mosaic_Stitcher__AGImageLoader__) */
