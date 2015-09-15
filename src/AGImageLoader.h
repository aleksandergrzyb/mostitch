//
//  Created by Aleksander Grzyb on 13/01/15.
//  Copyright (c) 2015 Aleksander Grzyb. All rights reserved.
//

#ifndef __Mosaic_Stitcher__AGImageLoader__
#define __Mosaic_Stitcher__AGImageLoader__

#include "../thirdparty/include/opencv2/highgui/highgui.hpp"
#include "../thirdparty/include/opencv2/core/core.hpp"
#include "AGDataStructures.h"

#include <stdio.h>
#include <string>
#include <vector>

class AGImageLoader {
public:
    AGImageLoader(const char *configFilePath, AGParameters &parameters, AGError &error);
    void loadTilesInMosaicNumber(std::vector<std::vector<AGImage>> &tiles, int mosaicNumber, AGError &error);
private:
    void loadConfigurationFile(const char *configFilePath, AGError &error);
    std::string getTilePathAtPosition(int x, int y, int mosaicNumber);
    std::string getTileNameAtPosition(int x, int y);
    std::string tilesBaseName;
    AGParameters parameters;
};

#endif /* defined(__Mosaic_Stitcher__AGImageLoader__) */
