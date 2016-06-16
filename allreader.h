#pragma once
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "CImg.h"
//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
using namespace cimg_library;


// This class should read several formats into 3D CImg
class AllReader{
private:
  bool display;

public:
  // Some obvious comment like: constructor
  inline AllReader(): display(false) {};
  inline AllReader(bool d): display(d) {};
  void read(const std::string &filename, std::vector<unsigned char> &volume, int &width, int &height, int &depth);
};