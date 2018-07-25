/*
 * 2016, Vladislav Tananaev
 * v.d.tananaev [at] gmail [dot] com
 */
#pragma once
#include <cstdlib>  // EXIT_SUCESS, EXIT_FAILURE
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/** @brief AllReader class
 *  allows to read videos, sequences ... PGM
 *
 */
class AllReader {
 public:
  AllReader(bool d) : display(d) {}
  AllReader() : AllReader(false) {}

  void read(const std::string& filename, std::vector<unsigned char>& volume, int& width,
            int& height, int& depth);
  void save(const std::string& filename, const std::vector<unsigned char>& volume, int width,
            int height, int depth);

  void readSequence(const std::string& filename, std::vector<unsigned char>& volume, int& width,
                    int& height, int& depth);
  void saveSequence(const std::string& filename, const std::vector<unsigned char>& volume,
                    int width, int height, int depth);

  void readTIFF(const std::string& filename, std::vector<unsigned char>& volume, int& width,
                int& height, int& depth);
  void saveTIFF(const std::string& filename, const std::vector<unsigned char>& volume, int width,
                int height, int depth);

  void readVideo(const std::string& filename, std::vector<unsigned char>& volume, int& width,
                 int& height, int& depth);
  void saveVideo(const std::string& filename, const std::vector<unsigned char>& volume, int width,
                 int height, int depth);

 private:
  bool display;
};
