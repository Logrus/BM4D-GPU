/*
 * 2016, Vladislav Tananaev
 * tananaev@cs.uni-freiburg.de
 */
#pragma once
#include <cstdlib>  // EXIT_SUCESS, EXIT_FAILURE
#include <iomanip>
#include <iostream>
#include <string>

struct Parameters {
  // Modifyable
  std::string filename = "data2/t.txt";
  std::string out_filename = "denoised.tiff";
  float sim_th = 2500.0;  // Similarity threshold for the first step
  float hard_th = 2.7;    // Hard schrinkage threshold

  // Can be changed but not advisable
  int window_size = 5;  // Search window, barely affects the results [Lebrun M., 2013]
  int step_size = 3;    // Reasonable values {1,2,3,4} Significantly (exponentially) affects speed,
                        // slightly affect results
  int gpu_device = -1;

  // Fixed in current implementation, changing this can lead to trouble
  int patch_size = 4;  // Patch size
  int maxN = 16;       // Maximal number of the patches in one group
};

void inline read_parameters(int argc, char* argv[], Parameters& p) {
  using namespace std;
  if (argc == 1) {
    cout << argv[0] << " input_file[tiff,avi] [sim_th] [hard_th]" << endl;
    // exit(EXIT_SUCESS);
  }

  if (argc >= 2) p.filename = argv[1];
  if (argc >= 3) p.out_filename = argv[2];
  if (argc >= 4) p.sim_th = atof(argv[3]);
  if (argc >= 5) p.hard_th = atof(argv[4]);

  if (argc >= 6) p.window_size = atoi(argv[5]);
  if (argc >= 7) p.step_size = atoi(argv[6]);

  if (argc >= 8) p.gpu_device = atoi(argv[7]);

  cout << "Parameters:" << endl;
  cout << "            input file: " << p.filename.c_str() << endl;
  cout << " similarity threshold: " << fixed << setprecision(3) << p.sim_th << endl;
  cout << "       hard threshold: " << fixed << setprecision(3) << p.hard_th << endl;
  cout << "          window size: " << p.window_size << endl;
  cout << "            step size: " << p.step_size << endl;
  cout << " max cubes in a group: " << p.maxN << endl;
  cout << "           patch size: " << p.patch_size << endl;
}
