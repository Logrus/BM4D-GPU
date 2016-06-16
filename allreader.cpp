#include "allreader.h"
#ifndef idx3
#define idx3(x,y,z,x_size,y_size) ((x) + ((y)+(y_size)*(z))*(x_size))
#endif

bool readPGM(const std::string &filename,
	std::vector<unsigned char> &image, int &width, int &height)
{
	std::ifstream File(filename.c_str(), std::ifstream::binary);
	int length, maxval;
	// Get size
	File.seekg(0, File.end); length = File.tellg(); File.seekg(0, File.beg);
	std::string dummy;
	File >> dummy >> width >> height >> maxval;
	File.get(); // Remove all excessive spaces
	image.resize(width*height); // (Re)Initialize image from sizes
	File.read(reinterpret_cast<char*>(image.data()), length);
	return true;
}

void AllReader::read(const std::string &filename, std::vector<unsigned char> &volume, int &width, int &height, int &depth){
	std::vector<std::string> input;
	std::string inputDir;
	// Determine input directory
	std::string s = filename;
	if (s.find_last_of("/") != -1)
		s.erase(s.find_last_of("/") + 1, s.length());
	else
		s.erase(s.find_last_of("\\") + 1, s.length());
	inputDir = s;

	s = filename;
	s.erase(0, s.find_last_of('.'));
	
	if (s == ".txt" || s == ".TXT") {
		std::ifstream aStream(filename.c_str());
		aStream >> depth;
		input.resize(depth);
		for (int i = 0; i < depth; i++) {
			std::string s;
			aStream >> s;
			input.at(i) = inputDir + s;
		}
	}
	else {
		std::cout << "Must pass a txt file as input" << std::endl;
		exit(1);
	}
	// First image 
	std::vector<unsigned char> image;
	readPGM(input.at(0).c_str(), image, width, height);
	volume.reserve(width*height*depth);
	volume.insert(volume.end(), image.begin(), image.end());
	// the rest
	for (int i = 1; i < input.size(); ++i){
		readPGM(input.at(i).c_str(), image, width, height);
		volume.insert(volume.end(), image.begin(), image.end());
	}
  //// Check if this is TIFF, AVI or txt with filenames of imeages
  //std::cout<<"Loading "<<filename<<std::endl;
  //cv::VideoCapture capture(filename);

  //width  = capture.get(CV_CAP_PROP_FRAME_WIDTH);
  //height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
  //depth  = capture.get(CV_CAP_PROP_FRAME_COUNT);

  //volume.resize(width*height*depth);

  //cv::Mat frame;

  //if( !capture.isOpened() )
  //  throw "Error when reading file";

  //for(int slice=0;;slice++){

  //  if (!capture.read(frame)) break;

  //  for (int y=0;y<height;++y)
  //    for (int x=0;x<width;++x)
  //      volume[idx3(x,y,slice,width,height)] = frame.at<unsigned char>(y,x);

  //  if (this->display){
  //    cv::imshow("window", frame);
  //    char key = cvWaitKey(10);
  //    if (key == 27) break; // ESC
  //  }


  //}

}