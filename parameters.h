#pragma once
#include <string>
#include <iostream>
#include <iomanip>

struct Parameters{
  std::string filename = "data/t.txt";
  int   patch_size=4;   // Patch size
  int   window_size=5; // Search window (size=window_size)
  int   step_size=3;
  float sim_th=2500.0;   // Similarity threshold for the first step
  int   maxN=16;          // Maximal number of the patches in one group
  float hard_th=2.7; // Hard schrinkage threshold
  float sigma=25.0;
  float noise_sigma=25.0;
  bool  add_noise_ = false;
  int   gpu_device=0;
};

void inline read_parameters(int argc, char *argv[], Parameters &p){
 using namespace std;
 if (argc==1){
   cout<<argv[0]<<" avi patch_size window_size sim_th maxN hard_th sigma"<<endl;
 }

 if(argc>=2) p.filename=argv[1];
 if(argc>=3) p.patch_size=atoi(argv[2]);
 if(argc>=4) p.window_size=atoi(argv[3]);
 if(argc>=5) p.sim_th=atof(argv[4]);
 if(argc>=6) p.maxN=atoi(argv[5]);
 if(argc>=7) p.hard_th=atof(argv[6]);
 if(argc>=8){ p.sigma=atof(argv[7]); }
 if(argc>=9){ p.noise_sigma=atof(argv[8]); }

 cout<<"Parameters:"<<endl;
 cout<<"         volume: "<<p.filename.c_str()<<endl;
 cout<<"     patch_size: "<<p.patch_size<<endl;
 cout<<"    window_size: "<<p.window_size<<endl;
 cout<<"         sim_th: "<<fixed<<setprecision(3)<<p.sim_th<<endl;
 cout<<"           maxN: "<<p.maxN<<endl;
 cout<<"        hard_th: "<<fixed<<setprecision(3)<<p.hard_th<<endl;
 cout<<"          sigma: "<<fixed<<setprecision(3)<<p.sigma<<endl;
 cout<<"    noise_sigma: "<<fixed<<setprecision(3)<<p.noise_sigma<<endl;
}
