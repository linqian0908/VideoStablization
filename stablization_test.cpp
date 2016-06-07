#define private   public
#define protected public
#include "videostab.h"

int main(){
    std::string input_path = "/home/linqian/Desktop/videostabilization/video_data/new_gleicher.mp4";
    std::string output_path = "/home/linqian/Desktop/videostabilization/video_data/new_gleicher_warp.avi";

    VideoStablizer vs( input_path );
    vs.run( output_path );
}
