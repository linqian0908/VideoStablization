#define private   public
#define protected public
#include "videostab.h"

int main(){
    std::string input_path = "/home/linqian/Desktop/videostabilization/video_data/SANY0025.avi";
    std::string output_path = "/home/linqian/Desktop/videostabilization/video_data/SANY0025_warp.avi";

    VideoStablizer vs( input_path );
    vs.run( output_path );
}
