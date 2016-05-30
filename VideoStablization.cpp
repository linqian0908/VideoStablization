#include "LandmarkCoreIncludes.h"
#include "VideoStablization.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

using namespace std;

vector<string> get_arguments(int argc, char **argv)
{

        vector<string> arguments;

        for(int i = 0; i < argc; ++i)
        {
                arguments.push_back(string(argv[i]));
        }
        return arguments;
}

// Visualising the results
cv::Mat& visualise_tracking(cv::Mat& captured_image, const LandmarkDetector::CLNF& face_model, const LandmarkDetector::FaceModelParameters& det_parameters)
{

        // Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
        double detection_certainty = face_model.detection_certainty;
        bool detection_success = face_model.detection_success;

        double visualisation_boundary = 0.2;

        // Only draw if the reliability is reasonable, the value is slightly ad-hoc
        if (detection_certainty < visualisation_boundary)
        {
                LandmarkDetector::Draw(captured_image, face_model);

                double vis_certainty = detection_certainty;
                if (vis_certainty > 1)
                        vis_certainty = 1;
                if (vis_certainty < -1)
                        vis_certainty = -1;

                vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);
        }
        return captured_image;
}

VideoStablizer::VideoStablizer( std::string path, double salient )
    : _path( path ), SmoothRatio(salient)
{

}

bool VideoStablizer::run( std::string output_path, vector<string> arguments )
{
    cv::VideoCapture cap( _path );
    assert( cap.isOpened() );

    cv::Mat cur, cur_grey;
    cv::Mat prev, prev_grey;

    cap >> prev;//get the first frame.ch
    int image_width = prev.cols;
    int image_height = prev.rows;
    cv::cvtColor( prev, prev_grey, cv::COLOR_BGR2GRAY );
    int kHorizontalBorderCrop = int(image_width*kHorizontalCropRatio);  // Crops the border to reduce missing pixels.
    int kVerticalBorderCrop = int(image_height*kVertialCropRatio);
    
    bool use_salient = (SmoothRatio>0.01);
    if (use_salient) {
        std::cout << "Using facial landmark with weight " << SmoothRatio << std::endl;
    } else {
        std::cout << "Not using facial landmark." << std::endl;
    }
    /* copied from FaceLandmarkVid.cpp */
    // facial landmark tracking
 
    LandmarkDetector::FaceModelParameters det_parameters(arguments);
    LandmarkDetector::CLNF clnf_model(det_parameters.model_location);
    cv::Mat_<float> depth_image;
    cv::Point2d frame_center(image_width/2.0,image_height/2.0);
    det_parameters.track_gaze = false;
    
    /* end copy */
    
    // Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
    std::vector<TransformParam> prev_to_cur_transform; // previous to current
    std::vector< vector<cv::Point2d> > landmarks; // facial landmarks for each frame
    
    int k = 0;
    int max_frames = cap.get( cv::CAP_PROP_FRAME_COUNT );
    cv::Mat last_T;
    
    
    std::cout << "Image width: " << image_width << std::endl;
    std::cout << "Image height: " << image_height <<std::endl;    
    std::cout << "Crop width: " << kHorizontalBorderCrop << std::endl;
    std::cout << "Crop height: " << kVerticalBorderCrop <<std::endl;
    
    std::cout << " Step 1: get frame transformation and salient points" << std::endl;
    while( true )
    {
        cap >> cur;
        if( cur.data == NULL ) { break; }
		
        cv::cvtColor( cur, cur_grey, cv::COLOR_BGR2GRAY );

        // vector from prev to cur
        std::vector<cv::Point2f> prev_corner, cur_corner;
        std::vector<cv::Point2f> prev_corner2, cur_corner2;
        std::vector<uchar> status;
        std::vector<float> err;

        cv::goodFeaturesToTrack( prev_grey, prev_corner, 200, 0.01, 30 );
        cv::calcOpticalFlowPyrLK( prev_grey, cur_grey, prev_corner, cur_corner, status, err );

        // weed out bad matches
        for( size_t i = 0; i < status.size(); i++ )
        {
            if( status[i] )
            {
                prev_corner2.push_back( prev_corner[i] );
                cur_corner2.push_back( cur_corner[i] );
            }
        }

        // translation + rotation only, rigid transform and no scaling/shearing
        cv::Mat T = cv::estimateRigidTransform( prev_corner2, cur_corner2, false );

        // in rare cases no transform is found. We'll just use the last known good transform.
        if( T.data == NULL ) { last_T.copyTo( T ); }
        T.copyTo( last_T );

        // decompose T
        double dx = T.at<double>( 0, 2 );
        double dy = T.at<double>( 1, 2 );
        double da = atan2( T.at<double>( 1, 0 ), T.at<double>( 0, 0 ) );

        prev_to_cur_transform.push_back( TransformParam( dx, dy, da ) );
        if (use_salient) {
            LandmarkDetector::DetectLandmarksInVideo(cur_grey, depth_image, clnf_model, det_parameters);
            landmarks.push_back(LandmarkDetector::CalculateLandmarks(clnf_model));
        }
        
        cur.copyTo( prev );
        cur_grey.copyTo( prev_grey );

        k++;
    }

    // Step 2 - Accumulate the transformations to get the image trajectory
    // Accumulated frame to frame transform
    std::cout << " Step 2: accumulate path and average salient features" << std::endl;
    double a = 0;
    double x = 0;
    double y = 0;
    
    std::vector<Trajectory> trajectory; // trajectory at all frames

    cv::Point2d point_zero(0.0,0.0);
    cv::Point2d point_sum;
    std::vector<cv::Point2d> landmarks_avg;
    
    for( size_t i = 0; i < prev_to_cur_transform.size(); i++ )
    {
        x += prev_to_cur_transform[i]._dx;
        y += prev_to_cur_transform[i]._dy;
        a += prev_to_cur_transform[i]._da;

        trajectory.push_back( Trajectory( x, y, a ) );
        
        if (use_salient) {
            point_sum = std::accumulate(landmarks[i].begin(),landmarks[i].end(),point_zero);
            landmarks_avg.push_back(point_sum*(1.0/landmarks[i].size()));
        }
    }

    // Step 3 - Smooth out the trajectory using an averaging window
    std::cout << " Step 3: trajectory smoothing" << std::endl;
    std::vector<Trajectory> smoothed_trajectory; // trajectory at all frames

    for( size_t i = 0; i < trajectory.size(); i++ )
    {
        double sum_x = 0;
        double sum_y = 0;
        double sum_a = 0;
        int count = 0;

        for( int j = -kSmoothingRadius; j <= kSmoothingRadius; j++ )
        {
            if( i + j >= 0 && i + j < trajectory.size() )
            {
                sum_x += trajectory[i + j]._x;
                sum_y += trajectory[i + j]._y;
                sum_a += trajectory[i + j]._a;

                count++;
            }
        }

        double avg_a = sum_a / count;
        double avg_x = sum_x / count;
        double avg_y = sum_y / count;
        
        if (use_salient) {
            cv::Point2d avg_fxy(0.0,0.0);
            count = 0;
            sum_x = 0;
            sum_y = 0;
            for (int j=-fSmoothingRadius; j<= fSmoothingRadius; j++) {
                if (i+j>=0 && i+j<landmarks_avg.size()) {
                    sum_x += trajectory[i+j]._x;
                    sum_y += trajectory[i+j]._y;
                    avg_fxy += landmarks_avg[i+j];
                    count++;
                }
            }
            
            sum_x /= count;
            sum_y /= count;
            avg_fxy *= (1.0/count);
            avg_fxy -= frame_center;
            
            avg_x = avg_x*(1-SmoothRatio) + (sum_x-avg_fxy.x)*SmoothRatio;
            avg_y = avg_y*(1-SmoothRatio) + (sum_y-avg_fxy.y)*SmoothRatio;
        }
        smoothed_trajectory.push_back( Trajectory( avg_x, avg_y, avg_a ) );
    }

    // Step 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
    std::cout << " Step 4: generate new transformations" << std::endl;
    std::vector<TransformParam> new_prev_to_cur_transform;

    // Accumulated frame to frame transform
    a = 0;
    x = 0;
    y = 0;

    for( size_t i = 0; i < prev_to_cur_transform.size(); i++ )
    {
        x += prev_to_cur_transform[i]._dx;
        y += prev_to_cur_transform[i]._dy;
        a += prev_to_cur_transform[i]._da;

        // target - current
        double diff_x = smoothed_trajectory[i]._x - x;
        double diff_y = smoothed_trajectory[i]._y - y;
        double diff_a = smoothed_trajectory[i]._a - a;

        double dx = prev_to_cur_transform[i]._dx + diff_x;
        double dy = prev_to_cur_transform[i]._dy + diff_y;
        double da = prev_to_cur_transform[i]._da + diff_a;

        new_prev_to_cur_transform.push_back( TransformParam( dx, dy, da ) );

    }

    // Step 5 - Apply the new transformation to the video    
    std::cout << " Step 5: applied transformation to video" << std::endl;
    cap.set( cv::CAP_PROP_POS_FRAMES, 0 );
    cv::Mat T( 2, 3, CV_64F );

    k = 0;
    cv::VideoWriter outputVideo(
        output_path ,
        cap.get( CV_CAP_PROP_FOURCC ),
        cap.get( CV_CAP_PROP_FPS ),
        cv::Size( cap.get( CV_CAP_PROP_FRAME_WIDTH ),
                  cap.get( CV_CAP_PROP_FRAME_HEIGHT ) ) );

    if( !outputVideo.isOpened() )
    {
        std::cout  << "Could not open the output video for write: " << std::endl;
        return -1;
    }

    while( k < max_frames - 1 ) // don't process the very last frame, no valid transform
    {
        cap >> cur;
        if( cur.data == NULL ) { break; }

        T.at<double>( 0, 0 ) =  cos( new_prev_to_cur_transform[k]._da );
        T.at<double>( 0, 1 ) = -sin( new_prev_to_cur_transform[k]._da );
        T.at<double>( 1, 0 ) =  sin( new_prev_to_cur_transform[k]._da );
        T.at<double>( 1, 1 ) =  cos( new_prev_to_cur_transform[k]._da );

        T.at<double>( 0, 2 ) = new_prev_to_cur_transform[k]._dx;
        T.at<double>( 1, 2 ) = new_prev_to_cur_transform[k]._dy;

        cv::Mat cur2;

        cv::warpAffine( cur, cur2, T, cur.size() );

        cur2 = cur2( cv::Range( kVerticalBorderCrop, cur2.rows - kVerticalBorderCrop ),
                     cv::Range( kHorizontalBorderCrop, cur2.cols - kHorizontalBorderCrop ) );

        // Resize cur2 back to cur size, for better side by side comparison
        cv::resize( cur2, cur2, cur.size() );
        
        /* face feature detection */
        if (use_salient) {
            cv::cvtColor( cur, cur_grey, cv::COLOR_BGR2GRAY );
            LandmarkDetector::DetectLandmarksInVideo(cur_grey, depth_image, clnf_model, det_parameters);
            cur = visualise_tracking(cur, clnf_model, det_parameters);
            
            //cv::cvtColor( cur2, cur_grey, cv::COLOR_BGR2GRAY );
            //LandmarkDetector::DetectLandmarksInVideo(cur_grey, depth_image, clnf_model, det_parameters);
            //cur2 = visualise_tracking(cur2, clnf_model, det_parameters); 
        }       
        /* end face feature detection */
        
        // Now draw the original and stablised side by side for coolness
        cv::Mat canvas = cv::Mat::zeros( cur.rows, cur.cols * 2 + 10, cur.type() );

        cur.copyTo( canvas( cv::Range::all(), cv::Range( 0, cur2.cols ) ) );
        cur2.copyTo( canvas( cv::Range::all(), cv::Range( cur2.cols + 10, cur2.cols * 2 + 10 ) ) );
        outputVideo << cur2;
        
        // If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
        //if( canvas.cols > 960 )
        //{
        //    cv::resize( canvas, canvas, cv::Size( canvas.cols / 4, canvas.rows / 4 ) );
        //}

        cv::imshow( "before and after", canvas );
        cv::waitKey( 3 );

        k++;
    }
    return 0;
}

int main (int argc, char **argv)
{
	vector<string> arguments = get_arguments(argc, argv);
	double SmoothRatio = 0; // SmoothRatio=1 tries to keep face at center; 0 uses pure path smoothing
	for (size_t i=0;i<arguments.size();i++) {
	    if (arguments[i].compare("-salient")==0) {
	        stringstream data(arguments[i+1]);
	        data >> SmoothRatio;
			break;	        
	    }
	}
	vector<string> files, depth_directories, output_video_files, out_dummy;
	bool u;
	LandmarkDetector::get_video_input_output_params(files, depth_directories, out_dummy, output_video_files, u, arguments);
	std::string input_path, output_path;
	for (size_t i=0;i<files.size();i++) {
	    input_path = files[i];
	    output_path = input_path.substr(0,input_path.length()-4)+"_stable.avi";
	    std::cout<<output_path<<std::endl;    
	    VideoStablizer vs(input_path,SmoothRatio);
	    vs.run(output_path,arguments);
	}
}

