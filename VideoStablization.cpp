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

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort( const std::string & error )
{
    std::cout << error << std::endl;
    abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

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

VideoStablizer::VideoStablizer( std::string path )
    : _path( path )
{

}

bool VideoStablizer::run(vector<string> arguments )
{
    cv::VideoCapture cap( _path );
    assert( cap.isOpened() );

    cv::Mat cur, cur_grey;
    cv::Mat prev, prev_grey;

    cap >> prev;//get the first frame.ch
    cv::cvtColor( prev, prev_grey, cv::COLOR_BGR2GRAY );
    
    /* copied from FaceLandmarkVid.cpp */
    // facial landmark tracking
    LandmarkDetector::FaceModelParameters det_parameters(arguments);
    LandmarkDetector::CLNF clnf_model(det_parameters.model_location);
    cv::Mat_<float> depth_image;

    det_parameters.track_gaze = false;
    /* end copy */
    
    // Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
    std::vector<TransformParam> prev_to_cur_transform; // previous to current

    int k = 1;
    int max_frames = cap.get( cv::CAP_PROP_FRAME_COUNT );
    cv::Mat last_T;

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
        
        cur.copyTo( prev );
        cur_grey.copyTo( prev_grey );

        k++;
    }

    // Step 2 - Accumulate the transformations to get the image trajectory
    // Accumulated frame to frame transform
    double a = 0;
    double x = 0;
    double y = 0;

    std::vector<Trajectory> trajectory; // trajectory at all frames
    for( size_t i = 0; i < prev_to_cur_transform.size(); i++ )
    {
        x += prev_to_cur_transform[i]._dx;
        y += prev_to_cur_transform[i]._dy;
        a += prev_to_cur_transform[i]._da;

        trajectory.push_back( Trajectory( x, y, a ) );
    }

    // Step 3 - Smooth out the trajectory using an averaging window
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

        smoothed_trajectory.push_back( Trajectory( avg_x, avg_y, avg_a ) );
    }

    // Step 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
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
    cap.set( cv::CAP_PROP_POS_FRAMES, 0 );
    cv::Mat T( 2, 3, CV_64F );

    const int vert_border = kHorizontalBorderCrop * prev.rows / prev.cols; // get the aspect ratio correct

    k = 0;
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

        cur2 = cur2( cv::Range( vert_border, cur2.rows - vert_border ),
                     cv::Range( kHorizontalBorderCrop, cur2.cols - kHorizontalBorderCrop ) );

        // Resize cur2 back to cur size, for better side by side comparison
        cv::resize( cur2, cur2, cur.size() );
        /* face feature detection */
        cv::cvtColor( cur, cur_grey, cv::COLOR_BGR2GRAY );
        LandmarkDetector::DetectLandmarksInVideo(cur_grey, depth_image, clnf_model, det_parameters);
        cur = visualise_tracking(cur, clnf_model, det_parameters);
        
        cv::cvtColor( cur2, cur_grey, cv::COLOR_BGR2GRAY );
        LandmarkDetector::DetectLandmarksInVideo(cur_grey, depth_image, clnf_model, det_parameters);
        cur2 = visualise_tracking(cur2, clnf_model, det_parameters);        
        /* end face feature detection */
        
        // Now draw the original and stablised side by side for coolness
        cv::Mat canvas = cv::Mat::zeros( cur.rows, cur.cols * 2 + 10, cur.type() );

        cur.copyTo( canvas( cv::Range::all(), cv::Range( 0, cur2.cols ) ) );
        cur2.copyTo( canvas( cv::Range::all(), cv::Range( cur2.cols + 10, cur2.cols * 2 + 10 ) ) );

        // If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
        //if( canvas.cols > 960 )
        //{
        //    cv::resize( canvas, canvas, cv::Size( canvas.cols / 4, canvas.rows / 4 ) );
        //}

        cv::imshow( "before and after", canvas );
        cv::waitKey( 400 );

        k++;
    }
    return 0;
}

int main (int argc, char **argv)
{
	vector<string> arguments = get_arguments(argc, argv);
	vector<string> files, depth_directories, output_video_files, out_dummy;
	bool u;
	LandmarkDetector::get_video_input_output_params(files, depth_directories, out_dummy, output_video_files, u, arguments);
	std::string input_path;
	for (int i=0;i<files.size();i++) {
	    input_path = files[i];
	    VideoStablizer vs(input_path);
	    vs.run(arguments);
	}
}

