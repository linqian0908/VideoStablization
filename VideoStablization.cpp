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
// #include "cvplot.h"

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

using namespace std;

namespace CvPlot
{

//  use anonymous namespace to hide global variables.
namespace
{
        const CvScalar CV_BLACK = CV_RGB(0,0,0);
        const CvScalar CV_WHITE = CV_RGB(255,255,255);
        const CvScalar CV_GREY = CV_RGB(150,150,150);

        PlotManager pm;
}


Series::Series(void)
{
        data = NULL;
        label = "";
        Clear();
}

Series::Series(const Series& s):count(s.count), label(s.label), auto_color(s.auto_color), color(s.color)
{
        data = new float[count];
        memcpy(data, s.data, count * sizeof(float));
}


Series::~Series(void)
{
        Clear();
}

void Series::Clear(void)
{
        if (data != NULL)
                delete [] data;
        data = NULL;

        count = 0;
        color = CV_BLACK;
        auto_color = true;
        label = "";
}

void Series::SetData(int n, float *p)
{
        Clear();

        count = n;
        data = p;
}

void Series::SetColor(int R, int G, int B, bool auto_color)
{
        R = R > 0 ? R : 0;
        G = G > 0 ? G : 0;
        B = B > 0 ? B : 0;
        color = CV_RGB(R, G, B);
        SetColor(color, auto_color);
}

void Series::SetColor(CvScalar color, bool auto_color)
{
        this->color = color;
        this->auto_color = auto_color;
}

Figure::Figure(const string name)
{
        figure_name = name;

        custom_range_y = false;
        custom_range_x = false;
        backgroud_color = CV_WHITE;
        axis_color = CV_BLACK;
        text_color = CV_BLACK;

        figure_size = cvSize(600, 200);
        border_size = 30;

        plots.reserve(10);
}

Figure::~Figure(void)
{
}

string Figure::GetFigureName(void)
{
        return figure_name;
}

Series* Figure::Add(const Series &s)
{
        plots.push_back(s);
        return &(plots.back());
}

void Figure::Clear()
{
      plots.clear();
}

void Figure::Initialize()
{
        color_index = 0;

        // size of the figure
        if (figure_size.width <= border_size * 2 + 100)
                figure_size.width = border_size * 2 + 100;
        if (figure_size.height <= border_size * 2 + 200)
                figure_size.height = border_size * 2 + 200;

        y_max = FLT_MIN;
        y_min = FLT_MAX;

        x_max = 0;
        x_min = 0;

        // find maximum/minimum of axes
        for (vector<Series>::iterator iter = plots.begin();
                iter != plots.end();
                iter++)
        {
                float *p = iter->data;
                for (int i=0; i < iter->count; i++)
                {
                        float v = p[i];
                        if (v < y_min)
                                y_min = v;
                        if (v > y_max)
                                y_max = v;
                }

                if (x_max < iter->count)
                        x_max = iter->count;
        }

        // calculate zoom scale
        // set to 2 if y range is too small
        float y_range = y_max - y_min;
        float eps = 1e-9f;
        if (y_range <= eps)
        {
                y_range = 1;
                y_min = y_max / 2;
                y_max = y_max * 3 / 2;
        }

        x_scale = 1.0f;
        if (x_max - x_min > 1)
                x_scale = (float)(figure_size.width - border_size * 2) / (x_max - x_min);

        y_scale = (float)(figure_size.height - border_size * 2) / y_range;
}

CvScalar Figure::GetAutoColor()
{
        // 	change color for each curve.
        CvScalar col;

        switch (color_index)
        {
        case 1:
                col = CV_RGB(60,60,255);	// light-blue
                break;
        case 2:
                col = CV_RGB(60,255,60);	// light-green
                break;
        case 3:
                col = CV_RGB(255,60,40);	// light-red
                break;
        case 4:
                col = CV_RGB(0,210,210);	// blue-green
                break;
        case 5:
                col = CV_RGB(180,210,0);	// red-green
                break;
        case 6:
                col = CV_RGB(210,0,180);	// red-blue
                break;
        case 7:
                col = CV_RGB(0,0,185);		// dark-blue
                break;
        case 8:
                col = CV_RGB(0,185,0);		// dark-green
                break;
        case 9:
                col = CV_RGB(185,0,0);		// dark-red
                break;
        default:
                col =  CV_RGB(200,200,200);	// grey
                color_index = 0;
        }
        color_index++;
        return col;
}

void Figure::DrawAxis(IplImage *output)
{
        int bs = border_size;
        int h = figure_size.height;
        int w = figure_size.width;

        // size of graph
        int gh = h - bs * 2;
        int gw = w - bs * 2;

        // draw the horizontal and vertical axis
        // let x, y axies cross at zero if possible.
        float y_ref = y_min;
        if ((y_max > 0) && (y_min <= 0))
                y_ref = 0;

        int x_axis_pos = h - bs - cvRound((y_ref - y_min) * y_scale);

        cvLine(output, cvPoint(bs,     x_axis_pos),
                           cvPoint(w - bs, x_axis_pos),
                                   axis_color);
        cvLine(output, cvPoint(bs, h - bs),
                           cvPoint(bs, h - bs - gh),
                                   axis_color);

        // Write the scale of the y axis
        CvFont font;
        cvInitFont(&font,CV_FONT_HERSHEY_PLAIN,0.55,0.7, 0,1,CV_AA);

        int chw = 6, chh = 10;
        char text[16];

        // y max
        if ((y_max - y_ref) > 0.05 * (y_max - y_min))
        {
                snprintf(text, sizeof(text)-1, "%.1f", y_max);
                cvPutText(output, text, cvPoint(bs / 5, bs - chh / 2), &font, text_color);
        }
        // y min
        if ((y_ref - y_min) > 0.05 * (y_max - y_min))
        {
                snprintf(text, sizeof(text)-1, "%.1f", y_min);
                cvPutText(output, text, cvPoint(bs / 5, h - bs + chh), &font, text_color);
        }

        // x axis
        snprintf(text, sizeof(text)-1, "%.1f", y_ref);
        cvPutText(output, text, cvPoint(bs / 5, x_axis_pos + chh / 2), &font, text_color);

        // Write the scale of the x axis
        snprintf(text, sizeof(text)-1, "%.0f", x_max );
        cvPutText(output, text, cvPoint(w - bs - strlen(text) * chw, x_axis_pos + chh),
                      &font, text_color);

        // x min
        snprintf(text, sizeof(text)-1, "%.0f", x_min );
        cvPutText(output, text, cvPoint(bs, x_axis_pos + chh),
                      &font, text_color);


}
void Figure::DrawPlots(IplImage *output)
{
        int bs = border_size;
        int h = figure_size.height;
        int w = figure_size.width;

        // draw the curves
        for (vector<Series>::iterator iter = plots.begin();
                iter != plots.end();
                iter++)
        {
                float *p = iter->data;

                // automatically change curve color
                if (iter->auto_color == true)
                        iter->SetColor(GetAutoColor());

                CvPoint prev_point;
                for (int i=0; i<iter->count; i++)
                {
                        int y = cvRound(( p[i] - y_min) * y_scale);
                        int x = cvRound((   i  - x_min) * x_scale);
                        CvPoint next_point = cvPoint(bs + x, h - (bs + y));
                        cvCircle(output, next_point, 1, iter->color, 1);

                        // draw a line between two points
                        if (i >= 1)
                                cvLine(output, prev_point, next_point, iter->color, 1, CV_AA);
                        prev_point = next_point;
                }
        }

}

void Figure::DrawLabels(IplImage *output, int posx, int posy)
{

        CvFont font;
        cvInitFont(&font,CV_FONT_HERSHEY_PLAIN,0.55,1.0, 0,1,CV_AA);

        // character size
        int chw = 6, chh = 8;

        for (vector<Series>::iterator iter = plots.begin();
                iter != plots.end();
                iter++)
        {
                string lbl = iter->label;
                // draw label if one is available
                if (lbl.length() > 0)
                {
                        cvLine(output, cvPoint(posx, posy - chh / 2), cvPoint(posx + 15, posy - chh / 2),
                                   iter->color, 2, CV_AA);

                        cvPutText(output, lbl.c_str(), cvPoint(posx + 20, posy),
                                          &font, iter->color);

                        posy += int(chh * 1.5);
                }
        }

}

// whole process of draw a figure.
void Figure::Show()
{
        Initialize();

        IplImage *output = cvCreateImage(figure_size, IPL_DEPTH_8U, 3);
        cvSet(output, backgroud_color, 0);

        DrawAxis(output);

        DrawPlots(output);

        DrawLabels(output, figure_size.width - 100, 10);

        cvShowImage(figure_name.c_str(), output);
        cvWaitKey(1);
        cvReleaseImage(&output);

}



bool PlotManager::HasFigure(string wnd)
{
        return false;
}

// search a named window, return null if not found.
Figure* PlotManager::FindFigure(string wnd)
{
        for(vector<Figure>::iterator iter = figure_list.begin();
                iter != figure_list.end();
                iter++)
        {
                if (iter->GetFigureName() == wnd)
                        return &(*iter);
        }
        return NULL;
}

// plot a new curve, if a figure of the specified figure name already exists,
// the curve will be plot on that figure; if not, a new figure will be created.
void PlotManager::Plot(const string figure_name, const float *p, int count, int step,
                                           int R, int G, int B)
{
        if (count < 1)
                return;

        if (step <= 0)
                step = 1;

        // copy data and create a series format.
        float *data_copy = new float[count];

        for (int i = 0; i < count; i++)
                *(data_copy + i) = *(p + step * i);

        Series s;
        s.SetData(count, data_copy);

        if ((R > 0) || (G > 0) || (B > 0))
                s.SetColor(R, G, B, false);

        // search the named window and create one if none was found
        active_figure = FindFigure(figure_name);
        if ( active_figure == NULL)
        {
                Figure new_figure(figure_name);
                figure_list.push_back(new_figure);
                active_figure = FindFigure(figure_name);
                if (active_figure == NULL)
                        exit(-1);
        }

        active_series = active_figure->Add(s);
        active_figure->Show();

}

// add a label to the most recently added curve
void PlotManager::Label(string lbl)
{
        if((active_series!=NULL) && (active_figure != NULL))
        {
                active_series->label = lbl;
                active_figure->Show();
        }
}

// plot a new curve, if a figure of the specified figure name already exists,
// the curve will be plot on that figure; if not, a new figure will be created.
// static method
template<typename T>
void plot(const string figure_name, const T* p, int count, int step,
                  int R, int G, int B)
{
        if (step <= 0)
                step = 1;

        float  *data_copy = new float[count * step];

        float   *dst = data_copy;
        const T *src = p;

        for (int i = 0; i < count * step; i++)
        {
                *dst = (float)(*src);
                dst++;
                src++;
        }

        pm.Plot(figure_name, data_copy, count, step, R, G, B);

        delete [] data_copy;
}

// delete all plots on a specified figure
void clear(const string figure_name)
{
        Figure *fig = pm.FindFigure(figure_name);
        if (fig != NULL)
        {
                fig->Clear();
        }

}
// add a label to the most recently added curve
// static method
void label(string lbl)
{
        pm.Label(lbl);
}

////// migght be function template overloading
//template
//void plot(const string figure_name, const unsigned char* p, int count, int step,
//          int R, int G, int B);

//template
//void plot(const string figure_name, const int* p, int count, int step,
//          int R, int G, int B);

//template
//void plot(const string figure_name, const short* p, int count, int step,
//          int R, int G, int B);

//template
//void plot(const string figure_name, const float* p, int count, int step,
//          int R, int G, int B);

//template
//void plot(const string figure_name, const double* p, int count, int step,
//          int R, int G, int B);

};

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

// Visualising the results
cv::Mat& visualise_frame(cv::Mat& captured_image)
{        
        // draw center and box
        cv::Point p1(0,0);
        cv::Point p2(captured_image.cols,captured_image.rows);
        cv::Point pc = (p1+p2)*0.5;
        cv::rectangle(captured_image,p1,p2,cv::Scalar(0,255,0),5);
        cv::circle(captured_image,pc,3,cv::Scalar(0,255,0),3);
        
        return captured_image;
}

VideoStablizer::VideoStablizer( std::string path, double salient, double crop, int pathradius, int faceradius)
    : _path( path ), SmoothRatio(salient), CropRatio(crop), kSmoothingRadius(pathradius),fSmoothingRadius(faceradius)
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
    int kHorizontalBorderCrop = int(image_width*CropRatio);  // Crops the border to reduce missing pixels.
    int kVerticalBorderCrop = int(image_height*CropRatio);

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
    std::cout << "Path smoothing radius: " << kSmoothingRadius <<std::endl;
    std::cout << "Face smoothing radius: " << fSmoothingRadius <<std::endl;
    
    bool use_salient = (SmoothRatio>0.01);
    if (use_salient) {
        std::cout << "Using facial landmark with weight " << SmoothRatio << std::endl;
    } else {
        std::cout << "Not using facial landmark. " << SmoothRatio << std::endl;
    }
    
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
    std::vector<float> Tx;
    std::vector<float> Ty;
    std::vector<float> Ta;
    for( size_t i = 0; i < prev_to_cur_transform.size(); i++ )
    {
        x += prev_to_cur_transform[i]._dx;
        y += prev_to_cur_transform[i]._dy;
        a += prev_to_cur_transform[i]._da;

        trajectory.push_back( Trajectory( x, y, a ) );
        Tx.push_back(x);
        Ty.push_back(y);
        Ta.push_back(a);

        if (use_salient) {
            point_sum = std::accumulate(landmarks[i].begin(),landmarks[i].end(),point_zero);
            landmarks_avg.push_back(point_sum*(1.0/landmarks[i].size()));
        }
    }

    // Step 3 - Smooth out the trajectory using an averaging window
    std::cout << " Step 3: trajectory smoothing & show smoothed traces" << std::endl;
    std::vector<Trajectory> smoothed_trajectory; // trajectory at all frames
    std::vector<float> Tx_smooth;
    std::vector<float> Ty_smooth;
    std::vector<float> Ta_smooth;
    std::vector<float> Tfx;
    std::vector<float> Tfy;

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
            Tfx.push_back(avg_fxy.x);
            Tfy.push_back(avg_fxy.y);
            avg_x = avg_x*(1-SmoothRatio) + (sum_x-avg_fxy.x)*SmoothRatio;
            avg_y = avg_y*(1-SmoothRatio) + (sum_y-avg_fxy.y)*SmoothRatio;
        }
        smoothed_trajectory.push_back( Trajectory( avg_x, avg_y, avg_a ) );
        Tx_smooth.push_back(avg_x);
        Ty_smooth.push_back(avg_y);
        Ta_smooth.push_back(avg_a);

    }

    // show Trajectory( x, y, a ), 2D or 3D image with respect to time
    CvPlot::plot("track_x", &Tx[0], Tx.size(), 1, 255, 0, 0);
    CvPlot::label("Before");
    CvPlot::plot("track_x", &Tx_smooth[0], Tx_smooth.size(), 1, 0, 0, 255);
    CvPlot::label("After");
    CvPlot::plot("track_y", &Ty[0], Ty.size(), 1, 255, 0, 0);
    CvPlot::label("Before");
    CvPlot::plot("track_y", &Ty_smooth[0], Ty_smooth.size(), 1, 0, 0, 255);
    CvPlot::label("After");
    CvPlot::plot("track_angle", &Ta[0], Ta.size(), 1, 255, 0, 0);
    CvPlot::label("Before");
    CvPlot::plot("track_angle", &Ta_smooth[0], Ta_smooth.size(), 1, 0, 0, 255);
    CvPlot::label("After");

    cv::waitKey( 14000 );

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
    std::cout << " Writing stablized video to: " << output_path << std::endl;
         
    std::string double_path = _path.substr(0,_path.length()-4)+"_double.avi";
    cv::VideoWriter doubleVideo(
        double_path ,
        cap.get( CV_CAP_PROP_FOURCC ),
        cap.get( CV_CAP_PROP_FPS ),
        cv::Size( cap.get( CV_CAP_PROP_FRAME_WIDTH )*2+10,
                  cap.get( CV_CAP_PROP_FRAME_HEIGHT ) ) );
    std::cout << " Writing both videos to: " << double_path << std::endl;
    
    if( !outputVideo.isOpened() )
    {
        std::cout  << "Could not open the output video for write: " << std::endl;
        return -1;
    }

    std::vector< vector<cv::Point2d> > landmarks_after; // facial landmarks for each frame after smoothing
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
        outputVideo << cur2;
        
        /* face feature detection */
        if (use_salient) {
            cv::cvtColor( cur, cur_grey, cv::COLOR_BGR2GRAY );
            LandmarkDetector::DetectLandmarksInVideo(cur_grey, depth_image, clnf_model, det_parameters);
            cur = visualise_tracking(cur, clnf_model, det_parameters);
            
            cv::cvtColor( cur2, cur_grey, cv::COLOR_BGR2GRAY );
            LandmarkDetector::DetectLandmarksInVideo(cur_grey, depth_image, clnf_model, det_parameters);
            cur2 = visualise_tracking(cur2, clnf_model, det_parameters); 

            // Calculate average and get the center of face, then plot
            landmarks_after.push_back(LandmarkDetector::CalculateLandmarks(clnf_model));
        }       
        /* end face feature detection */

        // Now draw the original and stablised side by side for coolness
        cv::Mat canvas = cv::Mat::zeros( cur.rows, cur.cols * 2 + 10, cur.type() );
        cur = visualise_frame(cur);
        cur2 = visualise_frame(cur2);
        
        cur.copyTo( canvas( cv::Range::all(), cv::Range( 0, cur2.cols ) ) );
        cur2.copyTo( canvas( cv::Range::all(), cv::Range( cur2.cols + 10, cur2.cols * 2 + 10 ) ) );
             
        doubleVideo << canvas;
        
        // If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
        //if( canvas.cols > 960 )
        //{
        //    cv::resize( canvas, canvas, cv::Size( canvas.cols / 4, canvas.rows / 4 ) );
        //}

        cv::imshow( "before and after", canvas );
        cv::waitKey( 3 );

        k++;
    }

    // Step 6 - Plot the center of face after smoothing
    if (use_salient) {

        std::cout << " Step 6: Plot the center of face after smoothing" << std::endl;
        cv::Point2d point_sum_after;
        std::vector<cv::Point2d> landmarks_avg_after;
        std::vector<float> Tfx_after;
        std::vector<float> Tfy_after;
        for( size_t i = 0; i < new_prev_to_cur_transform.size(); i++ )
        {
            point_sum_after = std::accumulate(landmarks_after[i].begin(),landmarks_after[i].end(),point_zero);
            landmarks_avg_after.push_back(point_sum_after*(1.0/landmarks_after[i].size()));

        }
        for( size_t i = 0; i < new_prev_to_cur_transform.size(); i++ )
        {
            int count = 0;
            cv::Point2d avg_fxy_after(0.0,0.0);

            for (int j=-fSmoothingRadius; j<= fSmoothingRadius; j++) {
                if (i+j>=0 && i+j<landmarks_avg_after.size()) {
                    avg_fxy_after += landmarks_avg_after[i+j];
                    count++;
                }
            }

            avg_fxy_after *= (1.0/count);
            avg_fxy_after -= frame_center;    // define new
            Tfx_after.push_back(avg_fxy_after.x);
            Tfy_after.push_back(avg_fxy_after.y);
        }

        CvPlot::plot("track_face_x", &Tfx[0], Tfx.size(), 1, 255, 0, 0);
        CvPlot::label("Before");
        CvPlot::plot("track_face_x", &Tfx_after[0], Tfx_after.size(), 1, 0, 0, 255);
        CvPlot::label("After");
        CvPlot::plot("track_face_y", &Tfy[0], Tfy.size(), 1, 255, 0, 0);
        CvPlot::label("Before");
        CvPlot::plot("track_face_y", &Tfy_after[0], Tfy_after.size(), 1, 0, 0, 255);
        CvPlot::label("After");
        cv::waitKey( 14000 );
    }
    return 0;
}

int main (int argc, char **argv)
{
	vector<string> arguments = get_arguments(argc, argv);
	double SmoothRatio = 0; // SmoothRatio=1 tries to keep face at center; 0 uses pure path smoothing
	double crop = 0.1;
	int kSmoothingRadius = 20;
	int fSmoothingRadius = 5;
	
	for (size_t i=0;i<arguments.size();i++) {
	    if (arguments[i].compare("-salient")==0) {
	        stringstream data(arguments[i+1]);
	        data >> SmoothRatio;
	        i++;
	    }
	    if (arguments[i].compare("-crop")==0) {
	        stringstream data(arguments[i+1]);
	        data >> crop;
	        i++;
	    }
	    if (arguments[i].compare("-pathsmooth")==0) {
	        stringstream data(arguments[i+1]);
	        data >> kSmoothingRadius;
	        i++;
	    }	    
	    if (arguments[i].compare("-facesmooth")==0) {
	        stringstream data(arguments[i+1]);
	        data >> fSmoothingRadius;
	        i++;
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
	    VideoStablizer vs(input_path,SmoothRatio,crop,kSmoothingRadius,fSmoothingRadius);
	    vs.run(output_path,arguments);
	}
}

