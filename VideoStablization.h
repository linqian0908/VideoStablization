#ifndef VIDEOSTAB_H
#define VIDEOSTAB_H


#include "opencv2/opencv.hpp"


using namespace std;



// This video stablisation smooths the global trajectory using a sliding average window

// 1. Get previous to current frame transformation (dx, dy, da) for all frames
// 2. Accumulate the transformations to get the image trajectory
// 3. Smooth out the trajectory using an averaging window
// 4. Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
// 5. Apply the new transformation to the video

struct TransformParam
{
    TransformParam() {}
    TransformParam( double dx,
                    double dy,
                    double da )
        : _dx( dx )
        , _dy( dy )
        , _da( da )
    { }

    double _dx = 0.0f;
    double _dy = 0.0f;
    double _da = 0.0f;
};



struct Trajectory
{
    Trajectory() {}
    Trajectory( double x,
                double y,
                double a )
        : _x( x )
        , _y( y )
        , _a( a )
    { }

    // "+"
    friend Trajectory operator+( const Trajectory & c1, const Trajectory & c2 )
    {
        return Trajectory( c1._x + c2._x,
                           c1._y + c2._y,
                           c1._a + c2._a );
    }
    //"-"
    friend Trajectory operator-( const Trajectory & c1, const Trajectory & c2 )
    {
        return Trajectory( c1._x - c2._x,
                           c1._y - c2._y,
                           c1._a - c2._a );
    }
    //"*"
    friend Trajectory operator*( const Trajectory & c1, const Trajectory & c2 )
    {
        return Trajectory( c1._x * c2._x,
                           c1._y * c2._y,
                           c1._a * c2._a );
    }
    //"/"
    friend Trajectory operator/( const Trajectory & c1, const Trajectory & c2 )
    {
        return Trajectory( c1._x / c2._x,
                           c1._y / c2._y,
                           c1._a / c2._a );
    }
    //"="
    Trajectory operator =( const Trajectory & rx )
    {
        return Trajectory( rx._x, rx._y, rx._a );
    }

    double _x;
    double _y;
    double _a; // angle
};



class VideoStablizer
{
public:
    VideoStablizer( std::string filepath, double salient, double crop, int pathradius, int faceradius );

    bool                            run( std::string output_path, vector<string> arguments );
    void                            addHat( cv::Mat & cur2, vector<cv::Vec2f> & landmarks );
    void                            addPNG( int x0, int y0, cv::Mat & cur2, cv::Mat & pic, cv::Mat & alpha );
    void                            overlayImg( int x0, int y0, cv::Mat & cur2, cv::Mat pic, cv::Mat & alpha );
    void                            addGlasses( cv::Mat & cur2, vector<cv::Vec2f> & landmarks );
private:

    std::vector<TransformParam>     estimateTransform();
    void                            rotateFigure( const cv::Mat & fig_in,
                                                  const cv::Mat & alpha_in,
                                                  cv::Mat & fig_out,
                                                  cv::Mat & alpha_out,
                                                  cv::Vec2f & pt_left,
                                                  cv::Vec2f & pt_right );

    std::string                     _path;
    int                             _num_frames;
    double                          _smooth_ratio; // SmoothRatio=1 tries to keep face at center; 0 uses pure path smoothing
    double                          _crop_ratio;

    int                             _path_smoothing_radius;
    // Large values give more stable video, but less flexible to sudden panning
    int                             _face_smoothing_radius;
};

#endif // VIDEOSTAB_H
