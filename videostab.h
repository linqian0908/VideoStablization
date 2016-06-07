#ifndef VIDEOSTAB_H
#define VIDEOSTAB_H

#include "opencv2/opencv.hpp"
#include <cmath>

// This video stablisation smooths the global trajectory using a sliding average window

// 1. Get previous to current frame transformation (dx, dy, da) for all frames
// 2. Accumulate the transformations to get the image trajectory
// 3. Smooth out the trajectory using an averaging window
// 4. Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
// 5. Apply the new transformation to the video
using namespace std;

namespace CvPlot
{
        // A curve.
        class Series
        {
        public:

                // number of points
                unsigned int count;
                float *data;
                // name of the curve
                string label;

                // allow automatic curve color
                bool auto_color;
                CvScalar color;

                Series(void);
                Series(const Series& s);
                ~Series(void);

                // release memory
                void Clear();

                void SetData(int n, float *p);

                void SetColor(CvScalar color, bool auto_color = true);
                void SetColor(int R, int G, int B, bool auto_color = true);
        };

        // a figure comprises of several curves
        class Figure
        {
        private:
                // window name
                string figure_name;
                CvSize figure_size;

                // margin size
                int    border_size;

                CvScalar backgroud_color;
                CvScalar axis_color;
                CvScalar text_color;

                // several curves
                vector<Series> plots;

                // manual or automatic range
                bool custom_range_y;
                float y_max;
                float y_min;

                float y_scale;

                bool custom_range_x;
                float x_max;
                float x_min;

                float x_scale;

                // automatically change color for each curve
                int color_index;

        public:
                Figure(const string name);
                ~Figure();

                string GetFigureName();
                Series* Add(const Series &s);
                void Clear();
                void DrawLabels(IplImage *output, int posx, int posy);

                // show plot window
                void Show();

        private:
                Figure();
                void DrawAxis(IplImage *output);
                void DrawPlots(IplImage *output);

                // call before plot
                void Initialize();
                CvScalar GetAutoColor();

        };

        // manage plot windows
        class PlotManager
        {
        private:
                vector<Figure> figure_list;
                Series *active_series;
                Figure *active_figure;

        public:

                // now useless
                bool HasFigure(string wnd);
                Figure* FindFigure(string wnd);

                void Plot(const string figure_name, const float* p, int count, int step,
                                  int R, int G, int B);

                void Label(string lbl);

        };

    // handle different data types; static methods;

    /**
     * @brief Matlab style plot functions for OpenCV by Changbo (zoccob@gmail). plot and label.
     *
     * @param figure_name required. multiple calls of this function with same figure_name plots multiple curves on a single graph.
     * @param p required. pointer to data.
     * @param count required. number of data.
     * @param step optional. step between data of two points, default 1.
     * @param R optional. assign a color to the curve. if not assigned, the curve will be assigned a unique color automatically.
     * @param G
     * @param B
     */
    template<typename T>
        void plot(const string figure_name, const T* p, int count, int step = 1,
                          int R = -1, int G = -1, int B = -1);

        void clear(const string figure_name);

        void label(string lbl);

}

struct AffineTransformParam
{
    AffineTransformParam() {}
    AffineTransformParam( double x, double y,
                          double a, double b,
                          double c, double d )
        : _x( x ) , _y( y )
        , _a( a ) , _b( b )
        , _c( c ) , _d( d )
    { } // affine

    AffineTransformParam( double x, double y,
                          double a, double b )
        : _x( x ) , _y( y )
        , _a( a ) , _b( b )
        , _c( -b ) , _d( a )
    { } // rigid

    //"*": apply transform this and then c2
    AffineTransformParam applyTransform( AffineTransformParam & c2 )
    {
        AffineTransformParam param( this->_x * c2._a + this->_y * c2._b + c2._x,
                                    this->_x * c2._c + this->_y * c2._d + c2._y,
                                    this->_a * c2._a + this->_c * c2._b,
                                    this->_b * c2._a + this->_d * c2._b,
                                    this->_a * c2._c + this->_c * c2._d,
                                    this->_b * c2._c + this->_d * c2._d );
        return param;
    }

    double getScale()
    {
        return sqrt(pow(this->_a,2.0) + pow(this->_b,2.0));
    }

    double getRotation()
    {
        return atan2(this->_b,this->_a);
    }

    void normalize() {
        double norm = this->getScale();
        this->_a /= norm;
        this->_b /= norm;
        this->_c /= norm;
        this->_d /= norm;
    }

    //"="
    AffineTransformParam operator =( const AffineTransformParam & rx )
    {
        return AffineTransformParam( rx._x, rx._y, rx._a, rx._b, rx._c, rx._d );
    }

    double _x = 0.0f;
    double _y = 0.0f;
    double _a = 1.0f;
    double _b = 0.0f;
    double _c = 0.0f;
    double _d = 0.1f;
};

class VideoStablizer
{
public:
    VideoStablizer( std::string filepath );

    bool    run( std::string output_path );
    void    calcSmoothRigidTransform( const std::vector<AffineTransformParam> & transforms,
                                              std::vector<AffineTransformParam> & optimal_transform );
    void    plotTrajectory(std::vector<AffineTransformParam> oldT, std::vector<AffineTransformParam> newT);

private:

    std::vector<AffineTransformParam>     estimateTransform();

    std::string    _path;
    int            _num_frames;

    const int      kHorizontalBorderCrop = 64;  // Crops the border to reduce missing pixels.
    const float    w1 = 10.0f;
    const float    w2 = 1.0f;
    const float    w3 = 100.0f;
    const float    w_affine = 100.0f;
};

#endif // VIDEOSTAB_H
