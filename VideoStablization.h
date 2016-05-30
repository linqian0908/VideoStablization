#ifndef VIDEOSTAB_H
#define VIDEOSTAB_H


#include "opencv2/opencv.hpp"

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
    {
        _dx = dx;
        _dy = dy;
        _da = da;
    }

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
    {
        _x = x;
        _y = y;
        _a = a;
    }
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
    VideoStablizer(std::string filepath, double salient );

    bool                            run(std::string output_path, vector<string> arguments);

private:

    std::vector<TransformParam>     estimateTransform();

    std::string                     _path;
    int                             _num_frames;
    double                          SmoothRatio; // SmoothRatio=1 tries to keep face at center; 0 uses pure path smoothing
    
    const int kSmoothingRadius = 20;        // Large values give more stable video, but less flexible to sudden panning
    const int fSmoothingRadius = 5; 
    const double kHorizontalCropRatio = 0.1;
    const double kVertialCropRatio = 0.1;
};

#endif // VIDEOSTAB_H
