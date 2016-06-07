#include <iostream>
#include <cassert>
#include <fstream>
#include <opencv2/opencv.hpp>
//#include <FaceAnalyser.h>
//#include <GazeEstimation.h>

#include "videostab.h"
#include "lp_lib.h"
//#include "cvplot.h"

using namespace std;

namespace CvPlot {
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
}

VideoStablizer::VideoStablizer( std::string path )
    : _path( path )
{

}

/// This function is to calculate the optimal transformations using l1 regularization
/// We formulate the problem in linear programming, and solve the system in Algorithm 1
/// [1] Auto-Directed Video Stabilization with Robust L1 Optimal Camera Paths, Grundmann et al., CVPR11
void VideoStablizer::calcSmoothRigidTransform( const std::vector<AffineTransformParam> & transforms,
                                          std::vector<AffineTransformParam> & optimal_transform )
{
    //    optimal_transform.clear();

    // const variable
    int num_frame = transforms.size();

    // Add lp_solve solution
    lprec * lp;
    int e_num = 4;
    int Ncol    = num_frame * (3+1) * e_num; // number of variables, including e and all p_t
    // first 18*num_frame are e_t^i (e_1^1, ..., e_n^1, e_1^2,...,e_n^2,e_1^3,...,e_n^3), e_t^i is a 4-dim vector
    // the rest are p_t
    int e_shift = num_frame * e_num;  // length of e^i = ( e_1^i, ... , e_n^i )
    int p_shift = num_frame * 3 * e_num; // length of e   = ( e^1, e^2, e^3 ) for first, second and third order

    lp = make_lp( 0, Ncol );
    set_add_rowmode( lp, TRUE );

    // create space for one row
    int * colno = NULL;
    REAL * row = NULL;
    colno = ( int * ) malloc( Ncol * sizeof( *colno ) );
    row = ( REAL * ) malloc( Ncol * sizeof( *row ) );

    // add constraints of e_t>=0
    row[0] = 1;
    for( int i = 0; i < e_shift; ++i )
    {
        set_unbounded(lp,p_shift+i+1);
    }

    // add constraints of -e_t^1 <= R_t(p) <= e_t^1. |Ft+1 Bt+1 - Ft|
    for( int i = 0; i < num_frame - 1; ++i )
    {
        AffineTransformParam param = transforms[i + 1];

        // tx component
        colno[0] = i * e_num + 1;
        row  [0] = 1;
        colno[1] = p_shift + i * e_num + 1;
        row  [1] = -1;
        colno[2] = p_shift + ( i + 1 ) * e_num + 1;
        row  [2] = 1;
        colno[3] = p_shift + ( i + 1 ) * e_num + 3;
        row  [3] = param._x;
        colno[4] = p_shift + ( i + 1 ) * e_num + 4;
        row  [4] = param._y;

        add_constraintex( lp, 5, row, colno, GE, 0 );
        row  [0] = -1;
        add_constraintex( lp, 5, row, colno, LE, 0 );

        // ty component
        colno[0] = i * e_num + 2;
        row  [0] = 1;
        colno[1] = p_shift + i * e_num + 2;
        row  [1] = -1;
        colno[2] = p_shift + ( i + 1 ) * e_num + 2;
        row  [2] = 1;
        colno[3] = p_shift + ( i + 1 ) * e_num + 3;
        row  [3] = param._y;
        colno[4] = p_shift + ( i + 1 ) * e_num + 4;
        row  [4] = -param._x;

        add_constraintex( lp, 5, row, colno, GE, 0 );
        row  [0] = -1;
        add_constraintex( lp, 5, row, colno, LE, 0 );

        // a component
        colno[0] = i * e_num + 3;
        row  [0] = 1;
        colno[1] = p_shift + i * e_num + 3;
        row  [1] = -1;
        colno[2] = p_shift + ( i + 1 ) * e_num + 3;
        row  [2] = param._a;
        colno[3] = p_shift + ( i + 1 ) * e_num + 4;
        row  [3] = -param._b;

        add_constraintex( lp, 4, row, colno, GE, 0 );
        row  [0] = -1;
        add_constraintex( lp, 4, row, colno, LE, 0 );

        // b component
        colno[0] = i * e_num + 4;
        row  [0] = 1;
        colno[1] = p_shift + i * e_num + 4;
        row  [1] = -1;
        colno[2] = p_shift + ( i + 1 ) * e_num + 3;
        row  [2] = param._b;
        colno[3] = p_shift + ( i + 1 ) * e_num + 4;
        row  [3] = param._a;

        add_constraintex( lp, 4, row, colno, GE, 0 );
        row  [0] = -1;
        add_constraintex( lp, 4, row, colno, LE, 0 );
    }

    // add constraints of -e_t^2 <= R_(t+1)(p) - R_t(p) <= e_t^2. |Ft+2 Bt+2 - (I + Ft+1)Bt+1 + Bt|
    for( int i = 0; i < num_frame - 2; ++i )
    {
        AffineTransformParam param1 = transforms[i + 1];
        AffineTransformParam param2 = transforms[i + 2];

        // tx component
        colno[0] = e_shift + i * e_num + 1;
        row  [0] = 1;
        colno[1] = p_shift + i * e_num + 1;
        row  [1] = 1;
        colno[2] = p_shift + ( i + 1 ) * e_num + 1;
        row  [2] = -2;
        colno[3] = p_shift + ( i + 1 ) * e_num + 3;
        row  [3] = -param1._x;
        colno[4] = p_shift + ( i + 1 ) * e_num + 4;
        row  [4] = -param1._y;
        colno[5] = p_shift + ( i + 2 ) * e_num + 1;
        row  [5] = 1;
        colno[6] = p_shift + ( i + 2 ) * e_num + 3;
        row  [6] = param2._x;
        colno[7] = p_shift + ( i + 2 ) * e_num + 4;
        row  [7] = param2._y;

        add_constraintex( lp, 8, row, colno, GE, 0 );
        row  [0] = -1;
        add_constraintex( lp, 8, row, colno, LE, 0 );

        // ty component
        colno[0] = e_shift + i * e_num + 2;
        row  [0] = 1;
        colno[1] = p_shift + i * e_num + 2;
        row  [1] = 1;
        colno[2] = p_shift + ( i + 1 ) * e_num + 2;
        row  [2] = -2;
        colno[3] = p_shift + ( i + 1 ) * e_num + 3;
        row  [3] = -param1._y;
        colno[4] = p_shift + ( i + 1 ) * e_num + 4;
        row  [4] = param1._x;
        colno[5] = p_shift + ( i + 2 ) * e_num + 2;
        row  [5] = 1;
        colno[6] = p_shift + ( i + 2 ) * e_num + 3;
        row  [6] = param2._y;
        colno[7] = p_shift + ( i + 2 ) * e_num + 4;
        row  [7] = -param2._x;

        add_constraintex( lp, 8, row, colno, GE, 0 );
        row  [0] = -1;
        add_constraintex( lp, 8, row, colno, LE, 0 );

        // a component
        colno[0] = e_shift + i * e_num + 3;
        row  [0] = 1;
        colno[1] = p_shift + i * e_num + 3;
        row  [1] = 1;
        colno[2] = p_shift + ( i + 1 ) * e_num + 3;
        row  [2] = -1 - param1._a;
        colno[3] = p_shift + ( i + 1 ) * e_num + 4;
        row  [3] = param1._b;
        colno[4] = p_shift + ( i + 2 ) * e_num + 3;
        row  [4] = param2._a;
        colno[5] = p_shift + ( i + 2 ) * e_num + 4;
        row  [5] = -param2._b;

        add_constraintex( lp, 6, row, colno, GE, 0 );
        row  [0] = -1;
        add_constraintex( lp, 6, row, colno, LE, 0 );

        // b component
        colno[0] = e_shift + i * e_num + 4;
        row  [0] = 1;
        colno[1] = p_shift + i * e_num + 4;
        row  [1] = 1;
        colno[2] = p_shift + ( i + 1 ) * e_num + 3;
        row  [2] = - param1._b;
        colno[3] = p_shift + ( i + 1 ) * e_num + 4;
        row  [3] = -1 - param1._a;
        colno[4] = p_shift + ( i + 2 ) * e_num + 3;
        row  [4] = param2._b;
        colno[5] = p_shift + ( i + 2 ) * e_num + 4;
        row  [5] = param2._a;

        add_constraintex( lp, 6, row, colno, GE, 0 );
        row  [0] = -1;
        add_constraintex( lp, 6, row, colno, LE, 0 );
    }


    // add constraints of -e_t^2 <= R_(t+2)(p) - 2R_(t+1)(p) + R_t(p) <= e_t^2. |Ft+3-(I + 2 Ft+2)Bt+2 + (2I + Ft+1) Bt+1 - Bt|
    for( int i = 0; i < num_frame - 3; ++i )
    {
        AffineTransformParam param1 = transforms[i + 1];
        AffineTransformParam param2 = transforms[i + 2];
        AffineTransformParam param3 = transforms[i + 3];

        // tx component
        colno[0]  = e_shift * 2 + i * e_num + 1;
        row  [0]  = 1;
        colno[1]  = p_shift + i * e_num + 1;
        row  [1]  = -1;
        colno[2]  = p_shift + ( i + 1 ) * e_num + 1;
        row  [2]  = 3;
        colno[3]  = p_shift + ( i + 1 ) * e_num + 3;
        row  [3]  = param1._x;
        colno[4]  = p_shift + ( i + 1 ) * e_num + 4;
        row  [4]  = param1._y;
        colno[5]  = p_shift + ( i + 2 ) * e_num + 1;
        row  [5]  = -3;
        colno[6]  = p_shift + ( i + 2 ) * e_num + 3;
        row  [6]  = - 2 * param2._x;
        colno[7]  = p_shift + ( i + 2 ) * e_num + 4;
        row  [7]  = - 2 * param1._y;
        colno[8]  = p_shift + ( i + 3 ) * e_num + 1;
        row  [8]  = 1;
        colno[9]  = p_shift + ( i + 3 ) * e_num + 3;
        row  [9]  = param3._x;
        colno[10] = p_shift + ( i + 3 ) * e_num + 4;
        row  [10] = param3._y;

        add_constraintex( lp, 11, row, colno, GE, 0 );
        row  [0] = -1;
        add_constraintex( lp, 11, row, colno, LE, 0 );

        // ty component
        colno[0]  = e_shift * 2 + i * e_num + 2;
        row  [0]  = 1;
        colno[1]  = p_shift + i * e_num + 2;
        row  [1]  = -1;
        colno[2]  = p_shift + ( i + 1 ) * e_num + 2;
        row  [2]  = 3;
        colno[3]  = p_shift + ( i + 1 ) * e_num + 3;
        row  [3]  = param1._y;
        colno[4]  = p_shift + ( i + 1 ) * e_num + 4;
        row  [4]  = -param1._x;
        colno[5]  = p_shift + ( i + 2 ) * e_num + 2;
        row  [5]  = -3;
        colno[6]  = p_shift + ( i + 2 ) * e_num + 3;
        row  [6]  = - 2 * param2._y;
        colno[7]  = p_shift + ( i + 2 ) * e_num + 4;
        row  [7]  = + 2 * param2._x;
        colno[8]  = p_shift + ( i + 3 ) * e_num + 2;
        row  [8]  = 1;
        colno[9]  = p_shift + ( i + 3 ) * e_num + 3;
        row  [9]  = param3._y;
        colno[10] = p_shift + ( i + 3 ) * e_num + 4;
        row  [10] = -param3._x;


        add_constraintex( lp, 11, row, colno, GE, 0 );
        row  [0]  = -1;
        add_constraintex( lp, 11, row, colno, LE, 0 );

        // a component
        colno[0] = e_shift * 2 + i * e_num + 3;
        row  [0] = 1;
        colno[1] = p_shift + i * e_num + 3;
        row  [1] = -1;
        colno[2] = p_shift + ( i + 1 ) * e_num + 3;
        row  [2] = 2 + param1._a;
        colno[3] = p_shift + ( i + 1 ) * e_num + 4;
        row  [3] = - param1._b;
        colno[4] = p_shift + ( i + 2 ) * e_num + 3;
        row  [4] = -1 - 2 * param2._a;
        colno[5] = p_shift + ( i + 2 ) * e_num + 4;
        row  [5] = 2 * param2._b;
        colno[6] = p_shift + ( i + 3 ) * e_num + 3;
        row  [6] = param3._a;
        colno[7] = p_shift + ( i + 3 ) * e_num + 4;
        row  [7] = -param3._b;

        add_constraintex( lp, 8, row, colno, GE, 0 );
        row  [0] = -1;
        add_constraintex( lp, 8, row, colno, LE, 0 );

        // b component
        colno[0] = e_shift * 2 + i * e_num + 4;
        row  [0] = 1;
        colno[1] = p_shift + i * e_num + 4;
        row  [1] = -1;
        colno[2] = p_shift + ( i + 1 ) * e_num + 3;
        row  [2] = param1._b;
        colno[3] = p_shift + ( i + 1 ) * e_num + 4;
        row  [3] = 2 + param1._a;
        colno[4] = p_shift + ( i + 2 ) * e_num + 3;
        row  [4] = - 2 * param2._b;
        colno[5] = p_shift + ( i + 2 ) * e_num + 4;
        row  [5] = -1 - 2 * param2._a;
        colno[6] = p_shift + ( i + 3 ) * e_num + 3;
        row  [6] = param3._b;
        colno[7] = p_shift + ( i + 3 ) * e_num + 4;
        row  [7] = param3._a;

        add_constraintex( lp, 8, row, colno, GE, 0 );
        row  [0] = -1;
        add_constraintex( lp, 8, row, colno, LE, 0 );
    }

    // add strict bound for affine parameters
    // 0.9 <= a_t,d_t <= 1.1, -0.1 <= b_t,c_t <= 0.1
    // -0.05 <= b_t + c_t <= 0.05, -0.1 <= a_t - d_t <= 0.1
    const float a_lb = 0.95f;
    const float a_ub = 1.05f;
    const float b_lb = -0.05f;
    const float b_ub = 0.05f;
    const float hcrop  = 32.0;
    const float vcrop  = 18.0;

    row[0] = 1;
    for( int i = 0; i < num_frame; ++i )
    {
        colno[0] = p_shift + i * e_num + 1;
        add_constraintex( lp, 1, row, colno, LE, hcrop );
        add_constraintex( lp, 1, row, colno, GE, -hcrop );

        colno[0] = p_shift + i * e_num + 2;
        add_constraintex( lp, 1, row, colno, LE, vcrop );
        add_constraintex( lp, 1, row, colno, GE, -vcrop );

        colno[0] = p_shift + i * e_num + 3;
        add_constraintex( lp, 1, row, colno, LE, a_ub );
        add_constraintex( lp, 1, row, colno, GE, a_lb );

        colno[0] = p_shift + i * e_num + 4;
        add_constraintex( lp, 1, row, colno, LE, b_ub );
        add_constraintex( lp, 1, row, colno, GE, b_lb );
    }

    row  [0] = 1;
    colno[0] = p_shift + 1;
    add_constraintex( lp, 1, row, colno, EQ, 0 ); // x0=0
    colno[0] = p_shift + 2;
    add_constraintex( lp, 1, row, colno, EQ, 0 ); // y0=0
    colno[0] = p_shift + 3;
    add_constraintex( lp, 1, row, colno, EQ, 1 ); // a0=1
    colno[0] = p_shift + 4;
    add_constraintex( lp, 1, row, colno, EQ, 0 ); // b0=0

    set_add_rowmode( lp, FALSE ); /* rowmode should be turned off again when done building the model */

    // set the objective function c^t e
    for( int i = 0; i < e_shift; i += e_num )
    {
        for( int j = 0; j < 2; ++j )
        {
            int ind = i + j;
            colno[ind] = ind + 1;
            row  [ind] = w1;

            colno[ind + e_shift] = ind + e_shift + 1;
            row  [ind + e_shift] = w2;

            colno[ind + e_shift * 2] = ind + e_shift * 2 + 1;
            row  [ind + e_shift * 2] = w3;
        }

        for( int j = 2; j < e_num; ++j )
        {
            int ind = i + j;
            colno[ind] = ind + 1;
            row  [ind] = w1 * w_affine;

            colno[ind + e_shift] = ind + e_shift + 1;
            row  [ind + e_shift] = w2 * w_affine;

            colno[ind + e_shift * 2] = ind + e_shift * 2 + 1;
            row  [ind + e_shift * 2] = w3 * w_affine;
        }
    }

    set_obj_fnex( lp, p_shift, row, colno );
    set_minim( lp );
    //    write_LP( lp, stdout );
    //    set_verbose( lp, IMPORTANT );
    solve( lp );

    //    /* objective value */
    //    printf( "Objective value: %f\n", get_objective( lp ) );
    get_variables( lp, row );
    optimal_transform.clear();

    //std::cout << ">>>>> Otimization output: Bt s >>>>> " << std::endl;
    //    optimal_transform = std::vector<AffineTransformParam>();
    for( size_t i = 0; i < num_frame; i++ )
    {
        int ind = p_shift + i * e_num;
        AffineTransformParam param( row[ind], row[ind + 1], row[ind + 2], row[ind + 3]);
        //printf( "%f %f %f %f \n", row[ind], row[ind + 1], row[ind + 2], row[ind + 3] );
        param.normalize();
        optimal_transform.push_back( param );
        //        optimal_transform[i] = param;
    }

    /* free allocated memory */
    if( row != NULL )
    {
        free( row );
    }
    if( colno != NULL )
    {
        free( colno );
    }

    if( lp != NULL )
    {
        /* clean up such that all used memory by lpsolve is freed */
        delete_lp( lp );
    }
}

void VideoStablizer::plotTrajectory(std::vector<AffineTransformParam> oldT, std::vector<AffineTransformParam> newT) {
    std::vector<float> oTx, nTx, oTy, nTy, oTs, nTs, oTa, nTa;
    float oX(0.0),oY(0.0),oA(0.0),nX(0.0),nY(0.0),nA(0.0);
    for (size_t i=0; i<oldT.size();i++) {
        oTx.push_back(oldT[i]._x);
        oTy.push_back(oldT[i]._y);
        //oTs.push_back(oldT[i].getScale());
        oTa.push_back(-oldT[i].getRotation());
        if (i>0) {
            oX += abs(oldT[i]._x-oldT[i-1]._x);
            oY += abs(oldT[i]._y-oldT[i-1]._y);
            oA += abs(oldT[i]._a-oldT[i-1]._a);
        }
    }
    for (size_t i=0; i<newT.size();i++) {
        nTx.push_back(newT[i]._x);
        nTy.push_back(newT[i]._y);
        //nTs.push_back(newT[i].getScale());
        nTa.push_back(-newT[i].getRotation());
        if (i>0) {
            nX += abs(newT[i]._x-newT[i-1]._x);
            nY += abs(newT[i]._y-newT[i-1]._y);
            nA += abs(newT[i]._a-newT[i-1]._a);
        }
    }
    std::cout<< "frame "<< newT.size() <<std::endl;
    printf( "old: %f %f %f\n", oX,oY,oA );
    printf( "new: %f %f %f\n", nX,nY,nA );
    cv::waitKey(100);

    CvPlot::plot("track_x", &oTx[0], oTx.size(), 1, 255, 0, 0);
    CvPlot::label("Before");
    CvPlot::plot("track_x", &nTx[0], nTx.size(), 1, 0, 0, 255);
    CvPlot::label("After");

    CvPlot::plot("track_y", &oTy[0], oTy.size(), 1, 255, 0, 0);
    CvPlot::label("Before");
    CvPlot::plot("track_y", &nTy[0], nTy.size(), 1, 0, 0, 255);
    CvPlot::label("After");

    CvPlot::plot("track_a", &oTa[0], oTa.size(), 1, 255, 0, 0);
    CvPlot::label("Before");
    CvPlot::plot("track_a", &nTa[0], nTa.size(), 1, 0, 0, 255);
    CvPlot::label("After");
}

bool VideoStablizer::run( std::string output_path )
{
    cv::VideoCapture cap( _path );
    assert( cap.isOpened() );

    cv::Mat cur, cur_grey;
    cv::Mat prev, prev_grey;

    cap >> prev;//get the first frame.ch
    cv::cvtColor( prev, prev_grey, cv::COLOR_BGR2GRAY );

    // Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
    std::vector<AffineTransformParam> prev_to_cur_transform; // previous to current

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

        // translation + rotation only, rigid transform and no shearing. RigidTransform does allow scaling
        cv::Mat T = cv::estimateRigidTransform( prev_corner2, cur_corner2, false );
        //        cv::Mat T = cv::getAffineTransform( prev_corner2, cur_corner2 );

        // in rare cases no transform is found. We'll just use the last known good transform.
        if( T.data == NULL ) { last_T.copyTo( T ); }
        T.copyTo( last_T );

        // decompose T
        AffineTransformParam param( T.at<double>( 0, 2 ), T.at<double>( 1, 2 ),
                                    T.at<double>( 0, 0 ), T.at<double>( 0, 1 ),
                                    T.at<double>( 1, 0 ), T.at<double>( 1, 1 ) );
        param.normalize();
        prev_to_cur_transform.push_back( param );

        cur.copyTo( prev );
        cur_grey.copyTo( prev_grey );

        k++;
    }

    // Step 2 - Apply L1 optimization to obtain framewise old to new transformation for smooth output
    std::vector<AffineTransformParam> new_prev_to_cur_transform;
    //    std::cout << ">>>>> Original frame to frame transform: Ft s >>>>> " << std::endl;
    //    for( size_t i = 0; i < prev_to_cur_transform.size(); ++i )
    //    {
    //        AffineTransformParam param = prev_to_cur_transform[i];
    //        printf( "%f %f %f %f %f %f\n", param._a, param._b, param._c, param._d, param._x, param._y );
    //    }

    // calcSmoothAffineTransform( prev_to_cur_transform, new_prev_to_cur_transform );
    calcSmoothRigidTransform( prev_to_cur_transform, new_prev_to_cur_transform );
    std::cout << ">>>>> Optmized frame-wise transform: Bt s >>>>> " << std::endl;
    for( size_t i = 0; i < new_prev_to_cur_transform.size(); ++i )
    {
        AffineTransformParam param = new_prev_to_cur_transform[i];
        printf( "%f %f %f %f %f %f\n", param._a, param._b, param._c, param._d, param._x, param._y );
    }
    std::cout << ">>>>> These are all Bt s >>>>> " << std::endl;

    // Step 3 - Apply the new transformation to the video
    cap.set( cv::CAP_PROP_POS_FRAMES, 0 );
    cv::Mat transform( 2, 3, CV_64F );

    const int vert_border = kHorizontalBorderCrop * prev.rows / prev.cols; // get the aspect ratio correct
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

    k = 0;
    while( k < max_frames - 1 ) // don't process the very last frame, no valid transform
    {
        cap >> cur;
        if( cur.data == NULL ) { break; }

        AffineTransformParam param = new_prev_to_cur_transform[k];
        transform.at<double>( 0, 0 ) = param._a;
        transform.at<double>( 0, 1 ) = param._b;
        transform.at<double>( 1, 0 ) = param._c;
        transform.at<double>( 1, 1 ) = param._d;
        transform.at<double>( 0, 2 ) = param._x;
        transform.at<double>( 1, 2 ) = param._y;

        cv::Mat cur2;
        cv::warpAffine( cur, cur2, transform, cur.size() );
        cur2 = cur2( cv::Range( vert_border, cur2.rows - vert_border ),
                     cv::Range( kHorizontalBorderCrop, cur2.cols - kHorizontalBorderCrop ) );

        // Resize cur2 back to cur size, for better side by side comparison
        cv::resize( cur2, cur2, cur.size() );

        //        // Now draw the original and stablised side by side for coolness
        cv::Mat canvas = cv::Mat::zeros( cur.rows, cur.cols * 2 + 10, cur.type() );

        cur.copyTo( canvas( cv::Range::all(), cv::Range( 0, cur2.cols ) ) );
        cur2.copyTo( canvas( cv::Range::all(), cv::Range( cur2.cols + 10, cur2.cols * 2 + 10 ) ) );

        outputVideo << cur2;

        //        // If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
        //        if( canvas.cols > 960 )
        //        {
        //            cv::resize( canvas, canvas, cv::Size( canvas.cols / 2, canvas.rows / 2 ) );
        //        }

        cv::imshow( "before and after", canvas );
        cv::waitKey( 30 );

        k++;
    }

    // Step 4 - Accumulate the transformations to get the image trajectory for original and new
    std::vector<AffineTransformParam> trajectory; // trajectory at all frames
    trajectory.push_back(AffineTransformParam(0.0,0.0,1.0,0.0));
    std::vector<AffineTransformParam> new_trajectory; // trajectory at all frames
    trajectory.push_back(AffineTransformParam(0.0,0.0,1.0,0.0));
    for( size_t i = 0; i < prev_to_cur_transform.size(); i++ )
    {
        trajectory.push_back(trajectory.back().applyTransform(prev_to_cur_transform[i]));
        new_trajectory.push_back(trajectory.back().applyTransform(new_prev_to_cur_transform[i]));
    }
    plotTrajectory(trajectory, new_trajectory);
    cv::waitKey(0);

    return 0;
}
