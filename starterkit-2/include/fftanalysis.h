
#ifndef FFTANALYSIS_H
#define FFTANALYSIS_H

#include <string>
#include "fftimgproc.h"
#include "imgproc.h"

namespace img
{

void load_fft( const ImgProc& input, FFTImgProc& fftoutput );

void twoChanEstimate(const ImgProc& input, int chan1, int chan2, float w1, float w2, ImgProc& output);
}
#endif
