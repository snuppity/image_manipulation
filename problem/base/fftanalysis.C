#include "imgproc.h"
#include "fftimgproc.h"
#include "fftanalysis.h"
using namespace img;
using namespace std;

void img::load_fft( const ImgProc& input, FFTImgProc& fftoutput )
{
   fftoutput.clear( input.nx(), input.ny(), input.depth() );
   for(int j=0;j<input.ny();j++)
   {
      for(int i=0;i<input.nx();i++)
      {
         std::vector<float> ci;
	 std::vector< std::complex<double> > citilde;
	 input.value(i,j,ci);
	 for(size_t c=0;c<ci.size();c++)
	 {
	    std::complex<double> v(ci[c], 0.0);
	    citilde.push_back(v);
	 }
	 fftoutput.set_value(i,j,citilde);
      }
   }
}

void img::twoChanEstimate(const ImgProc& input, int chan0, int chan1, float w0, float w1, ImgProc& output)
{
  double ave0 = 0.0;
  double ave1 = 0.0;
  double sigma00 = 0.0;
  double sigma01 = 0.0;
  double sigma11 = 0.0;

  for(int j = 0; j < input.ny(); j++)
  {
    for(int i = 0; i < input.nx(); i++)
    {
      vector<float> value;
      input.value(i, j, value);
      ave0 += value[chan0];
      ave1 += value[chan1];
    }
  }
  ave0 /= input.nx() * input.ny();
  ave1 /= input.nx() * input.ny();

  for(int j = 0; j < input.ny(); j++)
  {
    for(int i = 0; i < input.nx(); i++)
    {
      vector<float> value;
      input.value(i, j, value);
      sigma00 += (value[chan0]-ave0) * (value[chan0]-ave0);
      sigma01 += (value[chan0]-ave0) * (value[chan1]-ave1);
      sigma11 += (value[chan1]-ave1) * (value[chan1]-ave1);
    }
  }
  sigma00 /= input.nx() * input.ny();
  sigma01 /= input.nx() * input.ny();
  sigma11 /= input.nx() * input.ny();

  float determinant = (sigma00*sigma11) - (sigma01*sigma01);
  float denominator = w0*w0*sigma11 - 2.0*w0*w1*sigma01 + w1*w1*sigma11;
  denominator = denominator/determinant;

  output = input;

  for(int j = 0; j < input.ny(); j++)
  {
#pragma omp parallel for
    for(int i = 0; i < input.nx(); i++)
    {
      vector<float> value;
      input.value(i, j, value);
      float coherent_estimate = w0*(value[chan0]-ave0)*sigma11
                              - w0*(value[chan1]-ave1)*sigma01
                              - w1*(value[chan0]-ave0)*sigma01
                              * w1*(value[chan1]-ave1)*sigma00;
      for(size_t c=0; c < value.size(); c++)
      {
        value[c] = coherent_estimate;
      }
      output.set_value(i, j, value);
    }
  }
}
