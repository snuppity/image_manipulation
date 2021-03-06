#include <cmath>
#include <vector>
#include "imgproc.h"
#include "fftimgproc.h"
#include "fftanalysis.h"

#include <OpenImageIO/imageio.h>
OIIO_NAMESPACE_USING

using namespace img;
using namespace std;


ImgProc::ImgProc() :
  Nx (0),
  Ny (0),
  Nc (0),
  Nsize (0),
  img_data (nullptr)
{}

ImgProc::~ImgProc()
{
   clear();
}

void ImgProc::clear()
{
   if( img_data != nullptr ){ delete[] img_data; img_data = nullptr;}
   Nx = 0;
   Ny = 0;
   Nc = 0;
   Nsize = 0;
}

void ImgProc::clear(int nX, int nY, int nC)
{
   clear();
   Nx = nX;
   Ny = nY;
   Nc = nC;
   Nsize = (long)Nx * (long)Ny * (long)Nc;
   img_data = new float[Nsize];
#pragma omp parallel for
   for(long i=0;i<Nsize;i++){ img_data[i] = 0.0; }
}

bool ImgProc::load( const std::string& filename )
{
   auto in = ImageInput::create (filename);
   if (!in) {return false;}
   ImageSpec spec;
   in->open (filename, spec);
   clear();
   Nx = spec.width;
   Ny = spec.height;
   Nc = spec.nchannels;
   Nsize = (long)Nx * (long)Ny * (long)Nc;
   img_data = new float[Nsize];
   in->read_image(TypeDesc::FLOAT, img_data);
   in->close ();
   return true;
}


void ImgProc::value( int i, int j, std::vector<float>& pixel) const
{
   pixel.clear();
   if( img_data == nullptr ){ return; }
   if( i<0 || i>=Nx ){ return; }
   if( j<0 || j>=Ny ){ return; }
   pixel.resize(Nc);
   for( int c=0;c<Nc;c++ )
   {
      pixel[c] = img_data[index(i,j,c)];
   }
   return;
}

void ImgProc::set_value( int i, int j, const std::vector<float>& pixel)
{
   if( img_data == nullptr ){ return; }
   if( i<0 || i>=Nx ){ return; }
   if( j<0 || j>=Ny ){ return; }
   if( Nc > (int)pixel.size() ){ return; }
#pragma omp parallel for
   for( int c=0;c<Nc;c++ )
   {
      img_data[index(i,j,c)] = pixel[c];
   }
   return;
}

void ImgProc::output()
{
  const char* filename = "output.jpg";

  ImageOutput* out = ImageOutput::create(filename);
  if(!out) { return; }
  ImageSpec spec(Nx, Ny, Nc, TypeDesc::FLOAT);
  out->open(filename, spec);
  out->write_image(TypeDesc::FLOAT, img_data);
  out->close();
  ImageOutput::destroy(out);
}

ImgProc::ImgProc(const ImgProc& v) :
  Nx (v.Nx),
  Ny (v.Ny),
  Nc (v.Nc),
  Nsize (v.Nsize)
{
   img_data = new float[Nsize];
#pragma omp parallel for
   for( long i=0;i<Nsize;i++){ img_data[i] = v.img_data[i]; }
}

ImgProc& ImgProc::operator=(const ImgProc& v)
{
   if( this == &v ){ return *this; }
   if( Nx != v.Nx || Ny != v.Ny || Nc != v.Nc )
   {
      clear();
      Nx = v.Nx;
      Ny = v.Ny;
      Nc = v.Nc;
      Nsize = v.Nsize;
   }
   img_data = new float[Nsize];
#pragma omp parallel for
   for( long i=0;i<Nsize;i++){ img_data[i] = v.img_data[i]; }
   return *this;
}


void ImgProc::operator*=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] *= v; }
}

void ImgProc::operator/=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] /= v; }
}

void ImgProc::operator+=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] += v; }
}

void ImgProc::operator-=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] -= v; }
}


void ImgProc::compliment()
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] = 1.0 - img_data[i]; }
}

long ImgProc::index(int i, int j, int c) const
{
   return (long) c + (long) Nc * index(i,j); // interleaved channels

   // return index(i,j) + (long)Nx * (long)Ny * (long)c; // sequential channels
}

long ImgProc::index(int i, int j) const
{
   return (long) i + (long)Nx * (long)j;
}

void ImgProc::rmsContrast()
{
   float redRes = 0.0;
   float greenRes = 0.0;
   float blueRes = 0.0;
   float redMean = 0.0;
   float greenMean = 0.0;
   float blueMean = 0.0;

   if(img_data == nullptr) { return; }
   for(long i = 0; i < Nsize; i += Nc)
   {
      redMean += img_data[i];
      greenMean += img_data[i+1];
      blueMean += img_data[i+2];
   }
   redMean = redMean / (Nx * Ny);
   greenMean = greenMean / (Nx * Ny);
   blueMean = blueMean / (Nx * Ny);

   for(long i = 0; i < Nsize; i += Nc)
   {
      redRes += pow(img_data[i] - redMean, 2);
      greenRes += pow(img_data[i+1] - greenMean, 2);
      blueRes += pow(img_data[i+2] - blueMean, 2);
   }
   redRes = sqrt(redRes / (Nx * Ny));
   greenRes = sqrt(greenRes / (Nx * Ny));
   blueRes = sqrt(blueRes / (Nx * Ny));

   for(long i = 0; i < Nsize; i += Nc)
   {
      img_data[i] = (img_data[i] - redMean) / redRes;
      img_data[i+1] = (img_data[i+1] - greenMean) / greenRes;
      img_data[i+2] = (img_data[i+2] - blueMean) / blueRes;
   }
}

void img::swap(ImgProc& u, ImgProc& v)
{
   float* temp = v.img_data;
   int Nx = v.Nx;
   int Ny = v.Ny;
   int Nc = v.Nc;
   long Nsize = v.Nsize;

   v.Nx = u.Nx;
   v.Ny = u.Ny;
   v.Nc = u.Nc;
   v.Nsize = u.Nsize;
   v.img_data = u.img_data;

   u.Nx = Nx;
   u.Ny = Ny;
   u.Nc = Nc;
   u.Nsize = Nsize;
   u.img_data = temp;
}


//proj 5 functions
LinearWaveEstimate::LinearWaveEstimate(const ImgProc& init, const double disp_factor) :
  alpha(disp_factor),
  frameCount(0)
  {
    A.clear(init.nx(), init.ny(), init.depth());
    B.clear(init.nx(), init.ny(), init.depth());
  }

double LinearWaveEstimate::dispersion(double kx, double ky) const
{
  double kmag = sqrt(kx*kx + ky*ky);
  double freq = alpha * sqrt(kmag);
return freq;
}

void LinearWaveEstimate::ingest(const ImgProc& I)
{
  FFTImgProc Itilde;
  img::load_fft(I, Itilde);
  Itilde.fft_forward();

  for(int j = 0; j < Itilde.ny(); j++)
  {
#pragma omp parallel for
    for(int i = 0; i < Itilde.nx(); i++)
    {
      vector<complex<double>> itilde;
      vector<complex<double>> a;
      vector<complex<double>> b;
      Itilde.value(i, j, itilde);
      cout << "Itilde value\n";
      A.value(i, j, a);
      cout << "A value\n";
      B.value(i, j, b);
      cout << "B value\n";

      vector<complex<double>> aupdate = a;
      vector<complex<double>> bupdate = b;

      complex<double> phase(0.0, frameCount * dispersion(Itilde.kx(i), Itilde.ky(j)));
      phase = exp(phase);
      double one_over_N = 1.0 / (frameCount + 1);
      cout << "after 1/N\n";
      for(size_t c = 0; c < itilde.size(); c++)
      {
        aupdate[c] += (itilde[c]/phase - b[c]/(phase*phase) - a[c]) * one_over_N;
        bupdate[c] += (itilde[c]*phase - a[c]*(phase*phase) - b[c]) * one_over_N;
        cout << "abupdating\n";
      }
      A.set_value(i, j, aupdate);
      B.set_value(i, j, bupdate);
      cout << "ab set value\n";
    }
  }
  frameCount++;
}

void LinearWaveEstimate::lweValue(int i, int j, int n, vector<complex<double>>& amplitude)
{
  complex<double> phase(0.0, n * dispersion(A.kx(i), A.ky(j)));
  phase = exp(phase);
  vector<complex<double>> a;
  vector<complex<double>> b;
  A.value(i, j, a);
  B.value(i, j, b);
  amplitude.resize(a.size());
  for(size_t c = 0; c < a.size(); c++)
  {
    amplitude[c] = a[c]*phase + b[c]/phase;
  }
}

void img::extract_image(LinearWaveEstimate& l, int frame, ImgProc& img)
{
  img.clear(l.getA().nx(), l.getA().ny(), l.getA().depth());
  FFTImgProc fftimg;
  fftimg.clear(img.nx(), img.ny(), img.depth());
  for(int j = 0; j < img.ny(); j++)
  {
    for(int i = 0; i < img.nx(); i++)
    {
      vector<complex<double>> v;
      l.lweValue(i, j, frame, v);
      fftimg.set_value(i, j, v);
    }
  }

  fftimg.fft_backward();
  for(int j = 0; j < img.ny(); j++)
  {
    for(int i = 0; i < img.nx(); i++)
    {
      vector<complex<double>> v;
      fftimg.value(i, j, v);
      vector<float> iv;
      iv.resize(v.size());
      for(size_t c = 0; c < v.size(); c++)
      {
        iv[c] = v[c].real();
      }
      img.set_value(i, j, iv);
    }
  }
}
