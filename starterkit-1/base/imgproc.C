#include <cmath>
#include <vector>
#include <random>
#include "imgproc.h"

#include <OpenImageIO/imageio.h>
OIIO_NAMESPACE_USING
using namespace std;
using namespace img;

struct Color{
  float r;
  float g;
  float b;
};

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



ImgProc::ImgProc(const ImgProc& v) :
  Nx (1920),
  Ny (1080),
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


void ImgProc::brightnessUp()
{
   if(img_data == nullptr) { return; }
#pragma omp parallel for
   for(long i = 0; i < Nsize; i++)
   {
      img_data[i] = 2 * img_data[i];
   }
}

void ImgProc::brightnessDown()
{
   if(img_data == nullptr) { return; }
#pragma omp parallel for
   for(long i = 0; i < Nsize; i++)
   {
      img_data[i] = .4 * img_data[i];
   }
}

void ImgProc::biasUp()
{
   if(img_data == nullptr) { return; }
#pragma omp parallel for
   for(long i = 0; i < Nsize; i++)
   {
      img_data[i] = img_data[i] + .5;
   }
}

void ImgProc::biasDown()
{
   if(img_data == nullptr) { return; }
#pragma omp parallel for
   for(long i = 0; i < Nsize; i++)
   {
      img_data[i] = img_data[i] - .5;
   }
}

void ImgProc::gammaUp()
{
   if(img_data == nullptr) { return; }
#pragma omp parallel for
   for(long i = 0; i < Nsize; i++)
   {
      img_data[i] = pow(img_data[i], .5);
   }
}

void ImgProc::gammaDown()
{
   if(img_data == nullptr) { return; }
#pragma omp parallel for
   for(long i = 0; i < Nsize; i++)
   {
      img_data[i] = pow(img_data[i], 1.8);
   }
}

void ImgProc::grayscale()
{
   if(img_data == nullptr) { return; }
   float grayRes = 0.0;
   float redRes = 0.0;
   float greenRes = 0.0;
   float blueRes = 0.0;

#pragma omp parallel for
   for(long i = 0; i < Nsize; i++)
   {
      redRes = img_data[i] * 0.2126;
      greenRes = img_data[i+1] * 0.7152;
      blueRes = img_data[i+2] * 0.0722;
      grayRes = redRes + greenRes + blueRes;
      img_data[i] = grayRes;
      img_data[i+1] = grayRes;
      img_data[i+2] = grayRes;
   }
}

void ImgProc::quantize()
{
   float colorSteps = 5.0;

   if(img_data == nullptr) { return; }
#pragma omp parallel for
   for(long i = 0; i < Nsize; i++)
   {
      int tempVal = img_data[i] * colorSteps;
      img_data[i] = tempVal / colorSteps;
   }
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

void ImgProc::histogram(float max[], float min[])
{
  int N = 256;

  for(int i = 0; i < Nc; i++)
  {
    float histo[N] = {0.0};
    float pdf[N] = {0.0};
    float cdf[N] = {0.0};
    float deltaI = (max[i] - min[i])/N;

    int M = 0;
    for(int j = 0; j < Nsize; j+=Nc)
    {
      M = floor((img_data[i+j] - min[i])/deltaI);
      histo[M] += 1;
    }

    for(int k = 0; k < N; k++)
    {
      pdf[k] = histo[k] / (Nx * Ny);
    }
    cdf[0] = pdf[0];

    for(int l = 1; l < N; l++)
    {
      cdf[l] = cdf[l-1] + pdf[l];
    }

    for(int m = 0; m < Nsize; m+=Nc)
    {
      float Q = (img_data[m+i] - min[i])/deltaI;
      int q = floor(Q);
      float res = Q - q;

      if(q < N-1)
      {
        img_data[m+i] = (cdf[q] * (1 - res)) + (cdf[q+1] * res);
      }
      else if(q == N-1)
      {
        img_data[m+i] = cdf[q];
      }
    }
  }
}

void ImgProc::max(float result[])
{
  if(Nc == 1)
  {
    float tempMax = 0.0;
    for(int i = 0; i < Nsize; i+=Nc)
    {
      if(img_data[i] > tempMax) { tempMax = img_data[i]; }
    }
    result[0] = tempMax;
  }
  else if(Nc == 3)
  {
    float rMax = 0.0;
    float grMax = 0.0;
    float bMax = 0.0;
    for(int i = 0; i < Nsize; i+=Nc)
    {
      if(img_data[i] > rMax) { rMax = img_data[i]; }
      if(img_data[i+1] > grMax) { grMax = img_data[i+1]; }
      if(img_data[i+2] > bMax) { bMax = img_data[i+2]; }
    }
    result[0] = rMax;
    result[1] = grMax;
    result[2] = bMax;
  }
  else if(Nc == 5)
  {
    float maxZero = 0.0;
    float maxOne = 0.0;
    float maxTwo = 0.0;
    float maxThr = 0.0;
    float maxFour = 0.0;
    for(int i = 0; i < Nsize; i+=Nc)
    {
      if(img_data[i] > maxZero) { maxZero = img_data[i]; }
      if(img_data[i+1] > maxOne) { maxOne = img_data[i+1]; }
      if(img_data[i+2] > maxTwo) { maxTwo = img_data[i+2]; }
      if(img_data[i+3] > maxThr) { maxThr = img_data[i+3]; }
      if(img_data[i+4] > maxFour) { maxFour = img_data[i+4]; }
    }
    result[0] = maxZero;
    result[1] = maxOne;
    result[2] = maxTwo;
    result[3] = maxThr;
    result[4] = maxFour;
  }
}

void ImgProc::min(float result[])
{
  if(Nc == 1)
  {
    float tempMin = 1.0;
    for(int i = 0; i < Nsize; i+=Nc)
    {
      if(img_data[i] < tempMin) { tempMin = img_data[i]; }
    }
    result[0] = tempMin;
  }
  else if(Nc == 3)
  {
    float rMin = 1.0;
    float grMin = 1.0;
    float bMin = 1.0;
    for(int i = 0; i < Nsize; i+=Nc)
    {
      if(img_data[i] < rMin) { rMin = img_data[i]; }
      if(img_data[i+1] < grMin) { grMin = img_data[i+1]; }
      if(img_data[i+2] < bMin) { bMin = img_data[i+2]; }
    }
    result[0] = rMin;
    result[1] = grMin;
    result[2] = bMin;
  }
  else if(Nc == 5)
  {
    float minZero = 1.0;
    float minOne = 1.0;
    float minTwo = 1.0;
    float minThr = 1.0;
    float minFour = 1.0;
    for(int i = 0; i < Nsize; i+=Nc)
    {
      if(img_data[i] < minZero) { minZero = img_data[i]; }
      if(img_data[i+1] < minOne) { minOne = img_data[i+1]; }
      if(img_data[i+2] < minTwo) { minTwo = img_data[i+2]; }
      if(img_data[i+3] < minThr) { minThr = img_data[i+3]; }
      if(img_data[i+4] < minFour) { minFour = img_data[i+4]; }
    }
    result[0] = minZero;
    result[1] = minOne;
    result[2] = minTwo;
    result[3] = minThr;
    result[4] = minFour;
  }
}

void ImgProc::ave(float result[])
{
  if(Nc == 1)
  {
    float sum = 0.0;
    for(int i = 0; i < Nsize; i+=Nc)
    {
      sum += img_data[i];
    }
    sum /= Nsize;
    result[0] = sum;
  }
  else if(Nc == 3)
  {
    float redSum = 0.0;
    float greenSum = 0.0;
    float blueSum = 0.0;
    for(int i = 0; i < Nsize; i+=Nc)
    {
      redSum += img_data[i];
      greenSum += img_data[i+1];
      blueSum += img_data[i+2];
    }
    redSum /= Nsize;
    greenSum /= Nsize;
    blueSum /= Nsize;
    result[0] = redSum;
    result[1] = greenSum;
    result[2] = blueSum;
  }
  else if(Nc == 5)
  {
    float zeroSum = 0.0;
    float oneSum = 0.0;
    float twoSum = 0.0;
    float thrSum = 0.0;
    float fourSum = 0.0;
    for(int i = 0; i < Nsize; i+=Nc)
    {
      zeroSum += img_data[i];
      oneSum += img_data[i+1];
      twoSum += img_data[i+2];
      thrSum += img_data[i+3];
      fourSum += img_data[i+4];
    }
    zeroSum /= Nsize;
    oneSum /= Nsize;
    twoSum /= Nsize;
    thrSum /= Nsize;
    fourSum /= Nsize;
    result[0] = zeroSum;
    result[1] = oneSum;
    result[2] = twoSum;
    result[3] = thrSum;
    result[4] = fourSum;
  }
}

void ImgProc::stdDev(float stand[])
{
  if(Nc == 1)
  {
    float tempSum = 0.0;
    float tempAve[1] = {0.0};
    this->ave(tempAve);
    for(int i = 0; i < Nsize; i+=Nc)
    {
      tempSum += pow((img_data[i] - tempAve[0]), 2);
    }
    stand[0] = tempSum / (Nsize-1);
  }
  else if(Nc == 3)
  {
    float redSum = 0.0;
    float greenSum = 0.0;
    float blueSum = 0.0;
    float tempAve[3] = {0.0};
    this->ave(tempAve);
    for(int i = 0; i < Nsize; i+=Nc)
    {
      redSum += pow((img_data[i] - tempAve[0]), 2);
      greenSum += pow((img_data[i+1] - tempAve[1]), 2);
      blueSum += pow((img_data[i+2] - tempAve[2]), 2);
    }
    stand[0] = redSum / (Nsize-1);
    stand[1] = greenSum / (Nsize-1);
    stand[2] = blueSum / (Nsize-1);
  }
  else if(Nc == 5)
  {
    float zeroSum = 0.0;
    float oneSum = 0.0;
    float twoSum = 0.0;
    float thrSum = 0.0;
    float fourSum = 0.0;
    float tempAve[5] = {0.0};
    this->ave(tempAve);
    for(int i = 0; i < Nsize; i+=Nc)
    {
      zeroSum += pow((img_data[i] - tempAve[0]), 2);
      oneSum += pow((img_data[i+1] - tempAve[1]), 2);
      twoSum += pow((img_data[i+2] - tempAve[2]), 2);
      thrSum += pow((img_data[i+3] - tempAve[3]), 2);
      fourSum += pow((img_data[i+4] - tempAve[4]), 2);
    }
    stand[0] = zeroSum / (Nsize-1);
    stand[1] = oneSum / (Nsize-1);
    stand[2] = twoSum / (Nsize-1);
    stand[3] = thrSum / (Nsize-1);
    stand[4] = fourSum / (Nsize-1);
  }
}

void ImgProc::loadColorLUT(vector<Color> &color)
{
  Color royalPurp;
    royalPurp.r = 204/255.0;
    royalPurp.g = 0/255.0;
    royalPurp.b = 204/255.0;
  /*If the purple doesn't show up well for you, uncomment this and use red
  Color red;
    red.r = 255/255.0;
    red.g = 0/255.0;
    red.b = 0/255.0;*/
  Color pink;
    pink.r = 255/255.0;
    pink.g = 102/255.0;
    pink.b = 178/255.0;
  Color green;
    green.r = 102/255.0;
    green.g = 255/255.0;
    green.b = 102/255.0;
  Color blue;
    blue.r = 102/255.0;
    blue.g = 178/255.0;
    blue.b = 255/255.0;
  color.push_back(royalPurp); //have to change to red here if used
  color.push_back(pink);
  color.push_back(green);
  color.push_back(blue);
}

void ImgProc::LUT(float w, vector<Color> lut, vector<float>& newColor)
{
  if(w >= 0 && w <= 1)
  {
    int f = (int)floor(w * lut.size());
    newColor[0] = lut[f].r;
    newColor[1] = lut[f].g;
    newColor[2] = lut[f].b;
  }
  else
  {
    newColor[0] = 0.0;
    newColor[1] = 0.0;
    newColor[2] = 0.0;
  }
}

void ImgProc::hyper(Point& p)
{
  float theta = atan2(p.x, p.y);
  float newX = sin(theta) / sqrt(pow(p.x, 2) + pow(p.y, 2));
  float newY = sqrt(pow(p.x, 2) + pow(p.y, 2)) * cos(theta);
  p.x = newX;
  p.y = newY;
}

//r = sqrt(x^2 + y^2)
void ImgProc::polar(Point& p)
{
  float theta = atan2(p.x, p.y);
  float newX = theta / 3.14;
  float newY = sqrt((pow(p.x, 2) + pow(p.y, 2))) - 1;
  p.x = newX;
  p.y = newY;
}

void ImgProc::eyefish(Point& p)
{
  float value = 2 / sqrt((pow(p.x, 2) + pow(p.y, 2))) + 1;
  p.x = value * p.x;
  p.y = value * p.y;
}

void ImgProc::bubble(Point& p)
{
  float value = 4 / (pow(sqrt((pow(p.x, 2) + pow(p.y, 2))), 2) + 4);
  p.x = value * p.x;
  p.y = value * p.y;
}

void ImgProc::fractalFlame(ImgProc& img)
{
  Point point;
  point.x = 2 * drand48() - 1;
  point.y = 2 * drand48() - 1;
  vector<float> functionWeights(4, 0.0);
  vector<Color> color;
  float weight = drand48();

  functionWeights[0] = 0.0;
  functionWeights[1] = 0.5;
  functionWeights[2] = 0.75;
  functionWeights[3] = 1.0;
  loadColorLUT(color);

  for(int count = 0; count < 1500000; count++)
  {
    unsigned int choice = rand() % 4;
    if(choice == 0)
    {
      hyper(point);
    }
    else if(choice == 1)
    {
      polar(point);
    }
    else if(choice == 2)
    {
      eyefish(point);
    }
    else if(choice == 3)
    {
      bubble(point);
    }

    weight = (weight + functionWeights[choice]) / 2.0;

    if(count > 20)
    {
      if(point.x <= 1.0 && point.x >= -1.0 && point.y <= 1.0 && point.y >= -1.0)
      {
        float currX = point.x + 1.0;
        float currY = point.y + 1.0;
        currX *= 0.5 * img.nx();
        currY *= 0.5 * img.ny();

        int tempX, tempY;
        tempX = currX;
        if(tempX < img.nx())
        {
          tempY = currY;
          if(tempY < img.ny())
          {
            vector<float> newColor(4, 0.0);
            LUT(weight, color, newColor);

            vector<float> imgColor;
            value(tempX, tempY, imgColor);

            for(unsigned int j = 0; j < imgColor.size()-1; j++)
            {
              imgColor[j] = imgColor[j] * imgColor[imgColor.size()-1];
              imgColor[j] = (imgColor[j] + newColor[j]) / (imgColor[imgColor.size()-1] + 1);
            }
            imgColor[imgColor.size()-1] += 1;

            set_value(tempX, tempY, imgColor);
          }
        }
      }
    }
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
