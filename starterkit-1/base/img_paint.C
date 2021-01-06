//------------------------------------------------
//
//  img_paint
//
//
//-------------------------------------------------
#include <cmath>
#include <omp.h>
#include "imgproc.h"
#include "CmdLineFind.h"
#include <vector>
#include <OpenImageIO/imageio.h>



#include <GL/gl.h>   // OpenGL itself.
#include <GL/glu.h>  // GLU support library.
#include <GL/glut.h> // GLUT support library.


#include <iostream>
#include <stack>


using namespace std;
using namespace img;
OIIO_NAMESPACE_USING;

ImgProc image;




void setNbCores( int nb )
{
   omp_set_num_threads( nb );
}

void cbMotion( int x, int y )
{
   //?????
}

void cbMouse( int button, int state, int x, int y )
{
   //?????
}

void cbDisplay( void )
{
   glClear(GL_COLOR_BUFFER_BIT );
   glDrawPixels( image.nx(), image.ny(), GL_RGBA, GL_FLOAT, image.raw() );
   glutSwapBuffers();
}

void cbIdle()
{
   glutPostRedisplay();
}

void cbOnKeyboard( unsigned char key, int x, int y )
{
   int tempChannel = image.depth();
   float histMax[tempChannel] = {0.0};
   float histMin[tempChannel] = {0.0};
   float tempMax[tempChannel] = {0.0};
   float tempMin[tempChannel] = {0.0};
   float tempAve[tempChannel] = {0.0};
   float tempSD[tempChannel] = {0.0};

   switch (key)
   {
      case 'V':
         image.brightnessUp();
         cout << "Brightness Up\n";
         break;
      case 'v':
         image.brightnessDown();
         cout << "Brightness Down\n";
         break;
      case 'B':
         image.biasUp();
         cout << "Bias Up\n";
         break;
      case 'b':
         image.biasDown();
         cout << "Bias Down\n";
         break;
      case 'G':
         image.gammaUp();
         cout << "Gamma Up\n";
         break;
      case 'g':
         image.gammaDown();
         cout << "Gamma Down\n";
         break;
      case 'w':
         image.grayscale();
         cout << "Gray Scale\n";
         break;
      case 'q':
         image.quantize();
         cout << "Quantize\n";
         break;
      case 'C':
         image.rmsContrast();
         cout << "RMS Contrast\n";
         break;
      case 'c':
        image.compliment();
        cout << "Compliment\n";
        break;
      case 'H':
        image.max(histMax);
        image.min(histMin);
        image.histogram(histMax, histMin);
        cout << "Histogram Equilization\n";
        break;
      case 'S':
        image.max(tempMax);
        image.min(tempMin);
        image.ave(tempAve);
        image.stdDev(tempSD);
        for(int i = 0; i < image.depth(); i++)
        {
          cout << "Channel " << i << " max = " << tempMax[i] << "\n";
          cout << "Channel " << i << " min = " << tempMin[i] << "\n";
          cout << "Channel " << i << " average = " << tempAve[i] << "\n";
          cout << "Channel " << i << " standard deviation = " << tempSD[i] << "\n";
        }
        break;
      case 'o':
        image.output();
        cout << "Output\n";
        break;
      case 'N':
        image.clear(1920, 1080, 4);
        image.fractalFlame(image);
        cout << "Fractal Fun!\n";
        glutPostRedisplay();
        break;
   }
}

void PrintUsage()
{
   cout << "img_paint keyboard choices\n";
   cout << "V         brightness up\n";
   cout << "v         brightness down\n";
   cout << "B         bias up\n";
   cout << "b         bias down\n";
   cout << "G         gamma up\n";
   cout << "g         gamma down\n";
   cout << "w         grayscale\n";
   cout << "q         quantize\n";
   cout << "C         rms contrast\n";
   cout << "c         compliment\n";
   cout << "o         output\n";
   cout << "H         histogram\n";
   cout << "S         stats\n";
   cout << "N         Fractal Fun\n";
}


int main(int argc, char** argv)
{
   lux::CmdLineFind clf( argc, argv );

   setNbCores(8);

   string imagename = clf.find("-image", "micah.jpg", "Image to drive color");

   clf.usage("-h");
   clf.printFinds();
   PrintUsage();

   image.load(imagename);


   // GLUT routines
   glutInit(&argc, argv);

   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
   glutInitWindowSize( image.nx(), image.ny() );

   // Open a window
   char title[] = "img_paint";
   glutCreateWindow( title );

   glClearColor( 1,1,1,1 );

   glutDisplayFunc(&cbDisplay);
   glutIdleFunc(&cbIdle);
   glutKeyboardFunc(&cbOnKeyboard);
   glutMouseFunc( &cbMouse );
   glutMotionFunc( &cbMotion );

   glutMainLoop();
   return 1;
};
