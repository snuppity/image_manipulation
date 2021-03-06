//------------------------------------------------
//
//  img_paint
//
//
//-------------------------------------------------




#include <cmath>
#include <omp.h>
#include "imgproc.h"
#include "fftimgproc.h"
#include "fftanalysis.h"
#include "CmdLineFind.h"
#include <vector>



#include <GL/gl.h>   // OpenGL itself.
#include <GL/glu.h>  // GLU support library.
#include <GL/glut.h> // GLUT support library.


#include <iostream>
#include <stack>


using namespace std;
using namespace img;

ImgProc image;




void setNbCores( int nb )
{
   omp_set_num_threads( nb );
}

void cbMotion( int x, int y )
{
}

void cbMouse( int button, int state, int x, int y )
{
}

void cbDisplay( void )
{
   glClear(GL_COLOR_BUFFER_BIT );
   glDrawPixels( image.nx(), image.ny(), GL_RGB, GL_FLOAT, image.raw() );
   glutSwapBuffers();
}

void cbIdle()
{
   glutPostRedisplay();
}

void cbOnKeyboard( unsigned char key, int x, int y )
{
   switch (key)
   {
      case 'c':
      {
    	 image.compliment();
    	 cout << "Compliment\n";
    	 break;
     }
      case 't':
      {
        //ImgProc input;
        ImgProc twoChanOut;
        float w1, w2;
        w1 = 0.7059; //180
        w2 = 0.5961; //152
        twoChanEstimate(image, 0, 1, w1, w2, twoChanOut);
        swap(image, twoChanOut);
        cout << "Two Channel Estimation\n";
        break;
      }
      case 'r':
      {
        image.rmsContrast();
        cout << "RMS contrast\n";
        break;
      }
      case 'o':
      {
        image.output();
        cout << "Output Image\n";
        break;
      }
   }
}

void PrintUsage()
{
   cout << "img_paint keyboard choices\n";
   cout << "c         compliment\n";
   cout << "t         Two Channel Estimation\n";
   cout << "r         RMS Contrast\n";
   cout << "o         Output Image\n";
}


int main(int argc, char** argv)
{
   lux::CmdLineFind clf( argc, argv );

   setNbCores(8);

   string imagename = clf.find("-image", "foliage.jpg", "Image to drive color");
   string whatsthis = clf.find("-stuff", "", "Image to drive color");

   cout << whatsthis << endl;

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
