// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "SampleRenderer.h"

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  struct SampleWindow
  {
    SampleWindow(const std::string &title,
                 const Model *model,
                 const Camera &camera,
                 const float worldScale)
      : sample(model)
    {
      sample.setCamera(camera);
      loadedModel = model;
    }
    
    virtual void run() 
    {
      const int particleNum = 1;
      const int numTimesteps = 10;
      const vec2i fbSize(vec2i(1200,1024));
      sample.resize(fbSize);
      sample.setParticleNum(particleNum);
      Camera camera = { /*from*/vec3f(-10.0f, 0, 5.0),
                        /* at */loadedModel->bounds.center()-vec3f(0,0,0),
                        /* up */vec3f(0.f,1.f,0.f) };
      sample.setCamera(camera);
      const bool checkParticleSection = true;
      int timeStep = 0;
      while(timeStep < numTimesteps){
        //std::cout << "run timestep " << timeStep << "\n";
        sample.render();
        if(sample.timestepFinished()){
          timeStep++;
          if(checkParticleSection){
            float accuracy = sample.getParticleSectionAccuracy(particleNum);
            std::cout << "section tracking accuracy timestep " << timeStep-1 << " " << accuracy << std::endl;
          }
        }
      }
      std::cout << "done simulating " << "\n";
    }
    
    
    virtual void resize(const vec2i &newSize) 
    {
      fbSize = newSize;
      sample.resize(newSize);
      pixels.resize(newSize.x*newSize.y);
    }

    const Model*                loadedModel; 
    vec2i                 fbSize;
    GLuint                fbTexture {0};
    SampleRenderer        sample;
    std::vector<uint32_t> pixels;
  };
  
  
  /*! main entry point to this example - initially optix, print hello
    world, then exit */
  extern "C" int main(int ac, char **av)
  {
    try {
      Model *model = loadOBJ(
#ifdef _WIN32
      // on windows, visual studio creates _two_ levels of build dir
      // (x86/Release)
      "../../models/sponza.obj"
#else
      // on linux, common practice is to have ONE level of build dir
      // (say, <project>/build/)...
      "../models/2.71k-split.obj"
#endif
                             );
      //Camera camera = { /*from*/vec3f(-1293.07f, 154.681f, -0.7304f),
      //                  /* at */model->bounds.center()-vec3f(0,400,0),
      //                  /* up */vec3f(0.f,1.f,0.f) };
      
      Camera camera = { /*from*/vec3f(-10.0f, 0, 0),
                        /* at */model->bounds.center()-vec3f(0,0,0),
                        /* up */vec3f(0.f,1.f,0.f) };
      // something approximating the scale of the world, so the
      // camera knows how much to move for any given user interaction:
      const float worldScale = length(model->bounds.span());

      SampleWindow *window = new SampleWindow("Optix 7 Course Example",
                                              model,camera,worldScale);
      window->run();
      
    } catch (std::runtime_error& e) {
      std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                << GDT_TERMINAL_DEFAULT << std::endl;
	  std::cout << "Did you forget to copy sponza.obj and sponza.mtl into your optix7course/models directory?" << std::endl;
	  exit(1);
    }
    return 0;
  }
  
} // ::osc
