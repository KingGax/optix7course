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

#include "ParticleSimulation.h"

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"
#include <chrono>
#include "RSJparser.tcc"

/*! \namespace osc - Optix Siggraph Course */
namespace osc
{

  struct Experiment
  {
    Experiment(const Model *model, int _numParticles, int timesteps, float _delta, std::string name)
        : simulation(model)
    {
      loadedModel = model;
      numParticles = _numParticles;
      experimentName = name;
      numTimesteps = timesteps;
      delta = _delta;
    }

    virtual void run()
    {

      simulation.initialiseSimulation(numParticles, delta, maxParticleMultiplier);
      maxParticles = numParticles * maxParticleMultiplier;
      const bool checkParticleSection = false;
      const bool printParticles = false;
      int timeStep = 0;
      double simulationTime = 0;
      double verificationTime = 0;

      while (timeStep < numTimesteps)
      {
        // std::cout << "run timestep " << timeStep << "\n";
        auto pretrace = std::chrono::system_clock::now();
        simulation.runTimestep(timeStep);
        auto posttrace = std::chrono::system_clock::now();
        simulationTime += std::chrono::duration<double>(posttrace - pretrace).count();
        if (simulation.timestepFinished())
        {
          timeStep++;
          if (checkParticleSection && (timeStep % 1) == 0)
          {
            auto preVerif = std::chrono::system_clock::now();
            int currentParticleCount = simulation.getNumActiveParticles();
            float accuracy = simulation.getParticleSectionAccuracy(currentParticleCount);
            float fracEscaped = simulation.getParticleEscapePercentage(currentParticleCount);
            std::cout << "section tracking accuracy timestep " << timeStep - 1 << " --accuracy " << accuracy << " ---- num escaped " << fracEscaped << " particle count " << currentParticleCount << std::endl;
            auto postVerif = std::chrono::system_clock::now();
            verificationTime += std::chrono::duration<double>(postVerif - preVerif).count();
          }
          if (printParticles)
          {
            simulation.writeParticles(maxParticles, timeStep, experimentName);
          }
        }
      }
      float fracEscaped = simulation.getParticleEscapePercentage(simulation.getNumActiveParticles());

      std::cout << "done simulating " << numParticles << " start particles " << simulation.getNumActiveParticles() << " end particles with " << loadedModel->meshes[0]->index.size() << " tris "
                << "\n";
      std::cout << experimentName << " " << simulationTime << ",verif " << verificationTime << ",escaped " << fracEscaped << "\n";
    }

    const Model *loadedModel;
    vec2i fbSize;
    ParticleSimulation simulation;
    int numParticles = 10;
    float delta = 1;
    int numTimesteps = 10;
    const int maxParticleMultiplier = 2;
    int maxParticles;
    std::string experimentName = "";
  };

  /*! main entry point to this example - initially optix, print hello
    world, then exit */
  extern "C" int main(int ac, char **av)
  {
    try
    {
      std::string baseExperimentPath = "/nfs/home/zs18838/prime-owl/experiments/";
      std::string experimentPath = "";
      std::string defaultExperiment = "256p20-0.02.json";
      // std::string defaultExperiment = "128p5c.json";
      float overrideDT = 0;
      int overrideParticles = 0;
      int overrideCubeRecDepth = 0;
      int overrideTimesteps = 0;
      for (int i = 1; i < ac; i++)
      {
        const std::string arg = av[i];
        if (arg == "-exp")
        {
          experimentPath = baseExperimentPath + av[++i];
        }
        else if (arg == "-steps")
        {
          overrideTimesteps = std::stoi(av[++i]);
        }
        else if (arg == "-p")
        {
          overrideParticles = std::stoi(av[++i]);
        }
        else if (arg == "-cubes")
        {
          overrideCubeRecDepth = std::stoi(av[++i]);
        }
        else if (arg == "-dt")
        {
          overrideDT = std::stof(av[++i]);
        } else{
          std::cout << "unkown argument " + arg + "\n";
          throw;
        }
      }
      std::ifstream inFile;
      if (experimentPath == "")
      {
        std::cout << "No experiment specified with -exp flag, using default"
                  << "\n";
        experimentPath = baseExperimentPath + defaultExperiment;
      }
      std::cout << "Loading experiment at " + experimentPath << "\n";
      inFile.open(experimentPath); // open the input file

      std::stringstream strStream;
      strStream << inFile.rdbuf();       // read the file
      std::string str = strStream.str(); // str holds the content of the file

      RSJresource my_json(str);
      int numParticles = (overrideParticles != 0) ? overrideParticles : my_json["NumParticles"].as<int>(-1);
      int cubesPerAxis = (overrideCubeRecDepth != 0) ? overrideCubeRecDepth : my_json["NumCubes"].as<int>(-1);
      int timesteps = (overrideTimesteps != 0) ? overrideTimesteps : my_json["Timesteps"].as<int>(10);
      float delta = (overrideDT != 0) ? overrideDT : (float)my_json["Delta"].as<double>(1);

      std::cout << "Running " << numParticles << " particles" << std::endl;
      std::cout << "Creating cube with " << cubesPerAxis << " cubes per axis" << std::endl;
      if (numParticles <= 0 || cubesPerAxis <= 0)
      {
        std::cout << "invalid num of particles or cubes per axis in experiment\n";
        throw;
      }

      Model *model = loadOBJ(cubesPerAxis);
      std::cout << model->meshes[0]->index.size() << " triangles loaded\n";
      std::string experimentName = std::to_string(numParticles) + "particles-" + std::to_string(cubesPerAxis) + "cubes-" + std::to_string(timesteps) + "timestep-" + std::to_string(delta) + "delta";

      Experiment *experiment = new Experiment(model, numParticles, timesteps, delta, experimentName);
      experiment->run();
    }
    catch (std::runtime_error &e)
    {
      std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                << GDT_TERMINAL_DEFAULT << std::endl;
      std::cout << "Did you forget to copy sponza.obj and sponza.mtl into your optix7course/models directory?" << std::endl;
      exit(1);
    }
    return 0;
  }

} // ::osc
