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

#pragma once

#include "gdt/math/AffineSpace.h"
#include <vector>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
  using namespace gdt;

      struct Triangle {
      vec3f A, B, C;
      vec3f Na,Nb,Nc;
      int materialID;
      bool boundary;
      int posDotNormalSection;
      int negDotNormalSection;
    };

    struct Tetra {
      vec4i indes;
      vec3f A, B, C, D;
      int materialID;
      bool boundary;
      int sectionID;
      std::vector<Tetra*> neighbours;
      std::vector<vec4i> neighbourIndicesAndID;
    };
  
  /*! a simple indexed triangle mesh that our sample renderer will
      render */
  struct TriangleMesh {
    std::vector<vec3f> vertex;
    std::vector<vec3f> normal;
    std::vector<vec2f> texcoord;
    std::vector<vec3i> index;
    std::vector<vec2i> posNegNormalNeighbours;
    std::vector<int> sectionID;
    std::vector<vec3f> accel;
    std::vector<bool> boundaries;
    // material data:
    vec3f              diffuse;
    bool               boundary;
  };
  
  struct Model {
    ~Model()
    { for (auto mesh : meshes) delete mesh; }
    
    std::vector<TriangleMesh *> meshes;
    //! bounding box of all vertices in the model
    box3f bounds;
    std::vector<Tetra> tetras;
  };


  Model *loadOBJ(const std::string &objFile);
}
