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

#include "Model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"
//std
#include <set>

namespace std {
  inline bool operator<(const tinyobj::index_t &a,
                        const tinyobj::index_t &b)
  {
    if (a.vertex_index < b.vertex_index) return true;
    if (a.vertex_index > b.vertex_index) return false;
    
    if (a.normal_index < b.normal_index) return true;
    if (a.normal_index > b.normal_index) return false;
    
    if (a.texcoord_index < b.texcoord_index) return true;
    if (a.texcoord_index > b.texcoord_index) return false;
    
    return false;
  }
}

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
  


  /*! find vertex with given position, normal, texcoord, and return
      its vertex ID, or, if it doesn't exit, add it to the mesh, and
      its just-created index */
  int addVertex(TriangleMesh *mesh,
                tinyobj::attrib_t &attributes,
                const tinyobj::index_t &idx,
                std::map<tinyobj::index_t,int> &knownVertices)
  {
    if (knownVertices.find(idx) != knownVertices.end())
      return knownVertices[idx];

    const vec3f *vertex_array   = (const vec3f*)attributes.vertices.data();
    const vec3f *normal_array   = (const vec3f*)attributes.normals.data();
    const vec2f *texcoord_array = (const vec2f*)attributes.texcoords.data();
    
    int newID = mesh->vertex.size();
    knownVertices[idx] = newID;

    mesh->vertex.push_back(vertex_array[idx.vertex_index]);
    if (idx.normal_index >= 0) {
      while (mesh->normal.size() < mesh->vertex.size())
        mesh->normal.push_back(normal_array[idx.normal_index]);
    }
    if (idx.texcoord_index >= 0) {
      while (mesh->texcoord.size() < mesh->vertex.size())
        mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
    }

    // just for sanity's sake:
    if (mesh->texcoord.size() > 0)
      mesh->texcoord.resize(mesh->vertex.size());
    // just for sanity's sake:
    if (mesh->normal.size() > 0)
      mesh->normal.resize(mesh->vertex.size());
    
    return newID;
  }

      const int FOUR_C_THREE = 4;
    const int TETRAS_PER_CUBE = 5;
    const vec4i indexPermutations[FOUR_C_THREE] = {
        {0, 1, 2, 3},
        {0, 1, 3, 2},
        {0, 2, 3, 1},
        {1, 2, 3, 0}};

    int intPow(int x, int p)
  {
    int i = 1;
    for (int j = 1; j <= p; j++)
      i *= x;
    return i;
  }

  void addNeighbours(std::vector<Tetra> &  tetras, int numCubes, int cubesPerRow){
    int backCube = (-1)*TETRAS_PER_CUBE;
    int frontCube = (1)*TETRAS_PER_CUBE;
    int leftCube = (cubesPerRow)*TETRAS_PER_CUBE;
    int rightCube = (-cubesPerRow)*TETRAS_PER_CUBE;
    int bottomCube = (-cubesPerRow*cubesPerRow)*TETRAS_PER_CUBE;
    int topCube = (cubesPerRow*cubesPerRow)*TETRAS_PER_CUBE;
    const int NUM_POSSIBLE_NEIGHBOURS = 35;
    const int possibleNeighbours[NUM_POSSIBLE_NEIGHBOURS] = {
      0,1,2,3,4,
      backCube,backCube+1,backCube+2,backCube+3,backCube+4,
      frontCube,frontCube+1,frontCube+2,frontCube+3,frontCube+4,
      leftCube,leftCube+1,leftCube+2,leftCube+3,leftCube+4,
      rightCube,rightCube+1,rightCube+2,rightCube+3,rightCube+4,
      bottomCube,bottomCube+1,bottomCube+2,bottomCube+3,bottomCube+4,
      topCube,topCube+1,topCube+2,topCube+3,topCube+4
    };
    for (int i = 0; i < numCubes * TETRAS_PER_CUBE; i++)
    {
      int cubeNum = i - (i % TETRAS_PER_CUBE);
      
      for (int t = 0; t < FOUR_C_THREE; t++) //for each 2d triangle in tetra
      {
        for (int j = 0; j < NUM_POSSIBLE_NEIGHBOURS; j++)
        {
          int tetraIndex = possibleNeighbours[j] + cubeNum;
          int sharedIndesNum = 0;
          std::vector<int> sharedIndes;
          if ((tetraIndex >= 0) && (tetraIndex < numCubes * TETRAS_PER_CUBE)&& (tetraIndex != i))
          {
            for (int v = 0; v < 3; v++) //for each vertice 
            {
              int currentIndice = tetras[i].indes[indexPermutations[t][v]];
              if (currentIndice == tetras[tetraIndex].indes[0])
              {
                sharedIndesNum++;
                sharedIndes.push_back(currentIndice);
              }
              if (currentIndice == tetras[tetraIndex].indes[1])
              {
                sharedIndesNum++;
                sharedIndes.push_back(currentIndice);
              }
              if (currentIndice == tetras[tetraIndex].indes[2])
              {
                sharedIndesNum++;
                sharedIndes.push_back(currentIndice);
              }
              if (currentIndice == tetras[tetraIndex].indes[3])
              {
                sharedIndesNum++;
                sharedIndes.push_back(currentIndice);
              }
            }
            if (sharedIndesNum == 3)
            {
              tetras[i].neighbours.push_back(&tetras[tetraIndex]);
              tetras[i].neighbourIndicesAndID.push_back(vec4i(sharedIndes[0], sharedIndes[1], sharedIndes[2], tetras[tetraIndex].sectionID));
              //std::cout << i << " neighbour " << j << "\n";
            }
            else if (sharedIndesNum > 3)
            {
              std::cout << "MORE THAN 3 SHARED INDES??"
                        << "\n";
            }
          }
        }
      }
    }
  }

  
  int getTriangleNeighbourID(Tetra currentTetra, int triangleID){
    int neighbourMatch = -1;
    for (int n = 0; n < currentTetra.neighbours.size(); n++)
            { // for each neighbour check
              int indiceMatches = 0;
              for (int j = 0; j < 3; j++)
              {
                int currentIndice = currentTetra.indes[indexPermutations[triangleID][j]];
                vec4i neighbourIndices = currentTetra.neighbourIndicesAndID.at(n);
                if (currentIndice == neighbourIndices[0] || currentIndice == neighbourIndices[1] || currentIndice == neighbourIndices[2])
                {
                  indiceMatches++;
                }
              }
              if (indiceMatches == 3)
              {
                neighbourMatch = currentTetra.neighbours[n]->sectionID;
                break;
              }
              else if (indiceMatches > 3)
              {
                std::cout << ">3 indices??" << std::endl;
              }
            }
            return neighbourMatch;

  }

  void addTriangle(TriangleMesh *mesh, vec3i indes, vec3f normal, int posNeighbour, int negNeighbour, bool boundary){
    mesh->normal.push_back(normal);
    mesh->index.push_back(indes);
    mesh->posNegNormalNeighbours.push_back(vec2i(posNeighbour,negNeighbour));
    mesh->boundaries.push_back(boundary);
  }

  void addTetraTriangles(vec3f vertices[], Model* model, std::vector<Tetra> tetras){
    bool constructed = false;
    int numTetras = tetras.size();
    std::vector<int> constructionIndexQueue;
    constructionIndexQueue.push_back(0);
    std::vector<bool> constructedTetras(numTetras, false);

    
    while (!constructed)
    {
      Tetra currentTetra = tetras[constructionIndexQueue.back()];
      constructionIndexQueue.pop_back();
      if (!constructedTetras.at(currentTetra.sectionID))
      {
        if (currentTetra.boundary) // if a collision boundary leave space for non boundary
        {
          //std::cout << " boundary " << currentTetra.sectionID << std::endl;
          for (int i = 0; i < FOUR_C_THREE; i++)
          { // for each triangle in the tetra, check if indices shared with neighbours
            int neighbourMatch = getTriangleNeighbourID(currentTetra, i);
            if (neighbourMatch == -1 || (tetras[neighbourMatch].boundary && !constructedTetras.at(currentTetra.sectionID))) //if no neigh or matching another non-constructed boundary
            { // add if not a matching triangle with neighbour
              //std::cout << "add tri " << std::endl;
              vec3f A = vertices[currentTetra.indes[indexPermutations[i][0]]];
              vec3f B = vertices[currentTetra.indes[indexPermutations[i][1]]];
              vec3f C = vertices[currentTetra.indes[indexPermutations[i][2]]];
              vec3f sparePoint = vertices[currentTetra.indes[indexPermutations[i][2]]];

              bool boundary = (neighbourMatch == -1);
              vec3i indes = vec3i(currentTetra.indes[indexPermutations[i][0]],currentTetra.indes[indexPermutations[i][1]],currentTetra.indes[indexPermutations[i][2]]);
              int materialID = currentTetra.materialID;

              const vec3f N = normalize(cross(B - A, C - A));
              float dotProd = dot(N, sparePoint);
              
              int posDotNormalSection = (dotProd > 0) ? currentTetra.sectionID : -1;
              int negDotNormalSection = (dotProd > 0) ? -1 : currentTetra.sectionID;
              //std::cout << " add tri for section " << currentTetra.sectionID << std::endl;
              // std::cout << i << " " << N << std::endl;
              //triangles.push_back({A, B, C, N, N, N, materialID, boundary, posDotNormalSection, negDotNormalSection});
              addTriangle(model->meshes[0],indes,N,posDotNormalSection,negDotNormalSection,boundary);
            }
          }
        }
        else // if not a collision boundary, add all triangles - need to change for general case
        {
          //std::cout << " not boundary " << currentTetra.sectionID << std::endl;
          if (currentTetra.neighbours.size() < 4)
          {
            std::cout << "non boundary tetra has less than 4 neighbours?? " << std::endl;
            throw;
          }
          for (int i = 0; i < FOUR_C_THREE; i++)
          {
            int neighbourID = getTriangleNeighbourID(currentTetra, i);
            if(neighbourID != -1){
              if(!constructedTetras[neighbourID] || tetras[neighbourID].boundary){
                vec3f A = vertices[currentTetra.indes[indexPermutations[i][0]]];
                vec3f B = vertices[currentTetra.indes[indexPermutations[i][1]]];
                vec3f C = vertices[currentTetra.indes[indexPermutations[i][2]]];
                vec3f sparePoint = vertices[currentTetra.indes[indexPermutations[i][3]]];
                bool boundary = currentTetra.boundary;
                int materialID = currentTetra.materialID;
                //std::cout << "JHERE " << currentTetra.neighbours[i]->indes << " " << currentTetra.indes << "\n";

                // int connectedSection = currentTetra.neighbours[i]->
                vec3i indes = vec3i(currentTetra.indes[indexPermutations[i][0]],currentTetra.indes[indexPermutations[i][1]],currentTetra.indes[indexPermutations[i][2]]);
                const vec3f N = normalize(cross(B - A, C - A));
                float dotProd = dot(N, sparePoint);
                int posDotNormalSection = (dotProd > 0) ? currentTetra.sectionID : neighbourID;
                int negDotNormalSection = (dotProd > 0) ? neighbourID : currentTetra.sectionID;
                //std::cout << currentTetra.sectionID << " <> " << neighbourID << std::endl;
                //triangles.push_back({A, B, C, N, N, N, materialID, boundary, posDotNormalSection, negDotNormalSection});
                addTriangle(model->meshes[0],indes,N,posDotNormalSection,negDotNormalSection,boundary);
              }
            } else{
              std::cout << "non boundary triangle has no neighbour? " << std::endl;
              throw;
            }
            //std::cout << " add tri for section " << currentTetra.sectionID << std::endl;
            
          }
        }
      }

      constructedTetras.at(currentTetra.sectionID) = true;
      for (int i = 0; i < constructedTetras.size(); i++)
      {
        //std::cout << " constructed " << i << " " << constructedTetras.at(i) << std::endl;
      }
      for (int i = 0; i < currentTetra.neighbours.size(); i++)
      {
        int neighbourID = currentTetra.neighbours.at(i)->sectionID;
        //std::cout << " neighbour " << neighbourID << " " << constructedTetras.at(i) << std::endl;
        if (!constructedTetras.at(neighbourID))
        {
          constructionIndexQueue.push_back(neighbourID);
        }
        else
        {
          //std::cout << " already constructed " << neighbourID << " " << constructedTetras.at(neighbourID) << std::endl;
        }
      }
      if (constructionIndexQueue.size() <= 0)
      {

        constructed = true;
      }
      //std::cout << " tetras in queue " << constructionIndexQueue.size() << std::endl;
    }
  }

  

  Model* addTetraCube(const vec3f center, const float size,
                           int material,
                           int recDepth)
  {
    static const vec3f outerVertices[8] =
        {
            {0, 0, 0},
            {+1.f * size, 0, 0},
            {+1.f * size, +1.f * size, 0},
            {0, +1.f * size, 0},
            {0, 0, +1.f * size},
            {+1.f * size, 0, +1.f * size},
            {+1.f * size, +1.f * size, +1.f * size},
            {0, +1.f * size, +1.f * size},
        };


    int cubesPerAxis = recDepth;
    int numCubes = intPow(cubesPerAxis, 3);
    int verticesPerAxis = cubesPerAxis + 1;
    int verticesPerAxisSquared = verticesPerAxis * verticesPerAxis;
    
    
    static const vec4i ninetyDegreeCubeOffsets[TETRAS_PER_CUBE] =
        {
            {0, 1, verticesPerAxis, verticesPerAxisSquared},
            {1, verticesPerAxisSquared, verticesPerAxisSquared + 1, verticesPerAxisSquared + verticesPerAxis + 1},
            {verticesPerAxis, verticesPerAxisSquared, verticesPerAxisSquared + verticesPerAxis, verticesPerAxisSquared + verticesPerAxis + 1},
            {1, verticesPerAxis, verticesPerAxis + 1, verticesPerAxisSquared + verticesPerAxis + 1},
            {1, verticesPerAxis, verticesPerAxisSquared, verticesPerAxisSquared + verticesPerAxis + 1}};

    static const vec4i zeroDegreeCubeOffsets[TETRAS_PER_CUBE] =
        {
            {0, verticesPerAxisSquared, verticesPerAxisSquared + 1, verticesPerAxisSquared + verticesPerAxis},
            {0, 1, verticesPerAxis + 1, verticesPerAxisSquared + 1},
            {0, verticesPerAxis, verticesPerAxis + 1, verticesPerAxisSquared + verticesPerAxis},
            {verticesPerAxis + 1, verticesPerAxisSquared + 1, verticesPerAxisSquared + verticesPerAxis, verticesPerAxisSquared + verticesPerAxis + 1},
            {0, verticesPerAxis + 1, verticesPerAxisSquared + 1, verticesPerAxisSquared + verticesPerAxis}};
    
    std::vector<Tetra> tetras;
    Model *model = new Model;
    TriangleMesh *cubeMesh = new TriangleMesh;
    
    std::cout << "init vertices array size " << verticesPerAxis*verticesPerAxis*verticesPerAxis << std::endl;
    vec3f vertices[verticesPerAxis*verticesPerAxis*verticesPerAxis];
    const int NUM_VERTICES = 8;
    
    for (int y = 0; y < verticesPerAxis; y++)
    { // initialise xyz vertice array
      for (int z = 0; z < verticesPerAxis; z++)
      {
        for (int x = 0; x < verticesPerAxis; x++)
        {
          float xfrac = (float)x / (float)(verticesPerAxis - 1);
          float yfrac = (float)y / (float)(verticesPerAxis - 1);
          float zfrac = (float)z / (float)(verticesPerAxis - 1);
          int currentVertice = x + verticesPerAxis*z + verticesPerAxis*verticesPerAxis*y;
          vertices[currentVertice] = vec3f(xfrac * size, yfrac * size, zfrac * size);
          cubeMesh->vertex.push_back(vec3f(xfrac * size, yfrac * size, zfrac * size));
        }
      }
    }
    for (int y = 0; y < cubesPerAxis; y++)
    { // now initialise tetras
      for (int z = 0; z < cubesPerAxis; z++)
      {
        for (int x = 0; x < cubesPerAxis; x++)
        {
          int startingVertice = x + verticesPerAxis*z + verticesPerAxis*verticesPerAxis*y;
          bool rotated90 = ((x + y + z) % 2) == 0;
          for(int t = 0; t < TETRAS_PER_CUBE; t++){
            Tetra *newTetra = new Tetra();
            if(rotated90){
              newTetra->indes = vec4i(startingVertice+ninetyDegreeCubeOffsets[t][0],startingVertice+ninetyDegreeCubeOffsets[t][1]
              ,startingVertice+ninetyDegreeCubeOffsets[t][2],startingVertice+ninetyDegreeCubeOffsets[t][3]);
            } else{
              newTetra->indes = vec4i(startingVertice+zeroDegreeCubeOffsets[t][0],startingVertice+zeroDegreeCubeOffsets[t][1]
              ,startingVertice+zeroDegreeCubeOffsets[t][2],startingVertice+zeroDegreeCubeOffsets[t][3]);
            }
            
            newTetra->A = vertices[newTetra->indes[0]];
            newTetra->B = vertices[newTetra->indes[1]];
            newTetra->C = vertices[newTetra->indes[2]];
            newTetra->D = vertices[newTetra->indes[3]];
            newTetra->sectionID = t + TETRAS_PER_CUBE * (x + z * cubesPerAxis + y * cubesPerAxis * cubesPerAxis);
            tetras.push_back(*newTetra);
          }
        }
      }
    }
    std::cout << "added all tetras" << std::endl;
    // now calculate neighbours
    addNeighbours(tetras, numCubes,cubesPerAxis);
    std::cout << "added neighbours" << std::endl;
    for(int i = 0; i < tetras.size(); i++){
      if(tetras[i].neighbours.size() == 4){
        tetras[i].boundary = false;
      } else if(tetras[i].neighbours.size() > 4 || tetras[i].neighbours.size() < 1){
        std::cout << "MORE THAN 4 OR LESS THAN 1 NEIGHBOUR" << std::endl;
        std::cout << i << " " << tetras[i].neighbours.size() << std::endl;
        throw;
      } else{
        tetras[i].boundary = true;
      }
      //std::cout << i << " indes " << tetras[i].indes << "boundary " << tetras[i].boundary <<"\n";
    }
    model->meshes.push_back(cubeMesh);
    addTetraTriangles(vertices,model,tetras);
    std::cout << "added triangles" << std::endl;
    return model;
  }
  
  Model *loadOBJ(const std::string &objFile)
  {

    /*const std::string mtlDir
      = objFile.substr(0,objFile.rfind('/')+1);
    PRINT(mtlDir);
    
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = "";

    bool readOK
      = tinyobj::LoadObj(&attributes,
                         &shapes,
                         &materials,
                         &err,
                         &err,
						 objFile.c_str(),
                         mtlDir.c_str(),
                         /* triangulate *///true);
    /*if (!readOK) {
      throw std::runtime_error("Could not read OBJ model from "+objFile+":"+mtlDir+" : "+err);
    }

    if (materials.empty())
      throw std::runtime_error("could not parse materials ...");

    std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
    for (int shapeID=0;shapeID<(int)shapes.size();shapeID++) {
      tinyobj::shape_t &shape = shapes[shapeID];

      std::set<int> materialIDs;
      for (auto faceMatID : shape.mesh.material_ids)
        materialIDs.insert(faceMatID);
      
      std::map<tinyobj::index_t,int> knownVertices;
      
      for (int materialID : materialIDs) {
        TriangleMesh *mesh = new TriangleMesh;
        
        for (int faceID=0;faceID<shape.mesh.material_ids.size();faceID++) {
          if (shape.mesh.material_ids[faceID] != materialID) continue;
          tinyobj::index_t idx0 = shape.mesh.indices[3*faceID+0];
          tinyobj::index_t idx1 = shape.mesh.indices[3*faceID+1];
          tinyobj::index_t idx2 = shape.mesh.indices[3*faceID+2];
          
          vec3i idx(addVertex(mesh, attributes, idx0, knownVertices),
                    addVertex(mesh, attributes, idx1, knownVertices),
                    addVertex(mesh, attributes, idx2, knownVertices));
          mesh->index.push_back(idx);
          mesh->diffuse = (const vec3f&)materials[materialID].diffuse;
          mesh->diffuse = gdt::randomColor(materialID);
          
        }

        if(shape.name.compare("boundary") == 0){
            mesh->boundary = true;
          } else{
            mesh->boundary = false;
          }

        if (mesh->vertex.empty())
          delete mesh;
        else
          model->meshes.push_back(mesh);
      }
    }*/


    Model* model = addTetraCube(0,2,1,6);
    // of course, you should be using tbb::parallel_for for stuff
    // like this:
    for (auto mesh : model->meshes)
      for (auto vtx : mesh->vertex)
        model->bounds.extend(vtx);

    std::cout << "created a total of " << model->meshes.size() << " meshes" << std::endl;
    return model;
  }
}
