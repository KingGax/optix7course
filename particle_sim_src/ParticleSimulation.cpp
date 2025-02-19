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
#include "VisitWriter.hpp"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

/*! \namespace osc - Optix Siggraph Course */
namespace osc
{

  extern "C" char embedded_ptx_code[];

  /*! SBT record for a raygen program */
  struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
  {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    TriangleMeshSBTData data;
  };

  /*! SBT record for a miss program */
  struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
  {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    TriangleMeshSBTData data;
  };

  /*! SBT record for a hitgroup program */
  struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
  {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    TriangleMeshSBTData data;
  };

  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  ParticleSimulation::ParticleSimulation(const Model *model)
      : model(model)
  {
    initOptix();

    std::cout << "#osc: creating optix context ..." << std::endl;
    createContext();

    std::cout << "#osc: setting up module ..." << std::endl;
    createModule();

    std::cout << "#osc: creating raygen programs ..." << std::endl;
    createRaygenPrograms();
    std::cout << "#osc: creating miss programs ..." << std::endl;
    createMissPrograms();
    std::cout << "#osc: creating hitgroup programs ..." << std::endl;
    createHitgroupPrograms();

    launchParams.traversable = buildAccel();

    std::cout << "#osc: setting up optix pipeline ..." << std::endl;
    createPipeline();

    std::cout << "#osc: building SBT ..." << std::endl;
    buildSBT();

    launchParamsBuffer.alloc(sizeof(launchParams));
    std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

    std::cout << GDT_TERMINAL_GREEN;
    std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
    std::cout << GDT_TERMINAL_DEFAULT;
  }

  OptixTraversableHandle ParticleSimulation::buildAccel()
  {
    PING;
    PRINT(model->meshes.size());

    vertexBuffer.resize(model->meshes.size());
    indexBuffer.resize(model->meshes.size());
    posNegNormalSectionsBuffer.resize(model->meshes.size());
    normalBufffer.resize(model->meshes.size());
    sectionDataBuffer.resize(model->meshes.size());

    OptixTraversableHandle asHandle{0};

    // ==================================================================
    // triangle inputs
    // ==================================================================
    std::vector<OptixBuildInput> triangleInput(model->meshes.size());
    std::vector<CUdeviceptr> d_vertices(model->meshes.size());
    std::vector<CUdeviceptr> d_indices(model->meshes.size());
    std::vector<CUdeviceptr> d_normalSections(model->meshes.size());
    std::vector<CUdeviceptr> d_normal(model->meshes.size());
    std::vector<CUdeviceptr> d_sectionData(model->meshes.size());

    std::vector<uint32_t> triangleInputFlags(model->meshes.size());

    for (int meshID = 0; meshID < model->meshes.size(); meshID++)
    {
      // upload the model to the device: the builder
      TriangleMesh &mesh = *model->meshes[meshID];
      vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
      indexBuffer[meshID].alloc_and_upload(mesh.index);
      posNegNormalSectionsBuffer[meshID].alloc_and_upload(mesh.posNegNormalNeighbours);
      normalBufffer[meshID].alloc_and_upload(mesh.normal);
      sectionDataBuffer[meshID].alloc_and_upload(mesh.sectionData);
      triangleInput[meshID] = {};
      triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

      // create local variables, because we need a *pointer* to the
      // device pointers
      d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
      d_indices[meshID] = indexBuffer[meshID].d_pointer();
      d_normalSections[meshID] = posNegNormalSectionsBuffer[meshID].d_pointer();

      triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
      triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
      triangleInput[meshID].triangleArray.numVertices = (int)mesh.vertex.size();
      triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

      triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(vec3i);
      triangleInput[meshID].triangleArray.numIndexTriplets = (int)mesh.index.size();
      triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

      triangleInputFlags[meshID] = 0;

      // in this example we have one SBT entry, and no per-primitive
      // materials:
      triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
      triangleInput[meshID].triangleArray.numSbtRecords = 1;
      triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
      triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
      triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }
    // ==================================================================
    // BLAS setup
    // ==================================================================

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
                                             &accelOptions,
                                             triangleInput.data(),
                                             (int)model->meshes.size(), // num_build_inputs
                                             &blasBufferSizes));

    // ==================================================================
    // prepare compaction
    // ==================================================================

    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================

    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(optixContext,
                                /* stream */ 0,
                                &accelOptions,
                                triangleInput.data(),
                                (int)model->meshes.size(),
                                tempBuffer.d_pointer(),
                                tempBuffer.sizeInBytes,

                                outputBuffer.d_pointer(),
                                outputBuffer.sizeInBytes,

                                &asHandle,

                                &emitDesc, 1));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
                                  /*stream:*/ 0,
                                  asHandle,
                                  asBuffer.d_pointer(),
                                  asBuffer.sizeInBytes,
                                  &asHandle));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();

    return asHandle;
  }

  /*! helper function that initializes optix and checks for errors */
  void ParticleSimulation::initOptix()
  {
    std::cout << "#osc: initializing optix..." << std::endl;

    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw std::runtime_error("#osc: no CUDA capable devices found!");
    std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK(optixInit());
    std::cout << GDT_TERMINAL_GREEN
              << "#osc: successfully initialized optix... yay!"
              << GDT_TERMINAL_DEFAULT << std::endl;
  }

  static void context_log_cb(unsigned int level,
                             const char *tag,
                             const char *message,
                             void *)
  {
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
  }

  /*! creates and configures a optix device context (in this simple
    example, only for the primary GPU device) */
  void ParticleSimulation::createContext()
  {
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "#osc: running on device: " << deviceProps.name << std::endl;

    CUresult cuRes = cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS)
      fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
  }

  /*! creates the module that contains all the programs we are going
    to use. in this simple example, we use a single module from a
    single .cu file, using a single embedded ptx string */
  void ParticleSimulation::createModule()
  {
    moduleCompileOptions.maxRegisterCount = 50;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    //moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL ;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth = 0;

    const std::string ptxCode = embedded_ptx_code;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                         &moduleCompileOptions,
                                         &pipelineCompileOptions,
                                         ptxCode.c_str(),
                                         ptxCode.size(),
                                         log, &sizeof_log,
                                         &module));
    if (sizeof_log > 1)
      PRINT(log);
  }

  /*! does all setup for the raygen program(s) we are going to use */
  void ParticleSimulation::createRaygenPrograms()
  {
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log, &sizeof_log,
                                        &raygenPGs[0]));
    if (sizeof_log > 1)
      PRINT(log);
  }

  /*! does all setup for the miss program(s) we are going to use */
  void ParticleSimulation::createMissPrograms()
  {
    // we do a single ray gen program in this example:
    missPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log, &sizeof_log,
                                        &missPGs[0]));
    if (sizeof_log > 1)
      PRINT(log);
  }

  /*! does all setup for the hitgroup program(s) we are going to use */
  void ParticleSimulation::createHitgroupPrograms()
  {
    // for this simple example, we set up a single hit group
    hitgroupPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log, &sizeof_log,
                                        &hitgroupPGs[0]));
    if (sizeof_log > 1)
      PRINT(log);
  }

  bool ParticleSimulation::timestepFinished()
  {
    return (bounced[0] == 0);
  }

  /*! assembles the full pipeline of all programs */
  void ParticleSimulation::createPipeline()
  {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
      programGroups.push_back(pg);
    for (auto pg : missPGs)
      programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
      programGroups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(optixContext,
                                    &pipelineCompileOptions,
                                    &pipelineLinkOptions,
                                    programGroups.data(),
                                    (int)programGroups.size(),
                                    log, &sizeof_log,
                                    &pipeline));
    if (sizeof_log > 1)
      PRINT(log);

    OPTIX_CHECK(optixPipelineSetStackSize(/* [in] The pipeline to configure the stack size for */
                                          pipeline,
                                          /* [in] The direct stack size requirement for direct
                                             callables invoked from IS or AH. */
                                          2 * 1024,
                                          /* [in] The direct stack size requirement for direct
                                             callables invoked from RG, MS, or CH.  */
                                          2 * 1024,
                                          /* [in] The continuation stack requirement. */
                                          2 * 1024,
                                          /* [in] The maximum depth of a traversable graph
                                             passed to trace. */
                                          1));
    if (sizeof_log > 1)
      PRINT(log);
  }

  /*! constructs the shader binding table */
  void ParticleSimulation::buildSBT()
  {
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (int i = 0; i < raygenPGs.size(); i++)
    {
      RaygenRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
      rec.data.vertex = (vec3f *)vertexBuffer[0].d_pointer(); // change if more models
      rec.data.index = (vec3i *)indexBuffer[0].d_pointer();
      rec.data.posNegNormalSections = (vec2i *)posNegNormalSectionsBuffer[0].d_pointer();
      rec.data.normals = (vec3f *)normalBufffer[0].d_pointer();
      rec.data.sectionData = (SectionData *)sectionDataBuffer[0].d_pointer();
      raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i = 0; i < missPGs.size(); i++)
    {
      MissRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
      // rec.data = nullptr; /* for now ... */
      rec.data.vertex = (vec3f *)vertexBuffer[0].d_pointer(); // change if more models
      rec.data.index = (vec3i *)indexBuffer[0].d_pointer();
      rec.data.posNegNormalSections = (vec2i *)posNegNormalSectionsBuffer[0].d_pointer();
      rec.data.normals = (vec3f *)normalBufffer[0].d_pointer();
      rec.data.sectionData = (SectionData *)sectionDataBuffer[0].d_pointer();
      missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    int numObjects = (int)model->meshes.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID = 0; meshID < numObjects; meshID++)
    {
      HitgroupRecord rec;
      // all meshes use the same code, so all same hit group
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &rec));
      rec.data.color = model->meshes[meshID]->diffuse;
      rec.data.vertex = (vec3f *)vertexBuffer[meshID].d_pointer();
      rec.data.index = (vec3i *)indexBuffer[meshID].d_pointer();
      rec.data.posNegNormalSections = (vec2i *)posNegNormalSectionsBuffer[meshID].d_pointer();
      rec.data.sectionData = (SectionData *)sectionDataBuffer[meshID].d_pointer();
      hitgroupRecords.push_back(rec);
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
  }

  int ParticleSimulation::getNumActiveParticles()
  {
    int activeParticles[1];
    cudaMemcpy(activeParticles, launchParams.activeParticleCount, sizeof(int), cudaMemcpyDeviceToHost);
    return min(activeParticles[0], launchParams.maxParticles);
  }

  /*! render one frame */
  void ParticleSimulation::runTimestep(int timestep)
  {
    // sanity check: make sure we launch only after first resize is
    // already done:
    int init = 0;
    int activeParticles[1];
    cudaMemcpy(activeParticles, launchParams.activeParticleCount, sizeof(int), cudaMemcpyDeviceToHost);
    if (activeParticles[0] > launchParams.maxParticles)
    {
      activeParticles[0] = launchParams.maxParticles;
      cudaMemcpy(launchParams.activeParticleCount, activeParticles, sizeof(int), cudaMemcpyHostToDevice);
    }
    // cudaMemcpy(launchParams.bounced,&init,sizeof(int),cudaMemcpyDefault);
    launchParams.timestep = timestep;

    launchParamsBuffer.upload(&launchParams, 1);

    // std::cout << "launch timee" << "\n";
    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                            pipeline, stream,
                            /*! parameters and SBT */
                            launchParamsBuffer.d_pointer(),
                            launchParamsBuffer.sizeInBytes,
                            &sbt,
                            /*! dimensions of the launch: */
                            activeParticles[0],
                            1,
                            1));
    // sync - wait for particles to update position
    CUDA_SYNC_CHECK();
    /*bouncedBuffer.download(bounced,1);
    if(timestepFinished()){
      launchParams.firstTrace = true;
    } else {
      launchParams.firstTrace = false;
    }*/
    // std::cout << "sync check done" << "\n";
  }

  inline vec3f randomVector()
  {
    vec3f newVector = vec3f();
    for (int i = 0; i < 3; i++)
    {
      newVector[i] = ((((float)rand()) / (float)RAND_MAX) - 0.5) * 2;
    }
    return newVector;
  }

  float ScTP(vec3f a, vec3f b, vec3f c)
  {
    return dot(a, cross(b, c));
  }

  vec4f bary_tet(const vec3f a, const vec3f b, const vec3f c, const vec3f d, const vec3f p)
  {
    vec3f vap = p - a;
    vec3f vbp = p - b;

    vec3f vab = b - a;
    vec3f vac = c - a;
    vec3f vad = d - a;

    vec3f vbc = c - b;
    vec3f vbd = d - b;
    // ScTP computes the scalar triple product
    float va6 = ScTP(vbp, vbd, vbc);
    float vb6 = ScTP(vap, vac, vad);
    float vc6 = ScTP(vap, vad, vab);
    float vd6 = ScTP(vap, vab, vac);
    float v6 = 1 / ScTP(vab, vac, vad);
    return vec4f(va6 * v6, vb6 * v6, vc6 * v6, vd6 * v6);
  }

  float ParticleSimulation::getParticleEscapePercentage(const int numParticles)
  {
    particleBuffer.download(particles, numParticles);
    int numEscaped = 0;
    for (int j = 0; j < numParticles; j++)
    {
      Particle p = particles[j];
      vec3f pos = p.pos;
      if (abs(p.pos[0] - 1) > 1 || abs(p.pos[1] - 1) > 1 || abs(p.pos[2] - 1) > 1)
      {
        numEscaped++;
      }
    }
    return (float)numEscaped / (float)numParticles;
  }

  void ParticleSimulation::writeParticles(const int numParticles, const int timestep, std::string name)
  {
    particleBuffer.download(particles, numParticles);
    VisitWriter vw = *new VisitWriter();
    vw.write_particles(name, timestep, particles, numParticles);
  }

  float ParticleSimulation::getParticleSectionAccuracy(const int numParticles)
  {
    const bool calculateCorrectSection = false;
    int realSection[4] = {-5, -5, -5, -5};
    particleBuffer.download(particles, numParticles);
    std::cout << "particle 0 at " << particles[0].pos << "\n";
    int numCorrect = 0;
    int numCandidateSections = 0;
    for (int j = 0; j < numParticles; j++)
    {
      Particle p = particles[j];
      vec3f pos = p.pos;
      if (p.section != -1)
      {
        Tetra tet = model->tetras[p.section];
        vec4f baryTetVector = bary_tet(tet.A, tet.B, tet.C, tet.D, pos);
        if (baryTetVector[0] < 0 || baryTetVector[1] < 0 || baryTetVector[2] < 0 || baryTetVector[3] < 0)
        {

        }
        else
        {
          numCorrect++;
          realSection[0] = p.section;
        }
        if (calculateCorrectSection)
        {
          for (int t = 0; t < model->tetras.size(); t++)
          {
            Tetra tempTet = model->tetras[t];

            vec4f tempBaryTetVector = bary_tet(tempTet.A, tempTet.B, tempTet.C, tempTet.D, pos);
            if (!(tempBaryTetVector[0] < 0 || tempBaryTetVector[1] < 0 || tempBaryTetVector[2] < 0 || tempBaryTetVector[3] < 0))
            {
              realSection[numCandidateSections] = t;
              numCandidateSections++;
            }
            else
            {
              // std::cout << tempBaryTetVector << " " << std::endl;
            }
          }
          std::cout << " section " << p.section << " pos " << p.pos << " real sections " << realSection[0] << " " << realSection[1] << " " << realSection[2] << " " << realSection[3] << std::endl;
        }
      } else{
        if (abs(p.pos[0] - 1) > 1 || abs(p.pos[1] - 1) > 1 || abs(p.pos[2] - 1) > 1)
        {
          numCorrect++;
        }
      }
    }
    return (float)numCorrect / (float)numParticles;
  }

  void ParticleSimulation::initialiseSimulation(const int numParticles, const float delta, int maxParticleMultiplier)
  {

    std::cout << "initialising particles "
              << "\n";
    // resize our cuda frame buffer
    int maxParticles = numParticles * maxParticleMultiplier;
    particles = new Particle[maxParticles];
    launchParams.launchSize = numParticles;
    particleBuffer.resize(maxParticles * sizeof(Particle));
    launchParams.particles = (Particle *)particleBuffer.d_pointer();
    bounced = new int[1];
    bounced[0] = 0;
    activeParticleCount = new int[1];
    activeParticleCount[0] = numParticles;
    activeParticleBuffer.resize(sizeof(int));
    bouncedBuffer.resize(sizeof(int));
    launchParams.firstTrace = true;
    launchParams.delta = delta;
    launchParams.bounced = (int *)bouncedBuffer.d_pointer();
    launchParams.activeParticleCount = (int *)activeParticleBuffer.d_pointer();
    int init = 0;
    cudaMemcpy(launchParams.bounced, &init, sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(launchParams.activeParticleCount, &activeParticleCount[0], sizeof(int), cudaMemcpyDefault);
    float maxEdgeLength = 0;
    for(int i = 0; i < 5; i++){
      Tetra t = model->tetras[i];
      float currentMax = max({length(t.A-t.B),length(t.A-t.C),length(t.A-t.D),length(t.B-t.C),length(t.B-t.D),length(t.C-t.D)});
      if(currentMax > maxEdgeLength){
        maxEdgeLength = currentMax;
      }
    }
    launchParams.maxEdgeLength = maxEdgeLength;
    std::cout << "max edge length " << maxEdgeLength << "\n";
    std::srand(42);
    vec3f particleOrigin = vec3f(1, 1, 1);
    float particleSpeedMultiplier = 0.4;
    float particleoffsetMultiplier = 0;
    int startingTetra = model->tetras.size() / 2;
    std::cout << "begin  " << startingTetra << " \n";
    vec3f startingPos = 0.25f * (model->tetras[startingTetra].A + model->tetras[startingTetra].B + model->tetras[startingTetra].C + model->tetras[startingTetra].D);
    particleOrigin = startingPos;
    std::cout << "particle origin " << particleOrigin << "\n";
    for (int i = 0; i < maxParticles; i++)
    {
      if (i < numParticles)
      {
        vec3f vel = randomVector(); // vec3f(0.5,0.4876,0);
        vel.x = abs(vel.x);
        vel.y = abs(vel.y);
        vel.z = abs(vel.z);
        vec3f posOffset = randomVector();
        Particle *p = new Particle();
        p->vel = vec3f(vel.x * particleSpeedMultiplier, vel.y * particleSpeedMultiplier, vel.z * particleSpeedMultiplier);
        p->pos = vec3f(particleOrigin.x + posOffset.x * particleoffsetMultiplier, particleOrigin.y + posOffset.y * particleoffsetMultiplier, particleOrigin.z + posOffset.z * particleoffsetMultiplier);
        p->simPercent = 0;
        p->section = startingTetra;

        particles[i] = *p;
      }
      else
      {
        Particle *p = new Particle();
        p->vel = vec3f(0, 0, 0);
        p->section = -1;
        particles[i] = *p;
      }

      if ((i % (maxParticles / 10)) == 0)
      {
        std::cout << "init " << ((float)i / (float)maxParticles) * 100 << "%\n";
      }
      // launchParams.particles.sections[i] = 4;
    }
    cudaMemcpy((&launchParams.particles[0]), particles, sizeof(Particle) * maxParticles, cudaMemcpyDefault);
    launchParams.maxParticles = maxParticles;
    // std::cout << launchParams.particles[0].pos << "\n";

    std::cout << "set particles "
              << "\n";
  }

} // ::osc
