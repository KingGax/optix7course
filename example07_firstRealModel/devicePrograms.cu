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

#include <optix_device.h>

#include "LaunchParams.h"

using namespace osc;

namespace osc {
  
  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  // for this simple example, we have a single ray type
  enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };
  
  static __forceinline__ __device__
  void *unpackPointer( uint32_t i0, uint32_t i1 )
  {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
  }

  static __forceinline__ __device__
  void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
  {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T *getPRD()
  { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }
  
  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------
  
  extern "C" __global__ void __closesthit__radiance()
  {
    /*const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    // compute normal:
    const int   primID = optixGetPrimitiveIndex();
    const vec3i index  = sbtData.index[primID];
    const vec3f &A     = sbtData.vertex[index.x];
    const vec3f &B     = sbtData.vertex[index.y];
    const vec3f &C     = sbtData.vertex[index.z];
    const vec3f Ng     = normalize(cross(B-A,C-A));

    const vec3f rayDir = optixGetWorldRayDirection();
    const float cosDN  = 0.2f + .8f*fabsf(dot(rayDir,Ng));
    vec3f &prd = *(vec3f*)getPRD<vec3f>();
    //prd = cosDN * sbtData.color;
    prd = cosDN * vec3f(1*sbtData.boundary,0,1*(!sbtData.boundary));
    //prd[0] = 1*sbtData.boundary; 
    //prd[1] = 1*sbtData.boundary; 
    //prd[2] = 1; */
    }
  
  extern "C" __global__ void __anyhit__radiance()
  { 
    //printf("we in anyhit\n");
    float currentTmax = __uint_as_float(optixGetPayload_2());
    float t = optixGetRayTmax();
    //printf("is t > currentTMax %d\n", (t>currentTmax));
    //printf("t %d %f ", optixGetPrimitiveIndex(), t);
    if(t > currentTmax){
      
      //printf("getting sbt data\n");
      optixSetPayload_2(__float_as_uint(t));
      const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
      //printf("got sbt data\n");
      // compute normal:
     // printf("tri stuff\n");
      const int primID = optixGetPrimitiveIndex();
      optixSetPayload_4(primID);
      
      const vec3i index  = sbtData.index[primID];
      const vec3f &A     = sbtData.vertex[index.x];
      const vec3f &B     = sbtData.vertex[index.y];
      const vec3f &C     = sbtData.vertex[index.z];
      const vec3f N      = normalize(cross(B-A,C-A));
      const vec2i neighs = sbtData.posNegNormalSections[primID];
      const bool boundary = (neighs[0] == -1 || neighs[1] == -1);
      int firstTraceMultiplier = (optixGetPayload_3() + 1) & 1; //if first trace, multiplier is 0
      //printf("tri stuff done\n");
      Particle & p =  *(Particle*)getPRD<Particle>();
      //printf("particle\n");
      if(boundary){
        
        p.simPercent = firstTraceMultiplier * p.simPercent + t;
        p.pos += p.vel * t;
        vec3f newDir = p.vel - 2.0f*dot(p.vel, N)*N;
        p.vel = newDir;
        printf("HIT BOUNDARY\n");
        //printf("%f , %f, %f position %f , %f, %f \n", p.vel.x,p.vel.y,p.vel.z, p.pos.x,p.pos.y,p.pos.z);
        optixLaunchParams.bounced[0] = 1;
        p.section = (neighs[0] != -1) * neighs[0] + (neighs[1] != -1) * neighs[1];
        //printf("written to bounced\n");
        optixTerminateRay();
      } else {
       
        //printf("update section %d\n",p.section);
      }
      //const vec3f rayDir = optixGetWorldRayDirection();
      //const float cosDN  = 0.2f + .8f*fabsf(dot(rayDir,N));
      //vec3f &prd = *(vec3f*)getPRD<vec3f>();
      //prd = cosDN * vec3f(1*boundary,0,1*(!boundary));
    }
    //printf("irnmoring intersection\n");
    optixIgnoreIntersection();
    //printf("we done anyhit\n");
  }


  
  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT
  // ------------------------------------------------------------------------------
  
  extern "C" __global__ void __miss__radiance()
  {
    //printf("in miss\n");
    /*float currentTmax = __uint_as_float(optixGetPayload_2());
    if(currentTmax > 1e10){
      vec3f &prd = *(vec3f*)getPRD<vec3f>();
      // set to constant white as background color
      prd = vec3f(1.f);
    }*/
    //printf("we in miss\n");
    int lastPrim = optixGetPayload_4();
    if(lastPrim  != INT_MAX){
      const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
      const vec3i index  = sbtData.index[lastPrim];
      const vec3f &A     = sbtData.vertex[index.x];
      const vec3f &B     = sbtData.vertex[index.y];
      const vec3f &C     = sbtData.vertex[index.z];
      const vec3f &spare = sbtData.normals[lastPrim];
      const vec3f N      = normalize(cross(B-A,C-A));
      const vec2i neighs = sbtData.posNegNormalSections[lastPrim];
      const bool boundary = (neighs[0] == -1 || neighs[1] == -1);
      /*printf("\n%f %f %f \n", A[0],A[1],A[2]);
      printf("%f %f %f \n", B[0],B[1],B[2]);
      printf("%f %f %f \n", C[0],C[1],C[2]);
      printf("%f %f %f \n", spare[0],spare[1],spare[2]);
      printf("options %d %d \n", neighs[0], neighs[1]);*/
      Particle & p =  *(Particle*)getPRD<Particle>();
      if(!boundary){
        float dotProd = dot(p.vel,N); 
        p.section = (dotProd < 0) * neighs[1] + !(dotProd < 0) * neighs[0];
        if(dotProd == 0){
          printf("eww zero dot product");
        }
      }
    }
    
    Particle & p =  *(Particle*)getPRD<Particle>();
    int zeroIfFirstTrace = (optixGetPayload_3() + 1) & 1; 
    int oneIfFirstTrace = 1 - zeroIfFirstTrace;
    p.pos = p.pos + p.vel * oneIfFirstTrace + zeroIfFirstTrace * (1-p.simPercent) * p.vel;
    p.simPercent = 1;
    //printf("%f , %f, %f, %d \n", p.vel.x,p.vel.y,p.vel.z);
    //printf("%f , %f, %f, %d \n", p.pos.x,p.pos.y,p.pos.z, p.section);
    /*printf("%f , %f, %f, %d \n", p->pos.x,p->pos.y,p->pos.z, p->section);
    p->pos += p->vel;
    p->simPercent = 1;
    printf("%f , %f, %f \n", p->vel.x,p->vel.y,p->vel.z);
    printf("%f , %f, %f, %d \n", p->pos.x,p->pos.y,p->pos.z, p->section);*/
    //printf("we done miss\n");
  }

  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    //const int iy = optixGetLaunchIndex().y;

    //const auto &camera = optixLaunchParams.camera;

    // our per-ray data for this example. what we initialize it to
    // won't matter, since this value will be overwritten by either
    // the miss or hit program, anyway
    Particle * p = &optixLaunchParams.particles[ix];
    //printf("we in raygen %d\n", ix);
    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( p, u0, u1 );

    // normalized screen plane position, in [0,1]^2
    //const vec2f screen(vec2f(ix+.5f,iy+.5f)
    //                   / vec2f(optixLaunchParams.frame.size));
    
    // generate ray direction
    vec3f pos = p->pos;
    vec3f rayDir = p->vel;
    //printf("we in raygen\n");
    //normalize(camera.direction
    //                         + (screen.x - 0.5f) * camera.horizontal
    //                         + (screen.y - 0.5f) * camera.vertical);
    uint32_t tmaxPayload = __float_as_uint(0); //float max  as an integer
    uint32_t firstTraceFlag = (int)optixLaunchParams.firstTrace;
    uint32_t lastPrimPayload = INT_MAX;
    //printf("launbching trace\n");
    //printf("start trace: vel %f , %f, %f position %f , %f, %f \n", p->vel.x,p->vel.y,p->vel.z, p->pos.x,p->pos.y,p->pos.z);
    //printf("%f \n", rayDir.x);
    //printf("%f \n", rayDir.y);
    //printf("%f \n", rayDir.z);
    float tmax = optixLaunchParams.firstTrace * 1 + (!optixLaunchParams.firstTrace) * (1-p->simPercent);
    float eps = 5e-4;
    //printf("tmax %f  %d", tmax, (int)optixLaunchParams.firstTrace);
    optixTrace(optixLaunchParams.traversable,
               pos,
               rayDir,
               eps,    // tmin
               tmax,  // tmax
               0.0f,   // rayTime
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,//OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,             // SBT offset
               RAY_TYPE_COUNT,               // SBT stride
               SURFACE_RAY_TYPE,             // missSBTIndex 
               u0, u1 , tmaxPayload, firstTraceFlag, lastPrimPayload);
    //printf("trace launched\n");

    //const int r = int(255.99f*pixelColorPRD.x);
    //const int g = int(255.99f*pixelColorPRD.y);
    //const int b = int(255.99f*pixelColorPRD.z);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    /*const uint32_t rgba = 0xff000000
      | (r<<0) | (g<<8) | (b<<16);*/

    // and write to frame buffer ...
    //const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;
    //optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
  }
  
} // ::osc
