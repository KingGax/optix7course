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

namespace osc
{

  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  // for this simple example, we have a single ray type
  enum
  {
    SURFACE_RAY_TYPE = 0,
    RAY_TYPE_COUNT
  };

  static __forceinline__ __device__ void *unpackPointer(uint32_t i0, uint32_t i1)
  {
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void *ptr = reinterpret_cast<void *>(uptr);
    return ptr;
  }

  static __forceinline__ __device__ void packPointer(void *ptr, uint32_t &i0, uint32_t &i1)
  {
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }

  template <typename T>
  static __forceinline__ __device__ T *getPRD()
  {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T *>(unpackPointer(u0, u1));
  }

  __device__ void calculatePhysics(Particle *p, const float gas_temp, const vec3f sectionAcceleration, const float dt, const int randomSeed)
  {
    vec3f gas_vel = sectionAcceleration;

    // ray.direction += (accel / particleData.mass) * dt;
    const vec3f v1 = p->vel;
    const vec3f relative_drop_vel = gas_vel - v1;                  // DUMMY_VAL Relative velocity between droplet and the fluid
    const float relative_drop_vel_mag = length(relative_drop_vel); // DUMMY_VAL Relative acceleration between the gas and liquid phase.

    // const float diameter = 1e-3;
    const float mass = 0.1;

    const float gas_density = 0.59; // DUMMY VAL
    const float particle_temp = 288.6;
    const float fuel_density = 724. * (1. - 1.8 * 0.000645 * (particle_temp - 288.6) - 0.090 * pow(particle_temp - 288.6, 2.) / pow(548. - 288.6, 2.));
    const float three_over_fourPI = 3 / (4 * M_PI);
    const float diameter = 2 * cbrtf(three_over_fourPI * (mass / fuel_density));
    const float kinematic_viscosity = 1.48e-5 * pow(gas_temp, 1.5) / (gas_temp + 110.4); // DUMMY_VAL
    const float reynolds = gas_density * relative_drop_vel_mag * diameter / kinematic_viscosity;

    const float droplet_frontal_area = M_PI * (diameter / 2.) * (diameter / 2.);

    // Drag coefficient
    const float drag_coefficient = (reynolds <= 1000.) ? 24 * (1. + 0.15 * pow(reynolds, 0.687)) / reynolds : 0.424;

    const vec3f drag_force = (drag_coefficient * reynolds * 0.5f * gas_density * relative_drop_vel_mag * droplet_frontal_area) * relative_drop_vel;
    const vec3f a1 = ((drag_force) / mass) * dt;
    const vec3f newVel = v1 + a1 * dt;
    p->vel = newVel;

    int launchIndex = optixGetLaunchIndex().x;
    bool newParticle = (((launchIndex + randomSeed) & 127) == 0);
    if (newParticle)
    {
      int writeAddress = atomicAdd(optixLaunchParams.activeParticleCount, 1);
      // printf("%d\n",optixLaunchParams.activeParticleCount[0]);
      if (writeAddress < optixLaunchParams.maxParticles)
      {
        // printf("%d %d\n",writeAddress,optixLaunchParams.maxParticles);
        //float particleVel = length(newVel);
        const vec3f multiplier = vec3f(0.8,1,0.8);
        const vec3f dir = dot(newVel,multiplier);
        optixLaunchParams.particles[writeAddress].section = p->section;
        optixLaunchParams.particles[writeAddress].pos = p->pos;
        optixLaunchParams.particles[writeAddress].vel = dir;
      }
    }
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

    const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData *)optixGetSbtDataPointer();
    int lastPrim = optixGetPrimitiveIndex();
    //const vec3i index = sbtData.index[lastPrim];
    //const vec3f &A = sbtData.vertex[index.x];
    //const vec3f &B = sbtData.vertex[index.y];
    //const vec3f &C = sbtData.vertex[index.z];
    //const vec3f N = normalize(cross(B - A, C - A));
    const vec2i neighs = sbtData.posNegNormalSections[lastPrim];
    // const bool boundary = (neighs[0] == -1 || neighs[1] == -1);
    Particle &p = *(Particle *)getPRD<Particle>();
    //float dotProd = dot(p.vel, N);
    bool isBackFaceHit = optixIsTriangleBackFaceHit();
    p.section = (isBackFaceHit) * neighs[1] + !(isBackFaceHit) * neighs[0];
    /*if (dotProd == 0)
    {
      printf("eww zero dot product");
    }*/
  }

  extern "C" __global__ void __anyhit__radiance()
  {
    printf("I SHOULD NOT APPEAR\n");
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

    // int lastPrim = optixGetPayload_4();
    /*const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData *)optixGetSbtDataPointer();

    Particle &p = *(Particle *)getPRD<Particle>();
    //int zeroIfFirstTrace = (optixGetPayload_3() + 1) & 1;
    //int oneIfFirstTrace = 1 - zeroIfFirstTrace;
    const float delta = optixLaunchParams.delta;
      p->pos = p->pos + p->vel * delta;
      //p.simPercent = 1;
      vec3f accel = vec3f(0, 0, 0);
      float temp = 288.6;
      if (p->section != -1)
      {
      //printf("no hit %d %d\n", p.section, optixGetLaunchIndex().x);
        accel = sbtData.sectionData[p->section].accel;
        temp = 288.6;
      }*/
  }

  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;

    // our per-ray data for this example. what we initialize it to
    // won't matter, since this value will be overwritten by either
    // the miss or hit program, anyway
    Particle *p = &optixLaunchParams.particles[ix];
    if (p->section != -1)
    {
      // the values we store the PRD pointer in:
      uint32_t u0, u1;
      packPointer(p, u0, u1);

      // generate ray direction
      const vec3f startPos = p->pos;
      const vec3f startVel = p->vel;
      const vec3f rayPos = startPos + startVel * optixLaunchParams.delta;
      const vec3f rayDir = -startVel;
      
      const float tmax = optixLaunchParams.delta;
      //const vec3f rayDir = 
      /*if(ix % 1000000 == 0){
        printf("%f\n", tmax);
      }*/
      const float eps = 0;
      optixTrace(optixLaunchParams.traversable,
                 rayPos,
                 rayDir,
                 eps,  // tmin
                 tmax, // tmax
                 0.0f, // rayTime
                 OptixVisibilityMask(255),
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
                 SURFACE_RAY_TYPE,              // SBT offset
                 RAY_TYPE_COUNT,                // SBT stride
                 SURFACE_RAY_TYPE,              // missSBTIndex
                 u0, u1);
      const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData *)optixGetSbtDataPointer();
      const float delta = optixLaunchParams.delta;
      p->pos = startPos + startVel * delta;
      // p.simPercent = 1;
      vec3f accel = vec3f(0, 0, 0);
      float temp = 288.6;
      const int newSection = p->section;
      if (newSection != -1)
      {
        // printf("no hit %d %d\n", p.section, optixGetLaunchIndex().x);
        accel = sbtData.sectionData[newSection].accel;
        temp = sbtData.sectionData[newSection].temp;
      }
      calculatePhysics(p, temp, accel, delta, optixLaunchParams.timestep);
    }
  }

} // ::osc
