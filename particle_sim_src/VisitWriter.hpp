#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include "LaunchParams.h"


#define VTK_TETRA 10
#define TETRA_SIZE 4

namespace osc 
{   
    using namespace std;

    class VisitWriter 
    {

        public:

            const uint64_t num_particles = 0;
            Particle *particles;

            VisitWriter()
            {  }


            template < typename Type > std::string to_str (const Type & t)
            {
                std::ostringstream os;
                os << t;
                return os.str ();
            }

            std::string print_vec(vec3f vec){
                return " " + to_str(vec.x) + " " +   to_str(vec.y) + " " + to_str(vec.z);
            }


            void write_particles(string filename, int id, Particle *particles, int numParticles)
            {
                // Print VTK Header
                ofstream vtk_file;

                vtk_file.open ("out/"+filename+"_particle_timestep"+to_string(id)+".vtk");
                vtk_file << "# vtk DataFile Version 3.0 " << endl;
                vtk_file << "MiniCOMBUST " << endl;
                vtk_file << "ASCII " << endl;
                vtk_file << "DATASET POLYDATA " << endl;

                
                //TODO: Allow different datatypes
                //Print point data
                vec3f nullPos = vec3f(0,0,0);
                vtk_file << endl << "POINTS " << numParticles << " float"  << endl;
                for(int p = 0; p < numParticles; p++)
                {
                    const int data_per_line = 10;
                    if (p % data_per_line == 0)  vtk_file << endl;
                    else             vtk_file << "  ";
                    if(particles[p].section != -1){
                        vtk_file << print_vec(particles[p].pos) << "\t";
                    } else{
                        vtk_file << print_vec(nullPos) << "\t";
                    }
                    
                }
                vtk_file << endl << endl;
                vtk_file.close();
            }

    }; // class VisitWriter

}   // namespace minicombust::visit 