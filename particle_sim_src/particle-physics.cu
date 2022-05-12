__global__ void calculatePhysics(ParticleLaunchState particles, Scene world, const float dt, const int max_particles)
{

    // printf("hii");
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int threadid = threadIdx.x;
    int newParticle = 0;
    owl::common::vec3f particle_dir;
    float particle_vel;
    PartialParticle &particleData = particles.particleData[index];
    if (particleData.sectionID != -1)
    {
        Ray &ray = particles.rays[index]; // ray stores pos and vel
        vec3f gas_vel;
        if (particleData.sectionID == -1)
        {
            gas_vel = vec3f(0, 0, 0);
        }
        else
        {
            gas_vel = world.sectionAccelerations[particleData.sectionID];
        }
        // ray.direction += (accel / particleData.mass) * dt;
        vec3f v1 = ray.direction;
        const vec3f relative_drop_vel = gas_vel - v1;                  // DUMMY_VAL Relative velocity between droplet and the fluid
        const float relative_drop_vel_mag = length(relative_drop_vel); // DUMMY_VAL Relative acceleration between the gas and liquid phase.

        const float temp = 288.6;
        // const float diameter = 1e-3;
        const float mass = 0.1;

        const float gas_density = 0.59; // DUMMY VAL

        const float fuel_density = 724. * (1. - 1.8 * 0.000645 * (temp - 288.6) - 0.090 * pow(temp - 288.6, 2.) / pow(548. - 288.6, 2.));
        const float three_over_fourPI = 3 / (4 * M_PI);
        const float diameter = 2 * cbrtf(three_over_fourPI * (mass / fuel_density));
        // const float omega = 1.;                                                                            // DUMMY_VAL What is this?
        const float kinematic_viscosity = 1.48e-5 * pow(temp, 1.5) / (temp + 110.4); // DUMMY_VAL
        const float reynolds = gas_density * relative_drop_vel_mag * diameter / kinematic_viscosity;

        const float droplet_frontal_area = M_PI * (diameter / 2.) * (diameter / 2.);

        // Drag coefficient
        const float drag_coefficient = (reynolds <= 1000.) ? 24 * (1. + 0.15 * pow(reynolds, 0.687)) / reynolds : 0.424;

        const vec3f drag_force = (drag_coefficient * reynolds * 0.5f * gas_density * relative_drop_vel_mag * droplet_frontal_area) * relative_drop_vel;
        const vec3f a1 = ((drag_force) / mass) * dt;
        if (a1 != a1)
        {
            // printf("NAN %d %f %f %f %f %f %f\n", index, drag_force, drag_coefficient, reynolds, (24 * (1. + 0.15 * pow(reynolds, 0.687)) / reynolds), (1. + 0.15 * pow(reynolds, 0.687)) / reynolds,powf(reynolds, 0.687f));
            // printf("NAN %f %f %f\n", reynolds, relative_drop_vel_mag, kinematic_viscosity); //powf(reynolds, 0.687f),powf(0.424000f, 0.687f));
            printf("NAN %f %f %f\n", relative_drop_vel.x, relative_drop_vel.y, relative_drop_vel.z); // powf(reynolds, 0.687f),powf(0.424000f, 0.687f));
            printf("NAN %f %f %f\n", ray.direction.x, ray.direction.y, ray.direction.z);
            printf("NAN %f %f %f %d\n", gas_vel.x, gas_vel.y, gas_vel.z, particleData.sectionID);

            asm("trap;");
        }
        ray.direction = ray.direction + a1 * dt;
        particle_dir = ray.direction;
        particle_vel = length(ray.direction);
        if (index == 0)
        {

            printf("accel %f %f %f \n", a1.x, a1.y, a1.z);
            printf("vel %f %f %f %f\n", ray.direction.x, ray.direction.y, ray.direction.z, length(ray.direction));
        }
        newParticle = ((threadid & 31) == 0);
    }
    // reduction of new particles
    __shared__ int write_base;
    int write_index = prefixsum(threadid, newParticle);

    if (threadid == BLOCK_THREADS - 1)
    {
        int total = write_index + newParticle;
        write_base = atomicAdd(particles.num_simulated, total); // get a location to write them out
    }

    __syncthreads(); // ensure write_base is valid for all threads

    if (newParticle)
    {
        if (write_base + write_index < max_particles)
        {
            vec3f dir = particle_vel * normalize(cross(particle_dir, vec3f(1, 0, 0)));

            // printf("writing %d \n", write_base + write_index);
            particles.particleData[write_base + write_index] = particles.particleData[index];
            particles.rays[write_base + write_index] = particles.rays[index];
            particles.rays[write_base + write_index].direction = dir;
        }
        else
        {
            // printf("we full in thgis house %d %d \n", write_base , write_index);
        }
    }

    // printf("%d %f %f %f\n", index, ray.direction.x,ray.direction.y,ray.direction.z);
}
