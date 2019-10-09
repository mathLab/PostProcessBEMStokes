// ---------------------------------------------------------------------
//
// Copyright (C) 2014 - 2019 by the BEMStokes authors.
//
// This file is part of the BEMStokes library.
//
// The BEMStokes is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License version 2.1 as published by the Free Software Foundation.
// The full text of the license can be found in the file LICENSE at
// the top level of the BEMStokes distribution.
//
// Authors: Nicola Giuliani, Luca Heltai, Antonio DeSimone
//
// ---------------------------------------------------------------------

#ifndef __post_process_bem_stokes_h
#define __post_process_bem_stokes_h
// @sect3{Include files}

// The program starts with including a bunch of include files that we will use
// in the various parts of the program. Most of them have been discussed in
// previous tutorials already:
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_selector.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
// #include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/numerics/data_out.h>
//#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
// And here are a few C++ standard header files that we will need:
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
// We now include the kernels we have previously implemented
#include <deal.II/base/tensor_function.h>
#include <Sacado.hpp>
#include <Sacado_cmath.hpp>
#include <mpi.h>

#include <kernel.h>
#include <repeated_kernel.h>
#include <free_surface_kernel.h>
#include <no_slip_wall_kernel.h>

#include <deal2lkit/parameter_acceptor.h>
#include <deal2lkit/parsed_grid_generator.h>
#include <deal2lkit/parsed_quadrature.h>
#include <deal2lkit/parsed_finite_element.h>
#include <deal2lkit/utilities.h>
// The last part of this preamble is to import everything in the dealii
// namespace into the one into which everything in this program will go:
namespace PostProcess
{
  using namespace dealii;
  using namespace deal2lkit;

  template <int dim>
  class PostProcessBEMStokes : public deal2lkit::ParameterAcceptor
  {
  public:
    /// The simple constructor following the rules of ParameterAcceptor from deal2lkit.
    PostProcessBEMStokes(const MPI_Comm mpi_commy);

    ~PostProcessBEMStokes();

    /// The architecture of ParameterAcceptor demands the overload of declare_parameters and parse_parameters.
    void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);


    /// This function reads the triangulation identified by filename grid_format. In the PostProcessBEMStokes framework it reads the complete reference triangulation of
    /// the BEMStokes run comprehensive of all the wall or boxes required. In this way we can greatly simplify the creation of the triangulation in this framework.
    void read_input_triangulation(std::string filename, std::string grid_format, Triangulation<dim-1,dim> &triangulation);

    /// This function adds a wall on which analyse the velocity field. It reads the input wall file and it adds it to the post_process triangulation.
    void add_post_process_wall_to_tria(Triangulation<2, dim> &triangulation, unsigned int wall_number);

    /// This function reads the external grid specified by the user and fills the std::vector<Point<dim> > on which the velocity will be evaluated.
    void read_external_grid(const std::string &grid_filename, std::vector<Point<dim> > &ext_grid);

    /// Reinitialisation function that sets up all the vectors and DoFHandlers needed by the correct evaluation of the computed solution on the post process grid.
    void reinit();

    /// Workbalance between different processors. Basically a split of the external grid between different processors. We automatically manage any processors subdivision.
    void compute_processor_properties();

    /// The loader function for the results computed in a different BEMStokes run. They are store in a user specified directory and the function only cares of their retrival.
    void read_computed_results(unsigned int frame);

    /// We use a MappingFEField for the mapping of the swimmer triangulation (which is loaded by the computed resutls directory). We need to compute the value of the nodal point to be
    /// passed to such mapping.
    void compute_euler_vector(Vector<double> &vector, unsigned int frame, bool consider_rotations = false);//, unsigned int mag_factor = 1);

    /// We compute the singular kernel combining the 3 indices tensor W and the normal so as to retireve a standard 2 indices tensor.
    Tensor<2, dim> compute_singular_kernel(const Tensor<1, dim> normal, const Tensor<3,dim> W);

    /// we set real_stokes_forces and real_velocities that will be used in the computation of compute_exterior_stokes_solution_on_grid to value the convolution integrals on the boundary.
    /// At the moment we simply set them to be equal to stokes_forces and total_velocities respectively. Both of them are loaded by the computed simulation directory.
    void compute_real_forces_and_velocities();

    /// At this point we have already loaded the external grid for the velocity field exploration. We have balanced the workload between all the processors. The function evaluates the
    /// Stokes BEM on the selected points for the given processor. It also computes the mean velocity if more frame are considered.
    void compute_exterior_stokes_solution_on_grid();

    /// This function loads a computed Rotaion matrix R from the BEMStokes result directory.
    void read_rotation_matrix(FullMatrix<double> &rotation, const unsigned int frame);

    /// This function update the rotation matrix R using its representation using quaternions. It uses a forward euler scheme to update the quaternion.
    void update_rotation_matrix(FullMatrix<double> &rotation, const Vector<double> omega, const double dt);

    /// This function computes the average of the explored velocity field over the considered stroke.
    void compute_average(const unsigned int start_frame=0, unsigned int final_frame=120);

    /// It prepares the class members for the run considering the new frame.
    void reinit_for_new_frame(const unsigned int frame);

    /// It creates the IndexSet representing the swimmer in the loaded triangulation.
    void create_body_index_set();

    /// Computes the dissipation energy on the full dim-dimensional box.
    void compute_dissipation_energy();


    /// Helper function to convert the bool parameters in  std::vector<bools>.
    void convert_bool_parameters();

    /// The driver function of PostProcessBEMStokes. It takes as input the start  and end frames. Then it loads the reference_tria, it loads all the geometrical and computational
    /// inputs from the storage dir. Then, it loads/creates the external grid and it values the velocity field on the points of this exterior grid.
    void run(unsigned int start_frame=0, unsigned int end_frame=120);

    /// The second driver function of PostProcessBEMStokes. It takes as input the start  and end frames. Then it loads the previously computed output results (made with run)
    /// and compose them to obtain a mean value.
    void compose(unsigned int start_frame=0, unsigned int end_frame=120);

    /// This function: 1) reduces the split exterior velocity result on processor 0 2) it forces processor 0 to output the reduced velocity results.
    void reduce_output_grid_result(const unsigned int frame);

    /// If the bool parameter create_grid_in_deal is set this function takes care of the creation of the std::vector<Point<dim> > representing the external grid. We create the
    /// post process walls considering the settings of post_process_wall_bool, wall_spans, wall_positions.
    void create_ext_grid(std::vector<Point<dim> > &ext_grid);

    /// If the bool parameter create_ext_box is set this function takes care of the creation of the std::vector<Point<dim> > representing the external box. We create the
    /// post process walls considering the settings of the two points: point_box_1 and point_box_2.
    void create_ext_box(std::vector<Point<dim> > &ext_grid);

    /// Given the settings of the current simulation we can compute the first Green kernel G. It only uses one of the possible kernels depending on the input bool parameters.
    Tensor<2,dim> compute_G_kernel(const Tensor<1, dim> &R, const Tensor<1, dim> &R_image,
                                   const StokesKernel<dim> &stokes_kernel,
                                   const FreeSurfaceStokesKernel<dim> &fs_stokes_kernel,
                                   const NoSlipWallStokesKernel<dim> &ns_stokes_kernel,
                                   const bool reflect=false, const bool no_slip=false) const;

    /// Given the settings of the current simulation we can compute the second Green kernel W. It only uses one of the possible kernels depending on the input bool parameters.
    Tensor<3,dim> compute_W_kernel(const Tensor<1, dim> &R, const Tensor<1, dim> &R_image,
                                   const StokesKernel<dim> &stokes_kernel,
                                   const FreeSurfaceStokesKernel<dim> &fs_stokes_kernel,
                                   const NoSlipWallStokesKernel<dim> &ns_stokes_kernel,
                                   const bool reflect=false, const bool no_slip=false) const;

    Triangulation<dim-1, dim>   tria;
    Triangulation<2,dim> ext_tria;
    Triangulation<dim, dim> box_tria;

    std::unique_ptr<FiniteElement<dim-1, dim> > fe_stokes;
    std::unique_ptr<FiniteElement<dim-1, dim> > fe_map;
    std::unique_ptr<FiniteElement<2, dim> > grid_fe;
    std::unique_ptr<FiniteElement<dim, dim> > box_fe_vector;
    std::unique_ptr<FiniteElement<dim, dim> > box_fe_scalar;

    ParsedFiniteElement<dim-1, dim> parsed_fe_stokes;
    ParsedFiniteElement<dim-1, dim> parsed_fe_mapping;
    ParsedFiniteElement<2, dim> parsed_grid_fe;
    ParsedFiniteElement<dim, dim> parsed_box_fe_vector;
    ParsedFiniteElement<dim, dim> parsed_box_fe_scalar;

    DoFHandler<dim-1, dim> map_dh;
    DoFHandler<dim-1, dim> dh_stokes;
    DoFHandler<2, dim> grid_dh;
    DoFHandler<dim, dim> box_dh_vector;
    DoFHandler<dim, dim> box_dh_scalar;


    AffineConstraints<double> cm_stokes;

    std::string input_grid_base_name;
    std::string input_grid_format;
    std::string external_grid_filename;
    std::string stored_results_path;

    std::string velocity_kind;

    // std_cxx11::shared_ptr<Quadrature<dim-1> > quadrature;
    ParsedQuadrature<dim-1> quadrature;
    ParsedQuadrature<dim> quadrature_box;
    bool extra_debug_info;

    bool run_2d, run_3d;
    bool run_in_this_dimension;
    bool create_grid_in_deal;
    bool create_ext_box_bool;

    unsigned int external_grid_dimension;
    unsigned int n_frames;


    StokesKernel<dim> stokes_kernel;
    FreeSurfaceStokesKernel<dim> fs_stokes_kernel;
    NoSlipWallStokesKernel<dim> ns_stokes_kernel;

    // MappingQ<dim-1, dim>      mapping;
    Vector<double> euler_vec;
    Vector<double> next_euler_vec;
    shared_ptr<Mapping<dim-1, dim> > mappingeul;
    std::vector<Point<dim> > reference_support_points;

    std::vector<Point<dim> >  external_grid;

    Vector<double> stokes_forces;
    Vector<double> shape_velocities;
    Vector<double> total_velocities;
    Vector<double> rigid_puntual_velocities;
    Vector<double> external_velocities;
    Vector<double> mean_external_velocities;
    Vector<double> max_external_velocities;
    Vector<double> original_normal_vector;

    Vector<double> dissipation_energy;
    Vector<double> mean_dissipation_energy;

    std::vector<Vector<double> >  DN_N_rigid;
    Vector<double> rigid_velocities;
    Vector<double> real_stokes_forces;
    Vector<double> real_velocities;

    int rank;
    int size;

    unsigned int proc_start;
    unsigned int proc_end;
    unsigned int proc_external_dimension;
    unsigned int num_rigid;

    unsigned int n_rep_ext_wall_ref;
    unsigned int n_rep_ext_box_ref;


    FullMatrix<double> rotation_matrix;

    bool bool_rot;

    bool wall_bool_0;
    bool wall_bool_1;
    bool wall_bool_2;
    bool wall_bool_3;
    bool wall_bool_4;
    bool wall_bool_5;
    bool wall_bool_6;
    bool wall_bool_7;

    double refine_distance_from_center, wall_threshold;

    bool compute_dissipation_energy_bool;
    Point<dim> refinement_center;

    bool read_box_bool, read_cyl_bool, cylinder_manifold_bool;
    unsigned int first_index_box;
    Point<dim> cylinder_direction, cylinder_point_on_axis;
    Point<dim> point_box_1, point_box_2;

    std::shared_ptr<Manifold<dim-1, dim> > cylinder_manifold;

    std::vector<bool> wall_bool;

    bool post_process_wall_bool_0;
    bool post_process_wall_bool_1;
    bool post_process_wall_bool_2;
    bool post_process_wall_bool_3;
    bool post_process_wall_bool_4;

    std::vector<bool> post_process_wall_bool;

    bool reflect_kernel;
    bool no_slip_kernel;

    std::vector<Point<dim> > wall_positions;

    unsigned int delta_frame;

    unsigned int kernel_wall_orientation;

    bool compute_force_dipole_matrices;
    unsigned int max_momentum_order;
    std::vector<FullMatrix<double> > force_dipole_matrices;

    Point<dim> kernel_wall_position;

    std::vector<std::vector<double> > wall_spans;

    double time_step;

    const MPI_Comm mpi_communicator;

    unsigned int this_mpi_process;

    unsigned int n_mpi_processes;

    ConditionalOStream     pcout;
    ConditionalOStream dpcout;

    IndexSet body_cpu_set;

    // void reduce_exterior_results(const unsigned int frame);
    //
    // void remove_hanging_nodes_between_different_material_id(Triangulation<dim-1,dim> &tria_in,
    //                                                         const bool isotropic=false,
    //                                                         const unsigned int max_iterations=100);
    //
    // void evaluate_stokes_bie(
    //   const std::vector<Point<dim> > &val_points,
    //   const Vector<double> &vel,
    //   const Vector<double> &forces,
    //   Vector<double> &val_velocities);
    //
    // void refine_walls(Triangulation<dim-1, dim> &triangulation, const double max_distance, const double threshold, const Point<dim> &center);
    //
    // void read_parameters (const std::string &filename);
    //
    // void read_input_mesh_file(unsigned int frame, Triangulation<dim-1,dim> &triangulation);
    //
    // void read_domain();
    //
    // void add_wall_to_tria(Triangulation<dim-1, dim> &triangulation, unsigned int wall_number);
    //
    void add_box_to_tria(Triangulation<dim-1, dim> &triangulation);
    //
    // void add_cylinder_to_tria(Triangulation<dim-1, dim> &triangulation, bool apply_manifold, std::string filename="cylinder.inp");

  };

}

#endif
