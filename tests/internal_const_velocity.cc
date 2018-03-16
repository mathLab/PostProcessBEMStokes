#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <iostream>
#include <fstream>
#include <deal2lkit/error_handler.h>
#include <deal2lkit/parsed_function.h>
#include <deal2lkit/parameter_acceptor.h>
#include <deal2lkit/utilities.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <mpi.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/slepc_solver.h>
#include "post_process_bem_stokes.h"

using namespace deal2lkit;
template<int my_dim>
void impose_G_as_velocity(const PostProcess::PostProcessBEMStokes<my_dim> &pp_bem, const Point<my_dim> &source, Vector<double> &G_velocities)
{
  std::vector<Point<my_dim> > support_points(pp_bem.dh_stokes.n_dofs());
  DoFTools::map_dofs_to_support_points<my_dim-1, my_dim> (*pp_bem.mappingeul,
                                                          pp_bem.map_dh, support_points);

  for (unsigned int i=0; i<pp_bem.dh_stokes.n_dofs()/my_dim; ++i)
    {
      const Tensor<1,my_dim> R = support_points[i] - source;
      Tensor<2,my_dim> G = pp_bem.stokes_kernel.value_tens(R) ;
      for (unsigned int jdim=0; jdim<my_dim; ++jdim)
        {
          G_velocities[i+pp_bem.dh_stokes.n_dofs()/my_dim*jdim] = G[jdim][0];
        }
    }

}

template<int my_dim>
void impose_G_as_ext_velocity(const PostProcess::PostProcessBEMStokes<my_dim> &pp_bem, const Point<my_dim> &source, Vector<double> &G_velocities)
{
  std::vector<Point<my_dim> > support_points(pp_bem.grid_dh.n_dofs());
  DoFTools::map_dofs_to_support_points<my_dim-1, my_dim> (StaticMappingQ1<2,my_dim>::mapping,
                                                          pp_bem.grid_dh, support_points);

  for (unsigned int i=0; i<pp_bem.grid_dh.n_dofs()/my_dim; ++i)
    {
      const Tensor<1,my_dim> R = support_points[i] - source;
      Tensor<2,my_dim> G = pp_bem.stokes_kernel.value_tens(R) ;
      for (unsigned int jdim=0; jdim<my_dim; ++jdim)
        {
          G_velocities[i+pp_bem.grid_dh.n_dofs()/my_dim*jdim] = G[jdim][0];
        }
    }

}
template<int my_dim>
void compute_normal_vector(const PostProcess::PostProcessBEMStokes<my_dim> &pp_bem, Vector<double> &normals)
{
  std::vector<Point<my_dim> > support_points(pp_bem.dh_stokes.n_dofs());
  DoFTools::map_dofs_to_support_points<my_dim-1, my_dim> (*pp_bem.mappingeul,
                                                          pp_bem.map_dh, support_points);

  for (unsigned int i = 0; i<support_points.size()/my_dim; ++i)
    {
      for (unsigned int d = 0; d<my_dim; ++d)
        normals[i+d*(support_points.size()/my_dim)] = support_points[i][d];
    }
}

template<int my_dim>
void impose_G_as_trace_1( PostProcess::PostProcessBEMStokes<my_dim> &pp_bem,  const Point<my_dim> &source, Vector<double> &normals, Vector<double> &G_trace_1)
{
  std::vector<Point<my_dim> > support_points(pp_bem.dh_stokes.n_dofs());
  DoFTools::map_dofs_to_support_points<my_dim-1, my_dim>(*pp_bem.mappingeul,
                                                         pp_bem.map_dh, support_points);

  for (unsigned int i=0; i<pp_bem.dh_stokes.n_dofs()/my_dim; ++i)
    {
      const Tensor<1,my_dim> R = support_points[i] - source;
      Point<my_dim> normal;
      for (unsigned int jdim=0; jdim<my_dim; ++jdim)
        normal[jdim] = - normals[i+pp_bem.dh_stokes.n_dofs()/my_dim*jdim];
      Tensor<3,my_dim> W = pp_bem.stokes_kernel.value_tens2(R) ;
      Tensor<2,my_dim> singular_ker = pp_bem.compute_singular_kernel(normal, W) ;
      for (unsigned int jdim=0; jdim<my_dim; ++jdim)

        G_trace_1[i+pp_bem.dh_stokes.n_dofs()/my_dim*jdim] = 1 * singular_ker[jdim][0];
    }

}


int main (int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  using namespace dealii;
  using namespace PostProcess;

  // const unsigned int degree = 1;
  // const unsigned int mapping_degree = 1;
  double tol=1e-6;
  unsigned int ncycles = 4;
  unsigned int degree = 1;
  const unsigned int dim =3;

  std::cout<<"Test that the BEM is able to recover the Fundamental Solution outside"<<std::endl;
  // ParsedFunction<3,3> exact_solution_eig("Exact solution position",
  //         "x / (x*x + y*y + z*z)^0.5 ; y / (x*x + y*y + z*z)^0.5 ; z / (x*x + y*y + z*z)^0.5");


  Point<dim> source_point(0.3, 0.3, 0.3);
  std::cout<< "Testing for degree = "<<degree<<std::endl;
  PostProcessBEMStokes<dim> post_process(MPI_COMM_WORLD);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_box.prm","foo.prm");
  post_process.convert_bool_parameters();
  post_process.post_process_wall_bool[0] = true;
  post_process.wall_positions[0][0] = 0.;
  post_process.wall_positions[0][0] = 0.;
  post_process.wall_positions[0][0] = 0.;
  post_process.wall_spans[0][0] = .25;
  post_process.wall_spans[0][1] = .25;
  post_process.wall_spans[0][2] = 0.;
  post_process.post_process_wall_bool[1] = false;
  post_process.post_process_wall_bool[2] = false;
  post_process.post_process_wall_bool[3] = false;

  post_process.read_input_triangulation(post_process.stored_results_path+"reference_tria","bin",post_process.tria);

  post_process.fe_stokes = post_process.parsed_fe_stokes();
  post_process.fe_map = post_process.parsed_fe_mapping();
  post_process.grid_fe = post_process.parsed_grid_fe();

  post_process.create_grid_in_deal = true;
  post_process.n_rep_ext_wall_ref = 3;
  std::cout<<"read external grid"<<std::endl;
  post_process.read_external_grid(post_process.external_grid_filename, post_process.external_grid);
  std::cout<<"resize"<<std::endl;

  post_process.reinit();
  // post_process.tria.refine_global();
  post_process.create_body_index_set();
  std::cout<<"compute proc props"<<std::endl;
  post_process.compute_processor_properties();
  post_process.mappingeul = SP(new MappingQ<dim-1, dim>(degree));


  post_process.real_stokes_forces.reinit(post_process.dh_stokes.n_dofs());
  post_process.real_velocities.reinit(post_process.dh_stokes.n_dofs());
  for (unsigned int i=0; i<post_process.dh_stokes.n_dofs(); ++i)
    // WE NEED THE MINUS ONE BECAUSE THE NORMALS ARE INVERTED FOR AN INTERNAL EVALUATION
    post_process.real_velocities[i]=-1.;
  std::cout<<"computing"<<std::endl;

  post_process.compute_exterior_stokes_solution_on_grid();
  std::cout<<"reduce and output"<<std::endl;
  post_process.reduce_output_grid_result(0);
  // post_process.external_velocities.print(std::cout);
  // if(post_process.external_velocities.linfty_norm())
  // post_process.external_velocities.print(std::cout);

  for (unsigned int i=0; i<post_process.external_velocities.size()/dim; ++i)
    if (std::abs(post_process.external_velocities[i]-1.)<tol)
      std::cout<<"OK"<<std::endl;
    else
      std::cout<<post_process.external_velocities[i]<<" expected "<<1. <<std::endl;
  std::cout<<std::endl;
  for (unsigned int i=post_process.external_velocities.size()/dim; i<post_process.external_velocities.size()*2/dim; ++i)
    if (std::abs(post_process.external_velocities[i]-1.)<tol)
      std::cout<<"OK"<<std::endl;
    else
      std::cout<<post_process.external_velocities[i]<<" expected "<<1. <<std::endl;
  std::cout<<std::endl;
  for (unsigned int i=2*post_process.external_velocities.size()/dim; i<post_process.external_velocities.size(); ++i)
    if (std::abs(post_process.external_velocities[i]-1.)<tol)
      std::cout<<"OK"<<std::endl;
    else
      std::cout<<post_process.external_velocities[i]<<" expected "<<1. <<std::endl;
  // std::cout<<"norm of ext vel : "<<post_process.external_velocities.linfty_norm()<<" : "<<post_process.external_velocities.l2_norm()<<std::endl;







  return 0;
}
