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
void impose_G_as_ext_velocity(const PostProcess::PostProcessBEMStokes<my_dim> &bem, const Point<my_dim> &source, Vector<double> &G_velocities)
{

  for (unsigned int i=0; i<bem.external_grid.size(); ++i)
    {
      const Tensor<1,my_dim> R = bem.external_grid[i] - source;
      Tensor<2,my_dim> G = bem.LH_stokes_kernel.value(R) ;
      for (unsigned int jdim=0; jdim<my_dim; ++jdim)
        {
          // std::cout<<i*my_dim+jdim<<" "<<G_velocities.size()<<" "<<my_dim<<" "<<bem.external_grid.size()<<std::endl;
          G_velocities[i*my_dim+jdim] = G[jdim][0];
        }
    }

}

template<int my_dim>
void create_ext_grid_for_test(PostProcess::PostProcessBEMStokes<my_dim> &pp)
{

  Point<my_dim> body_center, P1, P2;
  if (my_dim == 2)
    {
      body_center[0]=5.;
      body_center[1]=5.;
      P1[0]=body_center[0]+2;
      P1[1]=body_center[1]+2;
      P2[0]=body_center[0]-2;
      P2[1]=body_center[1]-2;


    }
  if (my_dim == 3)
    {
      body_center[0]=5.;
      body_center[1]=5.;
      body_center[2]=5.;

      P1[0]=body_center[0]+2;
      P1[1]=body_center[1]+2;
      P2[0]=body_center[0]-2;
      P2[1]=body_center[1]-2;

    }
  std::vector<unsigned int> repetition(2);
  repetition[0]=4;
  repetition[1]=4;

  // Triangulation<2,dim> ext_tria;
  GridGenerator::subdivided_hyper_rectangle<2,my_dim> (pp.ext_tria, repetition, P1, P2);
  // if(dim == 3)
  //   GridTools::transform (&blender_rotation<dim>, ext_tria);
  // FESystem<2,dim> gridfe(FE_Q<2,dim> (1),dim);
  // DoFHandler<2,dim> grid_dh(ext_tria);
  pp.grid_dh.distribute_dofs(pp.grid_fe);
  std::vector<Point<my_dim> > grid_support_points(pp.grid_dh.n_dofs());
  pp.external_grid.resize(pp.grid_dh.n_dofs()/my_dim);
  DoFTools::map_dofs_to_support_points<2,my_dim>( StaticMappingQ1<2,my_dim>::mapping, pp.grid_dh, grid_support_points);

  for (unsigned int i=0; i<pp.external_grid.size(); ++i)
    pp.external_grid[i]=grid_support_points[i*my_dim];
  // std::cout<<"Mamma, butta la pasta!"<<ext_tria.n_active_cells()<<" "<<ext_grid.size()<<std::endl;


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

  std::cout<< "Testing for degree = "<<degree<<std::endl;
  PostProcessBEMStokes<dim> post_process(MPI_COMM_WORLD);

  ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_box.prm","foo.prm");
  post_process.read_input_triangulation("../../../tests/box/reference_tria","bin",post_process.tria);
  post_process.add_box_to_tria(post_process.tria);
  post_process.convert_bool_parameters();

  // We retrieve the two Finite Element Systems
  post_process.fe_stokes = SP(post_process.parsed_fe_stokes());
  post_process.fe_map = SP(post_process.parsed_fe_mapping());
  post_process.grid_fe = SP(post_process.parsed_grid_fe());

  SphericalManifold<dim-1,dim> manifold;
  post_process.tria.set_all_manifold_ids(0);
  post_process.tria.set_manifold(0, manifold);



  post_process.reinit();
  // Vector<double> G_velocities(post_process.dh_stokes.n_dofs()), G_trace_1(post_process.dh_stokes.n_dofs()),
  //                G_trace_1_ex(post_process.dh_stokes.n_dofs()), trace_1_vector_difference(post_process.dh_stokes.n_dofs());




  post_process.mappingeul = SP(new MappingQ<dim-1, dim>(degree));

  post_process.real_stokes_forces.reinit(post_process.dh_stokes.n_dofs());
  post_process.real_velocities.reinit(post_process.dh_stokes.n_dofs());
  for (unsigned int i=0; i<post_process.dh_stokes.n_dofs(); ++i)
    post_process.real_velocities[i]=1.;
  // std::cout<<"baba"<<std::endl;
  // impose_G_as_velocity(post_process, source_point, post_process.real_velocities);
  // std::cout<<"baba"<<std::endl;
  Vector<double> normals(post_process.map_dh.n_dofs());
  // compute_normal_vector(post_process, normals);
  // std::cout<<"baba"<<std::endl;
  // impose_G_as_trace_1(post_process, source_point, normals, post_process.real_stokes_forces);

  post_process.post_process_wall_bool[0] = true;
  post_process.wall_positions[0] = Point<dim> (0.,-15.,0.);
  post_process.wall_spans[0][0] = 10.;
  post_process.wall_spans[0][1] = 10.;
  post_process.wall_spans[0][2] = 0.;
  post_process.post_process_wall_bool[1] = false;
  post_process.post_process_wall_bool[2] = false;
  post_process.post_process_wall_bool[3] = false;


  post_process.create_grid_in_deal = true;
  post_process.n_rep_ext_wall_ref = 3;
  std::cout<<"read external grid"<<std::endl;
  post_process.read_external_grid(post_process.external_grid_filename, post_process.external_grid);
  post_process.compute_processor_properties();

  post_process.compute_exterior_stokes_solution_on_grid();
  std::cout<<"reduce and output"<<std::endl;
  post_process.reduce_output_grid_result(0);

  // Vector<double> G_ext(post_process.grid_dh.n_dofs()), G_vector_difference(post_process.grid_dh.n_dofs());
  //
  // impose_G_as_ext_velocity(post_process, source_point, G_ext);

  for (unsigned int i=0 ; i<post_process.external_velocities.size(); ++i)
    {
      if (std::fabs(post_process.external_velocities[i]-1.)>tol)
        std::cout<<"error : "<<post_process.external_velocities[i]<<"  "<< post_process.external_grid[i]<<std::endl;
      else
        std::cout<<"OK"<<std::endl;
    }







  post_process.tria.set_manifold(0);




  return 0;
}
