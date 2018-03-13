#include "post_process_bem_stokes.h"
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
  DoFTools::map_dofs_to_support_points<2, my_dim> (StaticMappingQ1<2,my_dim>::mapping,
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
  std::vector<Point<my_dim> > support_points(pp_bem.map_dh.n_dofs());
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
  double tol=5e-1;
  unsigned int max_degree = 1;
  const unsigned int dim =3;

  std::cout<<"Test that the BEM is able to recover the Fundamental Solution outside"<<std::endl;
  // ParsedFunction<3,3> exact_solution_eig("Exact solution position",
  //         "x / (x*x + y*y + z*z)^0.5 ; y / (x*x + y*y + z*z)^0.5 ; z / (x*x + y*y + z*z)^0.5");


  for (unsigned int degree=1; degree<=max_degree; degree++)
    {
      std::cout<< "Testing for degree = "<<degree<<std::endl;
      PostProcessBEMStokes<dim> post_process(MPI_COMM_WORLD);

      // post_process.compute_processor_properties();
      // PETScWrappers::SparseMatrix             V;
      // std::vector<PETScWrappers::MPI::Vector> eigenfunctions;
      // std::vector<Vector<double> > M_mult_eigenfunctions;
      // std::vector<double>                     eigenvalues;
      // PETScWrappers::MPI::Vector normal_vector_difference;

      // post_process.read_parameters(SOURCE_DIR "/parameters_test_3d_out.prm");
      ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_box.prm","foo.prm");
      // post_process.extra_debug_info = true;
      post_process.convert_bool_parameters();

      // We retrieve the two Finite Element Systems
      std::string fe_name_stokes = "FESystem<2,3>[FE_DGQ<2,3>(0)^3]";
      std::string fe_name_map = "FESystem<2,3>[FE_Q<2,3>(1)^3]";
      post_process.fe_stokes = SP(FETools::get_fe_by_name<2,3> (fe_name_stokes));
      post_process.fe_map = SP(FETools::get_fe_by_name<2,3> (fe_name_map));
      post_process.grid_fe = SP(post_process.parsed_grid_fe());
      post_process.box_fe_scalar = SP(post_process.parsed_box_fe_scalar());
      post_process.box_fe_vector = SP(post_process.parsed_box_fe_vector());

      // std::cout<<post_process.fe_stokes->get_name()<<std::endl;
      // std::cout<<post_process.fe_map->get_name()<<std::endl;
      post_process.extra_debug_info = false;
      post_process.create_grid_in_deal = false;
      post_process.create_ext_box_bool = true;
      post_process.n_rep_ext_box_ref = 4;
      post_process.point_box_1 = Point<dim> (-8.,-8.,-8.);
      post_process.point_box_2 = Point<dim> ( 8., 8., 8.);
      post_process.reflect_kernel=false;
      post_process.no_slip_kernel=false;
      post_process.velocity_kind="Total";

      // Vector<double> G_velocities(post_process.dh_stokes.n_dofs()), G_trace_1(post_process.dh_stokes.n_dofs()),
      //                G_trace_1_ex(post_process.dh_stokes.n_dofs()), trace_1_vector_difference(post_process.dh_stokes.n_dofs());



      // std::cout<<"baba"<<std::endl;
      // std::cout<<post_process.real_stokes_forces.linfty_norm()<<std::endl;
      // post_process.post_process_wall_bool[0] = true;
      // post_process.wall_positions[0] = Point<dim> (10.,0.,0.);
      // post_process.wall_spans[0][0] = 5.25;
      // post_process.wall_spans[0][1] = 20.25;
      // post_process.wall_spans[0][2] = 0.;
      // post_process.post_process_wall_bool[1] = false;
      // post_process.post_process_wall_bool[2] = false;
      // post_process.post_process_wall_bool[3] = false;


      std::cout<<"read external grid"<<std::endl;
      post_process.read_external_grid(post_process.external_grid_filename, post_process.external_grid);

      post_process.read_input_triangulation("../../../tests/box/reference_tria","bin",post_process.tria);
      // post_process.read_input_triangulation("../../../../BEMStokes/build_sphere/reference_tria","bin",post_process.tria);

      post_process.reinit();
      post_process.euler_vec.reinit(post_process.map_dh.n_dofs());
      VectorTools::get_position_vector(post_process.map_dh,post_process.euler_vec);
      post_process.mappingeul = SP(new MappingFEField<dim-1, dim>(post_process.map_dh,post_process.euler_vec));

      post_process.real_stokes_forces.reinit(post_process.dh_stokes.n_dofs());
      post_process.real_velocities.reinit(post_process.dh_stokes.n_dofs());
      post_process.original_normal_vector.reinit(post_process.dh_stokes.n_dofs());
      // std::cout<<"baba"<<std::endl;
      // std::cout<<"baba"<<std::endl;
      std::string filename_forces("../../../tests/results/stokes_forces_translated_sphere.bin");
      std::string filename_vel("../../../tests/results/total_velocities_translated_sphere.bin");
      std::string filename_normal("../../../tests/results/normal_vector_sphere.bin");


      std::ifstream forces(filename_forces.c_str());
      post_process.stokes_forces.block_read(forces);

      std::ifstream vel(filename_vel.c_str());
      post_process.total_velocities.block_read(vel);

      std::ifstream norm(filename_normal.c_str());
      post_process.original_normal_vector.block_read(norm);
      // post_process.total_velocities*=-1;

      // post_process.real_stokes_forces.print(std::cout);
      // post_process.real_velocities.print(std::cout);
      Vector<double> normals(post_process.map_dh.n_dofs());
      compute_normal_vector(post_process, normals);
      SphericalManifold<dim-1,dim> manifold;
      post_process.tria.set_all_manifold_ids(0);
      post_process.tria.set_manifold(0, manifold);
      std::cout<<post_process.external_grid.size()<<" "<<post_process.external_velocities.size()<<std::endl;
      post_process.compute_processor_properties();
      std::cout<< "Computing the exterior solution on the grid "  << std::endl;
      std::cout<< "There are " << post_process.external_grid.size() << " points, and "<<post_process.external_velocities.size()<<" velocity unknowns"<<std::endl;
      post_process.compute_real_forces_and_velocities();
      std::cout<<post_process.external_velocities.linfty_norm()<<" "<<post_process.external_velocities.l2_norm()<<std::endl;
      post_process.compute_exterior_stokes_solution_on_grid();
      std::cout<<post_process.external_velocities.linfty_norm()<<" "<<post_process.external_velocities.l2_norm()<<std::endl;
      std::cout<<"reduce and output"<<std::endl;
      post_process.compute_dissipation_energy();

      // for(auto i : post_process.dissipation_energy.locally_owned_elements())
      // {
      //   // double vel_norm = 0.;
      //   // for(unsigned int idim=0; idim<dim;++idim)
      //   //   vel_norm += external_velocities[i+idim*dissipation_energy.size()]*external_velocities[i+idim*dissipation_energy.size()];
      //   if(post_process.external_grid[i].square()<1.)
      //     post_process.dissipation_energy[i]=0.;
      // }

      FEValues<dim> fe_values_box_scalar (*post_process.box_fe_scalar, post_process.quadrature_box,
                                          update_values |
                                          update_quadrature_points | update_JxW_values);

      double dissipated_energy = 0.;
      auto n_q_points = post_process.quadrature_box.size();
      std::vector<double> local_energy(n_q_points);

      for (auto cell_scalar = post_process.box_dh_scalar.begin_active(); cell_scalar != post_process.box_dh_scalar.end(); ++cell_scalar)
        {
          fe_values_box_scalar.reinit (cell_scalar);
          fe_values_box_scalar.get_function_values(post_process.dissipation_energy, local_energy);
          for (unsigned int q=0; q<n_q_points; ++q)
            dissipated_energy += local_energy[q] * fe_values_box_scalar.JxW(q);


        }

      dissipated_energy *= 1.;
      double input_energy = 6.*numbers::PI;
      if (std::fabs(input_energy-dissipated_energy)/input_energy < tol)
        std::cout<<"OK DISSIPATED ENERGY"<<std::endl;
      else
        std::cout<<dissipated_energy<<" "<<input_energy<<std::endl;
      // Vector<double> G_ext(post_process.grid_dh.n_dofs()), G_vector_difference(post_process.grid_dh.n_dofs());

      // impose_G_as_ext_velocity(post_process, source_point, G_ext);

      // for (unsigned int i=0 ; i<G_ext.size(); ++i)
      //   {
      //     G_vector_difference[i] = std::fabs(post_process.external_velocities[i] - G_ext[i]);
      //     if (G_vector_difference[i]/std::fabs(G_ext[i]) > tol)
      //       std::cout<<"relative error : "<<G_vector_difference[i]/std::fabs(G_ext[i])<<" mine : "<<post_process.external_velocities[i]<<" true : "<< G_ext[i]<<std::endl;
      //     else
      //       std::cout<<"OK"<<std::endl;
      //   }




      post_process.reduce_output_grid_result(0);


      post_process.tria.set_manifold(0);

    }

  return 0;
}
