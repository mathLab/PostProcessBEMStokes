
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
#include <deal.II/base/function_lib.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/numerics/data_out.h>
//#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
// And here are a few C++ standard header files that we will need:
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <mpi.h>
// We now include the kernels we have previously implemented
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <post_process_bem_stokes.h>

#include <boost/archive/binary_iarchive.hpp>
// The last part of this preamble is to import everything in the dealii
// namespace into the one into which everything in this program will go:
namespace PostProcess
{
  using namespace dealii;


  // @sect3{Single and double layer operator kernels}

  // First, let us define a bit of the boundary integral equation machinery.

  // @sect4{BEMProblem::BEMProblem and BEMProblem::read_parameters}

  // The constructor initializes the various object in much the same way as
  // done in the finite element programs such as step-4 or step-6. The only
  // new ingredient here is the ParsedFunction object, which needs, at
  // construction time, the specification of the number of components.
  //
  // For the exact solution the number of vector components is one, and no
  // action is required since one is the default value for a ParsedFunction
  // object. The wind, however, requires dim components to be
  // specified. Notice that when declaring entries in a parameter file for the
  // expression of the Functions::ParsedFunction, we need to specify the
  // number of components explicitly, since the function
  // Functions::ParsedFunction::declare_parameters is static, and has no
  // knowledge of the number of components.
  template <>
  PostProcessBEMStokes<3>::PostProcessBEMStokes(const MPI_Comm mpi_commy)
    :
    // fe_map(FE_Q<dim-1,dim>(fe_degree),dim),
    // fe_stokes(FE_Q<dim-1, dim> (fe_degree), dim), // my_stokes_kernel.n_components si incazza ???
    // mapping(fe_degree, true),
    // grid_fe(FE_Q<2, dim> (1), dim),
    parsed_fe_stokes("Finite Element Stokes","FESystem<2,3>[FE_Q<2,3>(1)^3]","u,u,u",3),//,"FESystem<2,3>[FE_Q<2,3>(1)^3]","u,u,u",dim),
    parsed_fe_mapping("Finite Element Mapping","FESystem<2,3>[FE_Q<2,3>(1)^3]","u,u,u",3),//,"FESystem<2,3>[FE_Q<2,3>(1)^3]","u,u,u",dim),
    parsed_grid_fe("Finite Element External Grid","FESystem<2,3>[FE_Q<2,3>(1)^3]","u,u,u",3),
    map_dh(tria),
    dh_stokes(tria),
    grid_dh(ext_tria),
    mappingeul(NULL),
    rotation_matrix(3,3),
    wall_bool(8, false),
    post_process_wall_bool(4, false),
    wall_positions(4,Point<3>()),
    wall_spans(4,std::vector<double>(3)),
    time_step(0.1),
    mpi_communicator(mpi_commy),
    this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
    n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
    pcout (std::cout,
           (this_mpi_process
            == 0)),
    dpcout(std::cout)

  {}

  template <>
  PostProcessBEMStokes<2>::PostProcessBEMStokes(const MPI_Comm mpi_commy)
    :
    // fe_map(FE_Q<dim-1,dim>(fe_degree),dim),
    // fe_stokes(FE_Q<dim-1, dim> (fe_degree), dim), // my_stokes_kernel.n_components si incazza ???
    // mapping(fe_degree, true),
    // grid_fe(FE_Q<2, dim> (1), dim),
    parsed_fe_stokes("Finite Element Stokes","FESystem<1,2>[FE_Q<1,2>(1)^2]","u,u",2),//,"FESystem<2,3>[FE_Q<2,3>(1)^3]","u,u,u",dim),
    parsed_fe_mapping("Finite Element Mapping","FESystem<1,2>[FE_Q<1,2>(1)^2]","u,u",2),//,"FESystem<2,3>[FE_Q<2,3>(1)^3]","u,u,u",dim),
    parsed_grid_fe("Finite Element External Grid","FESystem<2,2>[FE_Q<2,2>(1)^2]","u,u",2),
    map_dh(tria),
    dh_stokes(tria),
    grid_dh(ext_tria),
    mappingeul(NULL),
    rotation_matrix(2,2),
    wall_bool(8, false),
    post_process_wall_bool(4, false),
    wall_positions(4,Point<2>()),
    wall_spans(4,std::vector<double>(2)),
    time_step(0.1),
    mpi_communicator(mpi_commy),
    this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
    n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
    pcout (std::cout,
           (this_mpi_process
            == 0)),
    dpcout(std::cout)

  {}

  template <int dim>
  PostProcessBEMStokes<dim>::~PostProcessBEMStokes()
  {}

  template <int dim>
  void PostProcessBEMStokes<dim>::declare_parameters (ParameterHandler &prm)
  {
    //deallog << std::endl << "Parsing parameter file " << filename << std::endl
    //        << "for a " << dim << " dimensional simulation. " << std::endl;

    // ParameterHandler prm;

    add_parameter(prm, &n_frames, "Total number of frames","120",Patterns::Integer());
    // prm.declare_entry("Total number of frames","120",Patterns::Integer());

    add_parameter(prm, &bool_rot, "Consider rigid rotations","true",Patterns::Bool());
    // prm.declare_entry("Consider rigid rotations","true",Patterns::Bool());
    add_parameter(prm, &run_2d, "Run 2d simulation", "true",
                  Patterns::Bool());
    add_parameter(prm, &run_3d, "Run 3d simulation", "true",
                  Patterns::Bool());

    // prm.declare_entry("Total number of frames","140",Patterns::Integer());
    //
    // prm.declare_entry("Run 2d simulation", "true",
    //                   Patterns::Bool());
    // prm.declare_entry("Run 3d simulation", "true",
    //                   Patterns::Bool());


    add_parameter(prm, &input_grid_base_name, "Input grid base name (for testing, unused in real PostProcessBEMStokes)", "../../BEMStokes/Guasto_input_grids_new_spline/single_mesh_3d_",
                  Patterns::Anything());
    add_parameter(prm, &input_grid_format, "Input grid format (for testing, unused in real PostProcessBEMStokes)", "msh",
                  Patterns::Anything());

    // prm.declare_entry("Input grid base name", "../input_grids/try_",
    //       Patterns::Anything());
    //
    // prm.declare_entry("Input grid format", "vtk",
    //       Patterns::Anything());


    add_parameter(prm, &external_grid_filename, "External grid file name", "../external_grids/cross.txt",
                  Patterns::Anything());
    add_parameter(prm, &stored_results_path, "Path to stored results", "../../BEMStokes/build/",
                  Patterns::Anything());


    add_parameter(prm, &velocity_kind, "Kind of velocity to be analysed", "Total",
                  Patterns::Selection("Total|BodyFrame"));

    add_parameter(prm, &external_grid_dimension, "External grid dimension","200",Patterns::Integer());


    add_parameter(prm, &create_grid_in_deal, "Create the grid inside the code","false", Patterns::Bool());

    add_parameter(prm, &extra_debug_info, "Print extra debug information", "false", Patterns::Bool());

    // prm.declare_entry("External grid file name", "../external_grid/cross.txt",
    //       Patterns::Anything());
    //
    // prm.declare_entry("Path to stored results","../../BEMStokes/build/",
    //       Patterns::Anything());

    // prm.declare_entry("External grid dimension", "200",
    //       Patterns::Integer());

    // add_parameter(prm,&singular_quadrature_order,"Singular quadrature order", "5", Patterns::Integer());

    // prm.enter_subsection("Quadrature rules");
    // {
    //   prm.declare_entry("Quadrature type", "gauss",
    //                     Patterns::Selection(QuadratureSelector<(dim-1)>::get_quadrature_names()));
    //   prm.declare_entry("Quadrature order", "4", Patterns::Integer());
    // }
    // prm.leave_subsection();

    if (dim == 3)
      {
        add_parameter(prm, &reflect_kernel,"Reflect the kernel","false",//,0,0,0",
                      Patterns::Bool());//Patterns::List(Patterns::Double(),4,4));

        add_parameter(prm, &no_slip_kernel,"Use no slip kernel","false",//,0,0,0",
                      Patterns::Bool());//Patterns::List(Patterns::Double(),4,4));

        add_parameter(prm, &wall_bool_0,"Wall 0 bool","false",
                      Patterns::Bool(),"Bool set to create wall 0.");
        add_parameter(prm, &wall_bool_1,"Wall 1 bool","false",
                      Patterns::Bool(),"Bool set to create wall 1.");
        add_parameter(prm, &wall_bool_2,"Wall 2 bool","false",
                      Patterns::Bool(),"Bool set to create wall 2.");
        add_parameter(prm, &wall_bool_3,"Wall 3 bool","false",
                      Patterns::Bool(),"Bool set to create wall 3.");
        add_parameter(prm, &wall_bool_4,"Wall 4 bool","false",
                      Patterns::Bool(),"Bool set to create wall 4.");
        add_parameter(prm, &wall_bool_5,"Wall 5 bool","false",
                      Patterns::Bool(),"Bool set to create wall 5.");
        add_parameter(prm, &wall_bool_6,"Wall 6 bool","false",
                      Patterns::Bool(),"Bool set to create wall 6.");
        add_parameter(prm, &wall_bool_7,"Wall 7 bool","false",
                      Patterns::Bool(),"Bool set to create wall 7.");

        add_parameter(prm, &read_box_bool,"Read box","false",
                      Patterns::Bool(),"Bool set to read a box.");

        add_parameter(prm, &read_cyl_bool,"Read cylinder","false",
                      Patterns::Bool(),"Bool set to read a cylinder as box.");
        add_parameter(prm, &cylinder_manifold_bool,"Cylinder Apply Manifold descriptor","true",
                      Patterns::Bool());
        add_parameter(prm, &cylinder_direction,"Cylinder axis orientation","0.,0.,1.",
                      Patterns::List(Patterns::Double(),dim,dim));

        add_parameter(prm, &cylinder_point_on_axis,"Cylinder point on axis","0.,0.,0.",
                      Patterns::List(Patterns::Double(),dim,dim));

        add_parameter(prm, &first_index_box, "First Index for Box","0",Patterns::Integer());
      }
    add_parameter(prm, &post_process_wall_bool_0,"Post Process Wall 0 bool","true",
                  Patterns::Bool(),"Bool set to create Post Process wall 0.");
    if (dim == 3)
      {
        add_parameter(prm, &post_process_wall_bool_1,"Post Process Wall 1 bool","false",
                      Patterns::Bool(),"Bool set to create Post Process wall 1.");

        add_parameter(prm, &post_process_wall_bool_2,"Post Process Wall 2 bool","false",
                      Patterns::Bool(),"Bool set to create Post Process wall 2.");

        add_parameter(prm, &post_process_wall_bool_3,"Post Process Wall 3 bool","false",
                      Patterns::Bool(),"Bool set to create Post Process wall 3.");

        add_parameter(prm, &(wall_spans[0]),"Wall 0 spans","10,0,10",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the spans of the wall 0. If -1 we intend infinite.");
        // std::cout<<foo[2]<<std::endl;
        add_parameter(prm, &(wall_spans[1]),"Wall 1 spans","10,0,10",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the spans of the wall 1. If -1 we intend infinite.");
        add_parameter(prm, &(wall_spans[2]),"Wall 2 spans","1,1,-1",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the spans of the wall 2. If -1 we intend infinite.");
        add_parameter(prm, &(wall_spans[3]),"Wall 3 spans","1,1,-1",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the spans of the wall 3. If -1 we intend infinite.");

        add_parameter(prm, &wall_positions[0],"Wall center position wall 0","0,5,0",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the positions of all the centers of the walls.");
        add_parameter(prm, &wall_positions[1],"Wall center position wall 1","0,-5,0",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the positions of all the centers of the walls.");
        add_parameter(prm, &wall_positions[2],"Wall center position wall 2","0,10,0",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the positions of all the centers of the walls.");
        add_parameter(prm, &wall_positions[3],"Wall center position wall 3","0,10,0",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the positions of all the centers of the walls.");

        add_parameter(prm, &kernel_wall_position,"Kernel Wall center position","0,5,0",
                      Patterns::List(Patterns::Double(),dim,dim));
      }
    else if (dim == 2)
      {
        add_parameter(prm, &(wall_spans[0]),"Wall 0 spans","10,10",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the spans of the wall 0.");
        add_parameter(prm, &wall_positions[0],"Wall center position wall 0","0,0",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the positions of all the centers of the walls.");

      }
    add_parameter(prm, &kernel_wall_orientation, "Kernel Wall orientation","1",Patterns::Integer());

    add_parameter(prm, &n_rep_ext_wall_ref, "Number of global_refinement for ext wall","20",Patterns::Integer());

    add_parameter(prm, &wall_threshold,"Wall Refinement Threshold","1.",
                  Patterns::Double(),"Minimum diameter you want for the wall at its center.");

    add_parameter(prm, &refine_distance_from_center,"Distance for wall refinement","2.",
                  Patterns::Double(),"Minimum distance from center you want to refine on the wall.");
    if (dim == 3)
      add_parameter(prm, &refinement_center,"Refinement Center For Wall Refinement","0.,0.,0.",
                    Patterns::List(Patterns::Double(),dim,dim));
    else
      add_parameter(prm, &refinement_center,"Refinement Center For Wall Refinement","0.,0.",
                    Patterns::List(Patterns::Double(),dim,dim));


  }

  template <int dim>
  void PostProcessBEMStokes<dim>::parse_parameters (ParameterHandler &prm)
  {
    ParameterAcceptor::parse_parameters(prm);
    dpcout.set_condition(extra_debug_info && this_mpi_process == 0);

    // After declaring all these parameters to the ParameterHandler object,
    // let's read an input file that will give the parameters their values. We
    // then proceed to extract these values from the ParameterHandler object:
    // prm.read_input(filename);

    // n_frames = prm.get_integer("Total number of frames");
    //
    // input_grid_base_name = prm.get("Input grid base name");
    //
    // input_grid_format = prm.get("Input grid format");
    //
    // external_grid_filename = prm.get("External grid file name");
    //
    // external_grid_dimension = prm.get_integer("External grid dimension");
    //
    // stored_results_path = prm.get("Path to stored results");
    //
    // prm.enter_subsection("Quadrature rules");
    // {
    //   quadrature =
    //     std_cxx11::shared_ptr<Quadrature<dim-1> >
    //     (new QuadratureSelector<dim-1> (prm.get("Quadrature type"),
    //                                     prm.get_integer("Quadrature order")));
    // }
    // prm.leave_subsection();
    //
    //
    //
    // run_in_this_dimension = prm.get_bool("Run " +
    //                                      Utilities::int_to_string(dim) +
    //                                      "d simulation");
  }

  // template <int dim>
  // void PostProcessBEMStokes<dim>::read_input_mesh_file(unsigned int frame, Triangulation<dim-1,dim> &triangulation)
  // {
  //   std::ifstream in;
  //   in.open (input_grid_base_name + Utilities::int_to_string(frame)+"." + input_grid_format);
  //   GridIn<dim-1, dim> gi;
  //   gi.attach_triangulation (triangulation);
  //   if (input_grid_format=="vtk")
  //     gi.read_vtk (in);
  //   else if (input_grid_format=="msh")
  //     gi.read_msh (in);
  //   else if (input_grid_format=="inp")
  //     gi.read_ucd (in);
  //   else
  //     Assert (false, ExcNotImplemented());
  //
  //   // for(unsigned int i=0; i<wall_bool.size(); ++i)
  //   //   if(wall_bool[i])
  //   //     add_wall_to_tria(tria, i);//wall_types[i], wall_positions[i], wall_spans[i], i, wall_threshold, refine_distance_from_center);
  //
  // }
  template <int dim>
  void PostProcessBEMStokes<dim>::read_input_triangulation(std::string filename, std::string grid_format, Triangulation<dim-1,dim> &triangulation)
  {
    std::ifstream in;
    in.open (filename+"." + grid_format);
    GridIn<dim-1, dim> gi;
    gi.attach_triangulation (triangulation);
    if (grid_format=="vtk")
      gi.read_vtk (in);
    else if (grid_format=="msh")
      gi.read_msh (in);
    else if (grid_format=="inp")
      gi.read_ucd (in);
    else if (grid_format=="bin")
      {
        pcout<<"reading "<<filename+"." + grid_format<<std::endl;

        boost::archive::binary_iarchive ia(in);
        triangulation.clear();
        triangulation.load(ia, 0);
        // pcout<<"pippo "<<filename+"." + grid_format<<std::endl;

      }
    else
      Assert (false, ExcNotImplemented());

    // for(unsigned int i=0; i<wall_bool.size(); ++i)
    //   if(wall_bool[i])
    //     add_wall_to_tria(tria, i);//wall_types[i], wall_positions[i], wall_spans[i], i, wall_threshold, refine_distance_from_center);

  }
  //
  // template<int dim>
  // void PostProcessBEMStokes<dim>::add_wall_to_tria(Triangulation<dim-1, dim> &triangulation, unsigned int i_wall)
  // {
  //   pcout<<"Adding wall "<<i_wall<<std::endl;
  //   Triangulation<dim-1, dim> foo_ext_tria;
  //   Triangulation<dim-1, dim> triangulation_old;
  //   triangulation_old.copy_triangulation(triangulation);
  //   GridIn<dim-1, dim> gi;
  //   gi.attach_triangulation (foo_ext_tria);
  //   std::ifstream in;
  //   std::string filename = stored_results_path+"wall_"+Utilities::int_to_string(i_wall)+".inp";
  //   in.open (filename);
  //   gi.read_ucd(in);
  //   GridGenerator::merge_triangulations(triangulation_old, foo_ext_tria, triangulation);
  //
  // }
  //
  // template<int dim>
  // void PostProcessBEMStokes<dim>::add_cylinder_to_tria(Triangulation<dim-1, dim> &triangulation, bool apply_manifold, std::string filename)
  // {
  //   Triangulation<dim-1, dim> triangulation_wall;
  //   Triangulation<dim-1, dim> triangulation_old;
  //   triangulation_old.copy_triangulation(triangulation);
  //   GridIn<dim-1, dim> gi;
  //   gi.attach_triangulation (triangulation_wall);
  //   std::ifstream in;
  //   std::string real_filename=stored_results_path+filename;
  //   in.open (real_filename);
  //   gi.read_ucd(in, true);
  //
  //   GridGenerator::merge_triangulations(triangulation_old, triangulation_wall, triangulation);
  //
  //   if (apply_manifold)
  //     {
  //       cylinder_manifold = std::shared_ptr<CylindricalManifold<dim-1, dim> > (new CylindricalManifold<dim-1, dim> (cylinder_direction, cylinder_point_on_axis));
  //       typename Triangulation<dim-1,dim>::active_cell_iterator
  //       cell = triangulation.begin_active(),
  //       endc = triangulation.end();
  //       for (cell=triangulation.begin_active(); cell != endc; ++cell)
  //         {
  //           if (cell->material_id()==2 || cell->material_id()==3)
  //             {
  //               cell->set_all_manifold_ids(99);
  //             }
  //         }
  //       if (cylinder_manifold)
  //         triangulation.set_manifold(99, *cylinder_manifold);
  //       else
  //         triangulation.set_manifold(99);
  //     }
  // }
  //
  template<int dim>
  void PostProcessBEMStokes<dim>::add_box_to_tria(Triangulation<dim-1, dim> &triangulation)
  {
    pcout<<"Adding box."<<std::endl;
    Triangulation<dim-1, dim> foo_ext_tria;
    Triangulation<dim-1, dim> triangulation_old;
    triangulation_old.copy_triangulation(triangulation);
    GridIn<dim-1, dim> gi;
    gi.attach_triangulation (foo_ext_tria);
    std::ifstream in;
    std::string filename = stored_results_path+"box.inp";
    pcout<<filename<<std::endl;
    in.open (filename);
    gi.read_ucd(in);
    pcout<<"read"<<std::endl;
    GridGenerator::merge_triangulations(triangulation_old, foo_ext_tria, triangulation);
    pcout<<"merged"<<std::endl;


  }

  template<int dim>
  void PostProcessBEMStokes<dim>::add_post_process_wall_to_tria(Triangulation<2, dim> &triangulation, unsigned int i_wall)
  {
    // const unsigned int dim = 3;
    pcout<<"Adding post process wall "<<i_wall<<std::endl;
    if (triangulation.n_active_cells()>0)
      {
        Triangulation<2, dim> foo_ext_tria;
        Triangulation<2, dim> triangulation_old;
        triangulation_old.copy_triangulation(triangulation);
        GridIn<2, dim> gi;
        gi.attach_triangulation (foo_ext_tria);
        std::ifstream in;
        std::string filename = "post_process_wall_"+Utilities::int_to_string(i_wall)+".inp";
        in.open (filename);
        gi.read_ucd(in);
        GridGenerator::merge_triangulations(triangulation_old, foo_ext_tria, triangulation);
      }
    else
      {
        Triangulation<2, dim> foo_ext_tria;
        GridIn<2, dim> gi;
        gi.attach_triangulation (foo_ext_tria);
        std::ifstream in;
        std::string filename = "post_process_wall_"+Utilities::int_to_string(i_wall)+".inp";
        in.open (filename);
        gi.read_ucd(in);
        triangulation.clear();
        triangulation.copy_triangulation(foo_ext_tria);
      }
  }

  // @sect4{BEMProblem::read_domain}

  // A boundary element method triangulation is basically the same as a
  // (dim-1) dimensional triangulation, with the difference that the vertices
  // belong to a (dim) dimensional space.
  //
  // Some of the mesh formats supported in deal.II use by default three
  // dimensional points to describe meshes. These are the formats which are
  // compatible with the boundary element method capabilities of deal.II. In
  // particular we can use either UCD or GMSH formats. In both cases, we have
  // to be particularly careful with the orientation of the mesh, because,
  // unlike in the standard finite element case, no reordering or
  // compatibility check is performed here.  All meshes are considered as
  // oriented, because they are embedded in a higher dimensional space. (See
  // the documentation of the GridIn and of the Triangulation for further
  // details on orientation of cells in a triangulation.) In our case, the
  // normals to the mesh are external to both the circle in 2d or the sphere
  // in 3d.
  //
  // The other detail that is required for appropriate refinement of the
  // boundary element mesh, is an accurate description of the manifold that
  // the mesh is approximating. We already saw this several times for the
  // boundary of standard finite element meshes (for example in step-5 and
  // step-6), and here the principle and usage is the same, except that the
  // HyperBallBoundary class takes an additional template parameter that
  // specifies the embedding space dimension. The function object still has to
  // be static to live at least as long as the triangulation object to which
  // it is attached.

  // template <int dim>
  // void PostProcessBEMStokes<dim>::read_domain()
  // {
  //   static const Point<dim> center = Point<dim>();
  //   static const HyperBallBoundary<dim-1, dim> boundary(center,1.);
  //
  //   switch (dim)
  //     {
  //     case 2:
  //     {
  //       // std::ifstream in;
  //       // in.open ("../coarse_square.inp");
  //       // GridIn<dim-1, dim> gi;
  //       // gi.attach_triangulation (tria);
  //       // gi.read_ucd (in);
  //       pcout << "We take " << input_grid_base_name + "0."+input_grid_format+", as reference grid" << std::endl;
  //       read_input_mesh_file(0, tria);
  //
  //       break;
  //     }
  //     case 3:
  //     {
  //       pcout << "We take " << input_grid_base_name + "0."+input_grid_format+", as reference grid" << std::endl;
  //       read_input_mesh_file(0, tria);
  //
  //       //in.open ("../coarse_sphere.inp");
  //       break;
  //     }
  //     default:
  //       Assert (false, ExcNotImplemented());
  //     }
  //
  //   pcout << "We have a tria of  " << tria.n_active_cells() << " active_cells" << std::endl;
  //   //GridGenerator::hyper_cube (tria);
  //
  //   //tria.set_boundary(1, boundary);
  // }

  template<int dim>
  void PostProcessBEMStokes<dim>::read_external_grid(const std::string &grid_filename, std::vector<Point<dim> > &ext_grid)
  {
    std::ifstream infile (grid_filename.c_str());
    // pcout<<grid_filename<<std::endl;
    std::string instring;
    unsigned int i = 0;
    // pcout<<"PIPPO!!!"<<external_grid_dimension<<std::endl;
    // depending on the user choice we can create internally the grid in deal.ii or read by file. We have left the choice with external_grid_dimension==1681 just for backward compatibility.
    if (create_grid_in_deal)
      {
        create_ext_grid(ext_grid);

      }
    else
      {
        //pcout<<"ahah";
        while (infile.good() && i<external_grid_dimension)
          {
            for (unsigned int j=0; j<2; ++j)
              {
                getline ( infile, instring, ' ');
                //pcout<<"pippo ";
                ext_grid[i](j) = std::stod(instring);
              }
            getline ( infile, instring, '\n');
            if (dim == 3)
              ext_grid[i](dim-1) = std::stod(instring);
            // pcout<<ext_grid[i]<<std::endl;
            i=i+1;
          }
      }
    // external_grid_dimension = i;
    // std::cout<<external_grid_dimension<<std::endl;

  }


  // @sect4{BEMProblem::reinit}
  // This function globally distributes degrees of freedom,
  // and resizes matrices and vectors.

  template <int dim>
  void PostProcessBEMStokes<dim>::reinit()
  {
    if (dim == 2)
      num_rigid = dim + 1;
    else if (dim == 3)
      num_rigid = dim + dim;

    dh_stokes.distribute_dofs(*fe_stokes);
    map_dh.distribute_dofs(*fe_map);
    DoFRenumbering::component_wise(dh_stokes);
    DoFRenumbering::component_wise(map_dh);

    pcout << "There are " << dh_stokes.n_dofs() << " degrees of freedom"<< std::endl;

    external_grid.resize(external_grid_dimension);


    const unsigned int n_dofs_stokes =  dh_stokes.n_dofs();


    rigid_puntual_velocities.reinit(n_dofs_stokes);

    stokes_forces.reinit(n_dofs_stokes);
    shape_velocities.reinit(n_dofs_stokes);
    total_velocities.reinit(n_dofs_stokes);

    DN_N_rigid.resize(num_rigid, Vector<double> (n_dofs_stokes));
    rigid_velocities.reinit(num_rigid);
    real_stokes_forces.reinit(n_dofs_stokes);
    real_velocities.reinit(n_dofs_stokes);

    euler_vec.reinit(map_dh.n_dofs());
    next_euler_vec.reinit(map_dh.n_dofs());


    external_velocities.reinit(dim*external_grid_dimension);
    mean_external_velocities.reinit(dim*external_grid_dimension);
    reference_support_points.resize(n_dofs_stokes);
    pcout<<n_dofs_stokes<<" "<<std::endl;
    DoFTools::map_dofs_to_support_points<dim-1, dim>( StaticMappingQ1<dim-1, dim>::mapping, dh_stokes, reference_support_points);

    cm_stokes.clear();
    DoFTools::make_hanging_node_constraints(dh_stokes,cm_stokes);

  }

  template<int dim>
  void PostProcessBEMStokes<dim>::compute_processor_properties()
  {
//    MPI_Comm_rank(comm, &rank);
//    MPI_Comm_size(comm, &size);
    rank = MPI::COMM_WORLD.Get_rank();
    size = MPI::COMM_WORLD.Get_size();
    int rest;
    rest = external_grid_dimension % size;
    proc_external_dimension = external_grid_dimension / size;
    if (rest != 0)
      {
        if (rank < rest)
          {
            proc_start = (proc_external_dimension + 1) * rank;
            proc_end = (proc_external_dimension + 1) * (rank + 1);
            proc_external_dimension = proc_external_dimension + 1;
          }
        else
          {
            proc_start = proc_external_dimension * rank + rest;
            proc_end = proc_external_dimension * (rank + 1) + rest;
          }
      }
    else
      {
        proc_start = proc_external_dimension * rank;
        proc_end = proc_external_dimension * (rank+1);
      }
  }

  template<>
  void PostProcessBEMStokes<2>::update_rotation_matrix(FullMatrix<double> &rotation, const Vector<double> omega, const double dt)
  {
    AssertThrow(false, ExcImpossibleInDim(2));

  }

  template<int dim>
  void PostProcessBEMStokes<dim>::update_rotation_matrix(FullMatrix<double> &rotation, const Vector<double> omega, const double dt)
  {
    pcout << "Updating the rotation matrix using quaternions" << std::endl;
    // Firstly we need to reconstruct the original quaternion given the rotation matrix
    // FullMatrix<double> dummy(dim,dim);
    // dummy[0][0]=1.;
    // dummy[2][1]=1.;
    // dummy[1][2]=-1.;
    // rotation = dummy;
    // rotation.print(std::cout);
    Vector<double> q(dim+1);
    double q_dummy;
    // IT SEEMS OK.
    q_dummy = 1.;
    for (int i = 0; i<dim; ++i)
      q_dummy += (rotation[i][i]);

    q[0] = std::pow(q_dummy, 0.5)/2;


    q[1] = 1 / q[0] * 0.25 * (rotation[2][1] - rotation[1][2]);
    q[2] = 1 / q[0] * 0.25 * (rotation[0][2] - rotation[2][0]);
    q[3] = 1 / q[0] * 0.25 * (rotation[1][0] - rotation[0][1]);

    double foo = std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    q/=std::sqrt(foo);
    Vector<double> qdot(dim+1), omega_plus(dim+1);
    for (unsigned int i=0; i<dim; ++i)
      omega_plus[i+1] = omega[i];
    // omega_plus[0] = 0.;
    // omega_plus[3] = 1.;
    //Next we can update the quaternion using qdot = S^-1 omega_plus
    FullMatrix<double> S_inv(dim+1,dim+1);
    //CHECK 0.5
    S_inv[0][0] = q[0];

    S_inv[1][0] = q[1];
    S_inv[2][0] = q[2];
    S_inv[3][0] = q[3];

    S_inv[0][1] = - q[1];
    S_inv[0][2] = - q[2];
    S_inv[0][3] = - q[3];

    S_inv[1][1] = q[0];
    S_inv[2][2] = q[0];
    S_inv[3][3] = q[0];

    S_inv[1][1] += 0.;
    S_inv[1][2] = q[3];
    S_inv[1][3] = -q[2];
    S_inv[2][1] = -q[3];
    S_inv[2][2] += 0.;
    S_inv[2][3] = q[1];
    S_inv[3][1] = q[2];
    S_inv[3][2] = -q[1];
    S_inv[3][3] += 0.;

    S_inv *= 0.5;
    // for(unsigned int i=0; i<dim+1; ++i)
    //   for(unsigned int j=0; j<dim+1; ++j)
    //     S_inv[i][j] = omega_plus[i] * q[j];

    S_inv.vmult(qdot, omega_plus);
    // S_inv.print(std::cout);
    // std::cout<<"old quat"<<std::endl;
    // q.print(std::cout);
    // std::cout<<"velocity"<<std::endl;
    // qdot.print(std::cout);
    // Vector<double> q_new(dim+1);
    q.sadd(1.,time_step,qdot);

    foo = std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    q /= (foo);

    // std::cout<<"new quat"<<std::endl;

    // q.print(std::cout);


    //Once we have the new quaternion we can update the rotation matrix using R = I + 2q0qx + 2qxqx
    // rotation[0][0] = 1. + 2 * q[0] * 0 + 2 * -(q[3]*q[3] + q[2]*q[2]);
    // rotation[0][1] = - 2 * q[0] * q[3] + 2 * (q[1] * q[2]);
    // rotation[0][2] = + 2 * q[0] * q[2] + 2 * (q[1] * q[3]);
    // rotation[1][0] = + 2 * q[0] * q[3] + 2 * (q[1] * q[2]);
    // rotation[1][1] = 1. + 2 * q[0] * 0 + 2 * -(q[3]*q[3] + q[1]*q[1]);
    // rotation[1][2] = - 2 * q[0] * q[1] + 2 * (q[3] * q[2]);
    // rotation[2][0] = - 2 * q[0] * q[2] + 2 * (q[1] * q[3]);
    // rotation[2][1] = + 2 * q[0] * q[1] + 2 * (q[3] * q[2]);
    // rotation[2][2] = 1. + 2 * q[0] * 0 + 2 * -(q[3]*q[3] + q[2]*q[2]);

    // IT SEEMS OK!

    rotation[0][0] = 1. + 2 * -(q[3]*q[3] + q[2]*q[2]);
    rotation[0][1] = - 2 * q[0] * q[3] + 2 * (q[1] * q[2]);
    rotation[0][2] = + 2 * q[0] * q[2] + 2 * (q[1] * q[3]);
    rotation[1][0] = + 2 * q[0] * q[3] + 2 * (q[1] * q[2]);
    rotation[1][1] = 1. + 2 * -(q[3]*q[3] + q[1]*q[1]);
    rotation[1][2] = - 2 * q[0] * q[1] + 2 * (q[3] * q[2]);
    rotation[2][0] = - 2 * q[0] * q[2] + 2 * (q[1] * q[3]);
    rotation[2][1] = + 2 * q[0] * q[1] + 2 * (q[3] * q[2]);
    rotation[2][2] = 1. + 2 * -(q[1]*q[1] + q[2]*q[2]);

    FullMatrix<double> foo_mat(dim,dim);

    rotation.Tmmult(foo_mat, rotation);

    double tol = 1e-7;
    for (unsigned int i=0; i<dim; ++i)
      for (unsigned int j=0; j<dim; ++j)
        {
          if (i == j)
            {
              if (std::fabs(foo_mat[i][j]-1) >= tol)
                pcout<<"Something Wrong in Rotations, on the diagonal "<<std::fabs(foo_mat[i][j]-1)<<std::endl;
            }
          else
            {
              if (std::fabs(foo_mat[i][j]) >=tol )
                pcout<<"Something Wrong in Rotations, out the diagonal "<<std::fabs(foo_mat[i][j])<<std::endl;

            }
        }

  }
  // From filename we read the grid on the next frame and we compute both the euler_vector to map the tria on the updated position and
  // the velocity.
  template <int dim>
  void PostProcessBEMStokes<dim>::compute_euler_vector(Vector<double> &euler, unsigned int frame, bool consider_rotations)
  {
    euler = 0;
    Triangulation<dim-1, dim> frame_tria;
    unsigned int k = fe_map->dofs_per_cell;
    DoFHandler<dim-1, dim> frame_map_dh(frame_tria);
    // std::vector<Point<dim> > frame_support_points(map_dh.n_dofs());
    bool print(true);
    // std::string filename = input_grid_base_name+Utilities::int_to_string(frame)+"."+input_grid_format;
    // pcout << "Analyzing file " << filename << std::endl;
    pcout << "Analyzing frame = "<< frame << " over " << n_frames << std::endl;
    // std::ifstream in;
    // in.open (filename);
    read_input_triangulation(stored_results_path+"euler_tria_"+Utilities::int_to_string(frame),"bin",frame_tria);
    frame_map_dh.distribute_dofs(*fe_map);
    DoFRenumbering::component_wise (frame_map_dh);

    AssertDimension(euler.size(), frame_map_dh.n_dofs());
    //MappingQEulerian<dim-1,Vector<double>, dim>  frame_mappingeul(mapping.get_degree(),euler,frame_map_dh);
    //DoFTools::map_dofs_to_support_points<dim-1, dim>( StaticMappingQ1<dim-1, dim>::mapping, frame_map_dh, frame_support_points);
    VectorTools::get_position_vector(frame_map_dh,euler);
    if (consider_rotations)
      for (types::global_dof_index i = 0; i<euler.size()/dim; ++i)
        {
          if (body_cpu_set.is_element(i))
            {
              Vector<double> pos(dim), new_pos(dim);
              for (unsigned int idim=0; idim<dim; ++idim)
                pos[idim] = euler[i+idim*euler.size()/dim];
              rotation_matrix.vmult(new_pos, pos);
              for (unsigned int idim=0; idim<dim; ++idim)
                euler[i+idim*euler.size()/dim] = new_pos[idim];
            }
          // if(i==10)
          // {
          //   std::cout<<"PIPPO"<<i*dim<<" "<<euler[30]<<" "<<euler[31]<<" "<<euler[32]<<std::endl;;
          //   pos.print(std::cout);
          //   rotation_matrix.print(std::cout);
          //   new_pos.print(std::cout);
          // }
        }

  }

  template<int dim>
  void PostProcessBEMStokes<dim>::read_computed_results(unsigned int frame)
  {

    std::string filename_vel, filename_forces, filename_shape_vel, filename_DN_rigid, filename_rigid, filename_total_vel;
    dpcout<<stored_results_path+"stokes_forces_" + Utilities::int_to_string(frame) + ".bin"<<std::endl;
    filename_forces = stored_results_path+"stokes_forces_" + Utilities::int_to_string(frame) + ".bin";
    std::ifstream forces(filename_forces.c_str());
    stokes_forces.block_read(forces);
    // cm_stokes.distribute(stokes_forces);

    filename_vel = stored_results_path+"stokes_rigid_vel_" + Utilities::int_to_string(frame) + ".bin";
    dpcout<<filename_vel<<std::endl;
    std::ifstream veloc(filename_vel.c_str());
    rigid_puntual_velocities.block_read(veloc);
    // cm_stokes.distribute(rigid_puntual_velocities);

    filename_shape_vel = stored_results_path+"shape_velocities_" + Utilities::int_to_string(frame) + ".bin";
    dpcout<<filename_shape_vel<<std::endl;
    std::ifstream s_vel(filename_shape_vel.c_str());
    shape_velocities.block_read(s_vel);
    // cm_stokes.distribute(shape_velocities);

    filename_total_vel = stored_results_path+"total_velocities_" + Utilities::int_to_string(frame) + ".bin";
    dpcout<<filename_total_vel<<std::endl;
    std::ifstream t_vel(filename_total_vel.c_str());
    total_velocities.block_read(t_vel);
    // cm_stokes.distribute(total_velocities);

    for (unsigned int i=0; i<num_rigid; ++i)
      {
        filename_DN_rigid = stored_results_path+"DN_rigid_mode_" + Utilities::int_to_string(i) + "_frame_"+Utilities::int_to_string(frame)+".bin";
        dpcout<<filename_DN_rigid<<std::endl;
        std::ifstream dn_i(filename_DN_rigid.c_str());
        DN_N_rigid[i].block_read(dn_i);
      }

    filename_rigid = stored_results_path+"4_6_rigid_velocities_"+Utilities::int_to_string(frame)+".bin";
    dpcout<<filename_rigid<<std::endl;
    std::ifstream rv46(filename_rigid.c_str());
    rigid_velocities.block_read(rv46);

    Vector<double> omega;

    if (dim == 3)
      {
        omega.reinit(3);
        omega[0] = rigid_velocities[3];
        omega[1] = rigid_velocities[4];
        omega[2] = rigid_velocities[5];
      }
    else if (dim == 2)
      {
        omega.reinit(1);
        omega[0] = rigid_velocities[2];

      }
    // rotation_matrix.print_formatted(std::cout);
    if (bool_rot)
      read_rotation_matrix(rotation_matrix,frame);
    // update_rotation_matrix(rotation_matrix, omega, time_step);


  }





  template<int dim>
  void PostProcessBEMStokes<dim>::read_rotation_matrix(FullMatrix<double> &rotation, const unsigned int frame)
  {

    Vector<double> q(dim*dim);
    std::string file_name1;
    file_name1 = "rotation_matrix_" + Utilities::int_to_string(frame) + ".bin";
    std::ifstream rot_mat (file_name1.c_str());
    q.block_read(rot_mat);

    for (int i = 0; i<dim; ++i)
      for (int j = 0; j<dim; ++j)
        rotation[i][j]=q[i*dim+j];

  }




  // template <int dim>
  // void PostProcessBEMStokes<dim>::reduce_exterior_results(const unsigned int frame)
  // {
  //   Vector<double> red_vel(external_grid_dimension*dim);
  //   MPI::COMM_WORLD.Reduce(&external_velocities(0), &red_vel(0),
  //                          external_velocities.size(), MPI_DOUBLE, MPI_SUM,
  //                          0);
  //   // Vector<double> dummy(4);
  //   // Vector<double> dummy_red(4);
  //   // for(unsigned int i = 4/size*rank; i<4/size*(rank+1); i++)
  //   //   dummy(i) = 4.;
  //   // MPI::COMM_WORLD.Reduce(&dummy(0), &dummy_red(0),
  //   //                             dummy.size(), MPI_DOUBLE, MPI_SUM,
  //   //                             0);
  //   // if(rank == 0)
  //   //   std::cout<< dummy_red << std::endl;
  //
  //   std::string file_name_vel;
  //   file_name_vel = "stokes_exterior_" + Utilities::int_to_string(frame) + ".bin";
  //   std::ofstream rvel (file_name_vel.c_str());
  //   red_vel.block_write(rvel);
  //
  // }


  template<int dim>
  Tensor<2, dim> PostProcessBEMStokes<dim>::compute_singular_kernel(const Tensor<1, dim> normal, const Tensor<3,dim> W)
  {
    Tensor<2,dim> singular_kernel;
    Tensor<2,dim> *result;
    for (unsigned  int i=0; i<dim; ++i)
      for (unsigned  int j=0; j<dim; ++j)
        for (unsigned  int k=0; k<dim; ++k)
          {
            singular_kernel[i][j] += W[i][j][k] * normal[k];
          }
    result=&singular_kernel;
    //std::cout<<"result ="<<*result<<" singular_kernel="<< singular_kernel<<std::endl;
    return *result;
  }


  template<int dim>
  void PostProcessBEMStokes<dim>::compute_real_forces_and_velocities()
  {
    real_stokes_forces=stokes_forces;
    // NOT NEEDED IF BEM MONOLITHIC
    // for(unsigned int i=0; i<num_rigid; ++i)
    // {
    //   real_stokes_forces.sadd(1., +rigid_velocities(i), DN_N_rigid[i]);
    // }
    pcout<<real_velocities.size()<<" "<<total_velocities.size()<<std::endl;
    real_velocities = total_velocities;
    // real_velocities.sadd(0.,1.,total_velocities);
    // pcout<<real_velocities.l2_norm()<<" : ";
    // real_velocities.sadd(0.,1.,shape_velocities);
    // real_velocities.sadd(1.,-1.,rigid_puntual_velocities);
    // pcout<<real_velocities.l2_norm()<<std::endl;
  }


  template<int dim>
  Tensor<2, dim> PostProcessBEMStokes<dim>::compute_G_kernel(const Tensor<1, dim> &R, const Tensor<1, dim> &R_image, const StokesKernel<dim> &stokes_kernel, const FreeSurfaceStokesKernel<dim> &fs_stokes_kernel, const NoSlipWallStokesKernel<dim> &ns_stokes_kernel, const bool reflect, const bool no_slip) const
  {
    Tensor<2, dim> G;
    if (reflect)
      {
        G = fs_stokes_kernel.value_tens_image(R,R_image);
        //  pcout<<"CLARABELLA"<<std::endl;

      }
    else if (no_slip)
      {
        G = ns_stokes_kernel.value_tens_image(R,R_image);
        // pcout<<"PIPPO"<<std::endl;

      }
    else
      {
        G = stokes_kernel.value_tens(R);
      }
    return G;
  }

  template<int dim>
  Tensor<3, dim> PostProcessBEMStokes<dim>::compute_W_kernel(const Tensor<1, dim> &R, const Tensor<1, dim> &R_image, const StokesKernel<dim> &stokes_kernel, const FreeSurfaceStokesKernel<dim> &fs_stokes_kernel, const NoSlipWallStokesKernel<dim> &ns_stokes_kernel, const bool reflect, const bool no_slip) const
  {
    Tensor<3, dim> W;
    if (reflect)
      {
        W = fs_stokes_kernel.value_tens_image2(R,R_image);
        //  pcout<<"MINNI"<<std::endl;
      }
    else if (no_slip)
      {
        W = ns_stokes_kernel.value_tens_image2(R,R_image);
        // pcout<<"TOPOLINO"<<std::endl;
      }
    else
      {
        W = stokes_kernel.value_tens2(R);
      }
    return W;
  }

  // @sect4{BEMProblem::compute_exterior_solution}

  // We'd like to also know something about the value of the potential $\phi$
  // in the exterior domain: after all our motivation to consider the boundary
  // integral problem was that we wanted to know the velocity in the exterior
  // domain!
  //
  // To this end, let us assume here that the boundary element domain is
  // contained in the box $[-2,2]^{\text{dim}}$, and we extrapolate the actual
  // solution inside this box using the convolution with the fundamental
  // solution. The formula for this is given in the introduction.
  //
  // The reconstruction of the solution in the entire space is done on a
  // continuous finite element grid of dimension dim. These are the usual
  // ones, and we don't comment any further on them. At the end of the
  // function, we output this exterior solution in, again, much the usual way.

  // template <int dim>
  // void PostProcessBEMStokes<dim>::evaluate_stokes_bie(
  //   const std::vector<Point<dim> > &val_points,
  //   const Vector<double> &vel,
  //   const Vector<double> &forces,
  //   Vector<double> &val_velocities)
  // {
  //   if (val_velocities.size() != val_points.size()*dim)
  //     val_velocities.reinit(val_points.size()*dim);
  //   typename DoFHandler<dim-1,dim>::active_cell_iterator
  //   cell = dh_stokes.begin_active(),
  //   endc = dh_stokes.end();
  //
  //
  //
  //   FEValues<dim-1,dim> fe_stokes_v(*mappingeul, *fe_stokes, quadrature,
  //                                   update_values |
  //                                   update_cell_normal_vectors |
  //                                   update_quadrature_points |
  //                                   update_JxW_values);
  //
  //   const unsigned int n_q_points_stokes = fe_stokes_v.n_quadrature_points;
  //
  //   pcout<<" Stokes Solution norms "<<forces.linfty_norm()<<" "<<forces.l2_norm()<<" , "
  //        <<vel.linfty_norm()<<" "<<vel.l2_norm()<<" "<<std::endl;
  //
  //   std::vector<Vector<double> > stokes_local_forces(n_q_points_stokes, Vector<double> (dim));
  //   std::vector<Vector<double> > stokes_local_velocities(n_q_points_stokes, Vector<double> (dim));
  //   fs_stokes_kernel.set_wall_orientation(kernel_wall_orientation);
  //   ns_stokes_kernel.set_wall_orientation(kernel_wall_orientation);
  //
  //   for (cell = dh_stokes.begin_active(); cell != dh_stokes.end(); ++cell)
  //     {
  //       fe_stokes_v.reinit(cell);
  //
  //       const std::vector<Point<dim> > &q_points = fe_stokes_v.get_quadrature_points();
  //       const std::vector<Tensor<1, dim> > &normals = fe_stokes_v.get_normal_vectors();
  //
  //       fe_stokes_v.get_function_values(forces, stokes_local_forces);
  //       fe_stokes_v.get_function_values(vel, stokes_local_velocities);
  //       //wind.vector_value_list(q_points, local_wind);
  //
  //
  //       for (types::global_dof_index i=proc_start; i<proc_end; ++i)
  //         {
  //           for (unsigned int q=0; q<n_q_points_stokes; ++q)
  //             {
  //
  //               const Tensor<1,dim> R = q_points[q] - val_points[i];
  //               Point<dim> support_point_image(val_points[i]);
  //               support_point_image[1] -= 2*(val_points[i][1]-kernel_wall_position[kernel_wall_orientation]);
  //               const Tensor<1,dim> R_image = q_points[q] - support_point_image;
  //               // std::cout<<q_image<<" "<<q_points[q]<<" "<<q_image - q_points[q]<<std::endl;
  //               // Point<dim> q_image(q_points[q]);
  //               // q_image[1] -= 2*(q_points[q][1]-wall_positions[0][1]);
  //               // const Tensor<1,dim> R_image = q_image - val_points[i];
  //               // // std::cout<<q_image<<" "<<q_points[q]<<" "<<q_image - q_points[q]<<std::endl;
  //               Tensor<2,dim> G = compute_G_kernel(R, R_image, stokes_kernel, fs_stokes_kernel, ns_stokes_kernel, reflect_kernel, no_slip_kernel); //stokes_kernel.value_tens(R) ;
  //               Tensor<3,dim> W = compute_W_kernel(R, R_image, stokes_kernel, fs_stokes_kernel, ns_stokes_kernel, reflect_kernel, no_slip_kernel);//LH_exterior_stokes_kernel.value_tens2(R) ;
  //               // Tensor<2,dim> G = fs_exterior_stokes_kernel.value_tens(R) ;
  //               // Tensor<3,dim> W = fs_exterior_stokes_kernel.value_tens2(R) ;
  //               // Tensor<2,dim> G = fs_exterior_stokes_kernel.value_tens_image(R,R_image) ;
  //               // Tensor<3,dim> W = fs_exterior_stokes_kernel.value_tens_image2(R,R_image) ;
  //               Tensor<2,dim> singular_ker = compute_singular_kernel(normals[q], W) ;
  //
  //               for (unsigned int idim = 0; idim < dim; ++idim)
  //                 {
  //                   for (unsigned int jdim = 0; jdim < dim; ++jdim)
  //                     {
  //                       val_velocities(i+val_velocities.size()/dim*idim) +=  G[idim][jdim] * //my_stokes_kernel.value(R, idim * dim * dim + jdim * dim) *
  //                                                                            stokes_local_forces[q](jdim) *
  //                                                                            fe_stokes_v.JxW(q) ;
  //                       val_velocities(i+val_velocities.size()/dim*idim) += singular_ker[idim][jdim] * //my_stokes_kernel.value(R, idim * dim * dim + jdim * dim) *
  //                                                                           stokes_local_velocities[q](jdim) *
  //                                                                           fe_stokes_v.JxW(q) ;
  //                       // if(idim == 1)
  //                       //  std::cout<<val_velocities(i+val_velocities.size()/dim*idim)<<" ";
  //                       //std::cout<<stokes_local_forces[q]<<" "<<i * dim + idim<<std::endl;
  //                     }
  //                 }
  //             }
  //         }
  //     }
  //
  //   for (types::global_dof_index i=0; i<external_velocities.size()/dim; ++i)
  //     {
  //       for (unsigned j=0; j<dim; ++j)
  //         mean_external_velocities(i+j*external_velocities.size()/dim)+=external_velocities(i+j*external_velocities.size()/dim);
  //
  //
  //     }
  //
  //   pcout<<val_points[0]<<" "<<val_velocities[0]<<" "<<val_velocities[val_velocities.size()/dim]<<" "<<val_velocities[val_velocities.size()/dim*2]<<" "<<std::endl;
  //   // val_velocities.print(std::cout);
  //
  //
  // }

  template <int dim>
  void PostProcessBEMStokes<dim>::compute_exterior_stokes_solution_on_grid()
  {
    // [  2.          -3.60467821   0.        ]
    //[  2.           3.00467821   0.        ]

    // Tensor<1,dim> body_center, point_up, point_down;
    // if (dim == 2)
    //   {
    //     body_center[0]=2.;
    //     body_center[1]=-0.3;
    //
    //     point_up[0]=2;
    //     point_up[1]=3.00467821;
    //
    //     point_down[0]=2;
    //     point_down[1]=-3.60467821;
    //   }
    // if (dim == 3)
    //   {
    //     body_center[0]=2.;
    //     body_center[1]=-0.3;
    //     body_center[2]=0.;
    //
    //     point_up[0]=2;
    //     point_up[1]=3.00467821;
    //     point_up[2]=0.;
    //
    //     point_down[0]=2;
    //     point_down[1]=-3.60467821;
    //     point_down[2]=0.;
    //   }

    typename DoFHandler<dim-1,dim>::active_cell_iterator
    cell = dh_stokes.begin_active(),
    endc = dh_stokes.end();


    FEValues<dim-1,dim> fe_stokes_v(*mappingeul, *fe_stokes, quadrature,
                                    update_values |
                                    update_cell_normal_vectors |
                                    update_quadrature_points |
                                    update_JxW_values);

    const unsigned int n_q_points_stokes = fe_stokes_v.n_quadrature_points;

    std::vector<types::global_dof_index> dofs(fe_stokes->dofs_per_cell);

    std::vector<Vector<double> > stokes_local_forces(n_q_points_stokes, Vector<double> (dim));
    std::vector<Vector<double> > stokes_local_shape_vel(n_q_points_stokes, Vector<double> (dim));
    fs_stokes_kernel.set_wall_orientation(kernel_wall_orientation);
    ns_stokes_kernel.set_wall_orientation(kernel_wall_orientation);

    // fs_stokes_kernel.set_wall_position_and_orientation(1.4, 1);
    // unsigned int fs_orientation = fs_stokes_kernel.get_wall_orientation();
    // double fs_position = fs_stokes_kernel.get_wall_position();
    //std::cout<<proc_start<<" "<<proc_end<<std::endl;
    pcout<<" Stokes Solution norms: forces "<<real_stokes_forces.linfty_norm()<<" "<<real_stokes_forces.l2_norm()<<" , velocities "
         <<real_velocities.linfty_norm()<<" "<<real_velocities.l2_norm()<<" "<<std::endl;
    external_velocities=0.;
    for (unsigned int i=proc_start; i<proc_end; ++i)
      {
        // std::cout<<i<<std::endl;
        for (cell = dh_stokes.begin_active(); cell != dh_stokes.end(); ++cell)
          {
            fe_stokes_v.reinit(cell);
            const std::vector<Point<dim> > &q_points = fe_stokes_v.get_quadrature_points();
            const std::vector<Tensor<1, dim> > &normals  = fe_stokes_v.get_normal_vectors();
            cell->get_dof_indices(dofs);
            // fe_stokes_v.get_function_values(stokes_forces, stokes_local_forces);
            // fe_stokes_v.get_function_values(shape_velocities, stokes_local_shape_vel);
            // std::cout<<"bubi"<<std::endl;
            // real_stokes_forces.print(std::cout);
            fe_stokes_v.get_function_values(real_stokes_forces, stokes_local_forces);
            fe_stokes_v.get_function_values(real_velocities, stokes_local_shape_vel);
            // std::cout<<"bubi"<<std::endl;

            for (unsigned int q=0; q<n_q_points_stokes; ++q)
              {
                const Tensor<1,dim> R = q_points[q] - external_grid[i];
                Point<dim> ext_point_image(external_grid[i]);
                ext_point_image[kernel_wall_orientation] -= 2*(ext_point_image[kernel_wall_orientation]-kernel_wall_position[kernel_wall_orientation]);//wall_positions[0][1]
                const Tensor<1,dim> R_image = q_points[q] - ext_point_image;
                // pcout<<R<< " ";
                // stokes_local_shape_vel[q].print(std::cout);
                // stokes_local_forces[q].print(std::cout);
                Tensor<2,dim> G = compute_G_kernel(R,R_image,stokes_kernel,fs_stokes_kernel,ns_stokes_kernel,reflect_kernel,no_slip_kernel); //stokes_kernel.value_tens(R);
                Tensor<3,dim> W = compute_W_kernel(R,R_image,stokes_kernel,fs_stokes_kernel,ns_stokes_kernel,reflect_kernel,no_slip_kernel); //stokes_kernel.value_tens2(R);
                // Tensor<2,dim> G = stokes_kernel.value_tens(R) ;
                // Tensor<3,dim> W = stokes_kernel.value_tens2(R) ;

                // pcout<<q_points[q]<<" : "<<external_grid[i]<<" : "<< R<<" : "<<R.norm_square()<<std::endl;
                Assert(R.norm_square() > 1e-6, ExcMessage("Error, R norm zero"));
                Tensor<2,dim> singular_ker = compute_singular_kernel(normals[q], W) ;
                // pcout<<normals[q].norm_square()<<" "<<G.norm_square()<<std::endl;
                for (unsigned int idim = 0; idim < dim; ++idim)
                  {
                    // HERE THE SIGNS APPEARS TO BE CORRECT.
                    for (unsigned int jdim = 0; jdim < dim; ++jdim)
                      {

                        // external_velocities(i * dim + idim)
                        // SIGN GOOD AND COHERENT WITH BEMStokes.
                        external_velocities(i + idim * external_velocities.size()/dim) += G[idim][jdim] * //my_stokes_kernel.value(R, idim * dim * dim + jdim * dim) *
                            stokes_local_forces[q](jdim) *
                            fe_stokes_v.JxW(q) ;
                        external_velocities(i + idim * external_velocities.size()/dim) += singular_ker[idim][jdim] * //my_stokes_kernel.value(R, idim * dim * dim + jdim * dim) *
                            stokes_local_shape_vel[q](jdim) *
                            fe_stokes_v.JxW(q) ;


                      }
                    // std::cout<<external_velocities(i + idim * external_velocities.size()/dim)<<" ";
                  }
              }
          }

        // Tensor<1,dim> dummy, dummy_up, dummy_down;
        // dummy=external_grid[i]-body_center;
        // dummy_up=external_grid[i]-point_up;
        // dummy_down=external_grid[i]-point_down;
        // //pcout<<body_center<<" "<<external_grid[i]<<std::endl;
        // if (dummy.norm()<0.01 || dummy_down.norm()<0.01 || dummy_up.norm()<0.01)
        //   {
        //     deallog << "Point "<<external_grid[i] << " " << proc_start << " "<< proc_end<< " " << i << std::endl;
        //     deallog << "computed vel "<<external_velocities(i) << " " << external_velocities(i+external_velocities.size()/dim) << std::endl;
        //     deallog << "rigid vel "<<rigid_puntual_velocities(0) << " " << rigid_puntual_velocities(external_velocities.size()/dim) << std::endl;
        //
        //   }
        //We normalize using the rigid puntual velocity of the zeroth point
        //for(unsigned int j=0; j<dim; ++j)
        //  external_velocities(i*dim+j)/=rigid_puntual_velocities(0+j)+1;

        if (velocity_kind == "BodyFrame")
          {
            if (dim == 3)
              {
                external_velocities[i+0*external_velocities.size()/dim] += rigid_velocities[4] * (0.+external_grid[i][2]) - rigid_velocities[5] * (0.+external_grid[i][1])-rigid_velocities[0];
                external_velocities[i+1*external_velocities.size()/dim] += rigid_velocities[5] * (0.+external_grid[i][0]) - rigid_velocities[3] * (0.+external_grid[i][2])-rigid_velocities[1];
                external_velocities[i+2*external_velocities.size()/dim] += rigid_velocities[3] * (0.+external_grid[i][1]) - rigid_velocities[4] * (0.+external_grid[i][0])-rigid_velocities[2];
              }
            else
              {
                external_velocities[i+0*external_velocities.size()/dim] += - rigid_velocities[2] * (0.+external_grid[i][1]) - rigid_velocities[0];
                external_velocities[i+1*external_velocities.size()/dim] +=   rigid_velocities[2] * (0.+external_grid[i][0]) - rigid_velocities[1];
              }
          }
        for (unsigned j=0; j<dim; ++j)
          mean_external_velocities(i+j*external_velocities.size()/dim)+=external_velocities(i+j*external_velocities.size()/dim);


      }
    // rigid_velocities.print(std::cout);
    // pcout<<external_grid[0]<<" "<<external_velocities[0]<<" "<<external_velocities[external_velocities.size()/dim/dim]<<" "<<external_velocities[external_velocities.size()/dim*2]<<" "<<std::endl;
    // external_velocities.print(std::cout);

  }
  template <int dim>
  void PostProcessBEMStokes<dim>::create_body_index_set()
  {
    typename DoFHandler<dim-1,dim>::active_cell_iterator
    cell = dh_stokes.begin_active(),
    endc = dh_stokes.end();

    std::vector<types::global_dof_index> local_dof_indices(fe_stokes->dofs_per_cell);

    body_cpu_set.clear();
    body_cpu_set.set_size(dh_stokes.n_dofs());
    for (cell = dh_stokes.begin_active(); cell<endc; ++cell)
      {
        // if(cell->subdomain_id() == this_mpi_process)
        {
          cell->get_dof_indices(local_dof_indices);
          if (cell->material_id() != 2 && cell->material_id() != 3)
            {
              for (unsigned int j=0; j<local_dof_indices.size(); ++j)
                {
                  body_cpu_set.add_index(local_dof_indices[j]);
                }

            }
        }
      }
    body_cpu_set.compress();
  }

  // template<int dim>
  // void PostProcessBEMStokes<dim>::remove_hanging_nodes_between_different_material_id( Triangulation<dim-1,dim> &tria_in,
  //     const bool isotropic,
  //     const unsigned int max_iterations)
  // {
  //   unsigned int iter = 0;
  //   bool continue_refinement = true;
  //
  //   typename Triangulation<dim-1, dim>::active_cell_iterator
  //   cell = tria_in.begin_active(),
  //   endc = tria_in.end();
  //
  //   while ( continue_refinement && (iter < max_iterations) )
  //     {
  //       if (max_iterations != numbers::invalid_unsigned_int) iter++;
  //       continue_refinement = false;
  //
  //       for (cell=tria_in.begin_active(); cell!= endc; ++cell)
  //         for (unsigned int j = 0; j < GeometryInfo<dim-1>::faces_per_cell; j++)
  //           if (cell->at_boundary(j)==false && cell->neighbor(j)->has_children() && (cell->neighbor(j)->material_id() != cell->material_id()))
  //             {
  //               if (isotropic)
  //                 {
  //                   cell->set_refine_flag();
  //                   continue_refinement = true;
  //                 }
  //               else
  //                 continue_refinement |= cell->flag_for_face_refinement(j);
  //             }
  //
  //       tria_in.execute_coarsening_and_refinement();
  //     }
  // }
  //
  // template <int dim>
  // void PostProcessBEMStokes<dim>::refine_walls(Triangulation<dim-1, dim> &triangulation, const double max_distance, const double threshold, const Point<dim> &center)
  // {
  //   bool refine = true;
  //   while (refine)
  //     {
  //       refine = false;
  //       typename Triangulation<dim-1,dim>::active_cell_iterator
  //       cell = triangulation.begin_active(),
  //       endc = triangulation.end();
  //       for (cell=triangulation.begin_active(); cell != endc; ++cell)
  //         {
  //           for (unsigned int v=0; v < GeometryInfo<dim-1>::vertices_per_cell; ++v)
  //             {
  //
  //               double distance_from_center = center.distance(cell->vertex(v));
  //               if (distance_from_center < max_distance && cell->diameter() > threshold)
  //                 {
  //                   cell->set_refine_flag();
  //                   refine = true;
  //                 }
  //               // else
  //               // pcout<<distance_from_center<<" "<<max_distance<<" : "<<cell->diameter()<<" "<<threshold<<std::endl;
  //               // break;
  //             }
  //         }
  //       triangulation.prepare_coarsening_and_refinement();
  //       triangulation.execute_coarsening_and_refinement();
  //     }
  //   typename Triangulation<dim-1,dim>::active_cell_iterator
  //   cell = triangulation.begin_active(),
  //   endc = triangulation.end();
  //
  //
  //
  //
  // }

  template <int dim>
  void PostProcessBEMStokes<dim>::convert_bool_parameters()
  {
    wall_bool[0] = wall_bool_0;
    wall_bool[1] = wall_bool_1;
    wall_bool[2] = wall_bool_2;
    wall_bool[3] = wall_bool_3;
    wall_bool[4] = wall_bool_4;
    wall_bool[5] = wall_bool_5;
    wall_bool[6] = wall_bool_6;
    wall_bool[7] = wall_bool_7;

    post_process_wall_bool[0] = post_process_wall_bool_0;
    post_process_wall_bool[1] = post_process_wall_bool_1;
    post_process_wall_bool[2] = post_process_wall_bool_2;
    post_process_wall_bool[3] = post_process_wall_bool_3;

  }
  // @sect4{BEMProblem::run}

  // This is the main function. It should be self explanatory in its
  // briefness:
  template <int dim>
  void PostProcessBEMStokes<dim>::run(unsigned int start_frame, unsigned int end_frame)
  {


    // As first step we read the parameter file.;
    // read_parameters("../parameters.prm");
    // pcout<<n_frames;
    if (dim==2)
      run_in_this_dimension=run_2d;
    else if (dim==3)
      run_in_this_dimension=run_3d;

    if (run_in_this_dimension == false)
      {
        pcout << "Run in dimension " << dim
              << " explicitly disabled in parameter file. "
              << std::endl;
        return;
      }

    // As first step we convert the bool parameters into vectors.
    convert_bool_parameters();

    // We retrieve the two Finite Element Systems
    fe_stokes = SP(parsed_fe_stokes());
    fe_map = SP(parsed_fe_mapping());
    grid_fe = SP(parsed_grid_fe());


    // read_domain();
    // if(read_cyl_bool)
    // {
    //   add_cylinder_to_tria(tria,cylinder_manifold_bool);
    // }
    // else if(read_box_bool)
    // {
    //   for(unsigned int i_wall=0; i_wall<first_index_box; ++i_wall)
    //   {
    //     if(wall_bool[i_wall]==true)
    //     {
    //       // pcout<<"Add wall "<<i_wall<<std::endl;
    //       add_wall_to_tria(tria,i_wall);
    //     }
    //   }
    //   add_box_to_tria(tria);
    //   for(unsigned int i_wall=first_index_box+6; i_wall<wall_bool.size(); ++i_wall)
    //   {
    //     if(wall_bool[i_wall]==true)
    //     {
    //       // pcout<<"Add wall "<<i_wall<<std::endl;
    //       add_wall_to_tria(tria,i_wall);
    //     }
    //   }
    //
    // }
    // else
    // {
    //   for(unsigned int i_wall=0; i_wall<wall_bool.size(); ++i_wall)
    //   {
    //     if(wall_bool[i_wall]==true)
    //     {
    //       // pcout<<"Add wall "<<i_wall<<std::endl;
    //       add_wall_to_tria(tria,i_wall);
    //     }
    //   }
    // }
    // refine_walls(tria,refine_distance_from_center,wall_threshold,refinement_center);
    // remove_hanging_nodes_between_different_material_id(tria);
    pcout<<"reading input tria"<<std::endl;
    // We load the complete reference triangulation stored as a binary file.
    pcout<<stored_results_path+"reference_tria"<<std::endl;
    read_input_triangulation(stored_results_path+"reference_tria","bin",tria);
    pcout<<refine_distance_from_center<<" "<<wall_threshold<<" "<<refinement_center<<std::endl;
    pcout<<"reinit"<<std::endl;
    reinit();
    // pcout<<"mmmmm "<<dh_stokes.n_dofs()<<std::endl;
    pcout<<"body index set"<<std::endl;
    create_body_index_set();
    pcout<<"read external grid"<<std::endl;
    read_external_grid(external_grid_filename, external_grid);
    pcout<<"compute proc props"<<std::endl;
    compute_processor_properties();
    pcout<<"compute euler vector"<<std::endl;
    compute_euler_vector(euler_vec, start_frame);
    mappingeul = SP(new MappingFEField<dim-1, dim>(map_dh,euler_vec));
    for (unsigned int i=start_frame; i<=end_frame; ++i)
      {
        if (i>start_frame)
          {
            compute_euler_vector(euler_vec, i);
          }
        pcout<< "Analyzing frame " << i << std::endl;
        pcout<< "Recovering computed results from " << stored_results_path << std::endl;
        read_computed_results(i);
        //shape_velocities.sadd(0., 1./time_step, next_euler_vec, -1./time_step, euler_vec);

        pcout<< "Computing the exterior solution on the grid " << external_grid_filename << std::endl;
        compute_real_forces_and_velocities();
        external_grid.resize(1);
        // std::vector<Point<dim> > new_ext_grid(1);
        // external_grid[0][0]=40;
        // external_grid[0][1]=4;
        // external_grid[0][2]=5;
        // external_grid_dimension=1;
        // compute_processor_properties();
        // pcout<<proc_start<<" "<<proc_end<<std::endl;
        // external_velocities.reinit(dim);
        // // real_stokes_forces.print(std::cout);
        // evaluate_stokes_bie(external_grid, real_velocities, real_stokes_forces, external_velocities);
        // external_velocities.print(std::cout);
        compute_exterior_stokes_solution_on_grid();
        // pcout<<"reduce"<<std::endl;
        // reduce_exterior_results(i);
        pcout<<"reduce and output"<<std::endl;
        reduce_output_grid_result(i);
        pcout<<"reinit"<<std::endl;
        reinit_for_new_frame(i);

      }
    pcout<<"Computing the average on the stroke"<<std::endl;
    compute_average(start_frame, end_frame);
    MPI::COMM_WORLD.Barrier();


  }
  template <int fdim>
  Point<fdim> blender_rotation(const Point<fdim> &p)
  {
    Assert(fdim==3,ExcNotImplemented());
    Point<fdim> q = p;
    q[2]=p[1];
    q[1]=0;
    return q;
  }

  template<int dim>
  void PostProcessBEMStokes<dim>::create_ext_grid(std::vector<Point<dim> > &ext_grid)
  {
    Point<dim> body_center;
    double body_diam=1.;//1.45;//1.65233910563;
    double span=20;
    std::vector<Point<dim> > vertices(4);


    // if(dim == 2)
    // {
    //   body_center[0]=2.;
    //   body_center[1]=-0.3;
    //   P1[0]=body_center[0]+span*body_diam;
    //   P1[1]=body_center[1]+span*body_diam;
    //   P2[0]=body_center[0]-span*body_diam;
    //   P2[1]=body_center[1]-span*body_diam;
    //
    //
    // }
    if (dim ==2)
      {
        if (post_process_wall_bool[0]==true)
          {
            pcout<<"Creating the single wall in 2d, we consder only post_process_wall 0"<<std::endl;
            Assert(wall_spans[0].size() == dim, ExcMessage("Incopatible span, size. Expected == dim"));

            // Triangulation<dim-1, dim> triangulation1(triangulation);
            Triangulation<dim, dim> triangulation_wall;
            std::vector<Point<dim> > vertices(2);
            // Point<dim> P1(position), P2(position), P3(position), P4(position);
            unsigned int foo_dim=numbers::invalid_unsigned_int, k=0;
            // std::vector<unsigned int> true_dim(dim-1);
            vertices[0] = wall_positions[0];
            vertices[1] = wall_positions[0];
            vertices[0][0]-=wall_spans[0][0];
            vertices[0][1]-=wall_spans[0][1];
            vertices[1][0]+=wall_spans[0][0];
            vertices[1][1]+=wall_spans[0][1];
            pcout<<vertices[0]<<std::endl;
            pcout<<vertices[1]<<std::endl;

            GridGenerator::hyper_rectangle(triangulation_wall,vertices[0],vertices[1]);

            triangulation_wall.refine_global(n_rep_ext_wall_ref);


            // pcout<<"GGGG"<<std::endl;

            if (this_mpi_process == 0)
              {
                std::string filename = "post_process_wall_"+Utilities::int_to_string(0)+".inp";
                std::ofstream wall_ofs;
                wall_ofs.open(filename, std::ofstream::out);
                GridOut go;
                go.write_ucd(triangulation_wall,wall_ofs);
              }
            // pcout<<"GGGG"<<std::endl;
            pcout<<"BUBI"<<std::endl;
            MPI_Barrier(mpi_communicator);
            // go.write_eps(triangulation_wall,wall_ofs);
            pcout<<"Adding the wall to the tria"<<std::endl;
            add_post_process_wall_to_tria(ext_tria,0);


          }
      }
    else if (dim == 3)
      {
        for (unsigned int i_wall=0; i_wall<post_process_wall_bool.size(); ++i_wall)
          {
            // pcout<<i_wall<<std::endl;
            if (post_process_wall_bool[i_wall])
              {
                pcout<<"creating post process wall with id "<<i_wall<<std::endl;
                pcout<<wall_positions[i_wall]<<std::endl;
                Assert(wall_spans[i_wall].size() == dim, ExcMessage("Incopatible span, size. Expected == dim"));

                // Triangulation<dim-1, dim> triangulation1(triangulation);
                Triangulation<dim-1, dim> triangulation_wall;

                double max_span = *std::max_element(wall_spans[i_wall].begin(),wall_spans[i_wall].end());
                double infinite_factor = 20.;
                std::vector<Point<dim> > vertices(4);
                // Point<dim> P1(position), P2(position), P3(position), P4(position);
                unsigned int foo_dim=numbers::invalid_unsigned_int, k=0;
                std::vector<unsigned int> true_dim(dim-1);

                vertices[0] = wall_positions[i_wall];
                vertices[1] = wall_positions[i_wall];
                vertices[2] = wall_positions[i_wall];
                vertices[3] = wall_positions[i_wall];
                // pcout<<"GGGG"<<std::endl;

                for (unsigned int i=0; i<dim; ++i)
                  {
                    if (wall_spans[i_wall][i] != -1. && wall_spans[i_wall][i] !=0)
                      {
                        vertices[0][i] += wall_spans[i_wall][i];
                        vertices[2][i] -= wall_spans[i_wall][i];
                      }
                    if (wall_spans[i_wall][i]==0)
                      {
                        foo_dim=i;
                      }
                    else
                      {
                        true_dim[k]=i;
                        k+=1;
                      }
                    // else
                    // {
                    //   P1[i] += max_span * infinite_factor;
                    //   P2[i] -= max_span * infinite_factor;
                    // }

                  }
                // pcout<<foo_dim<<std::endl;
                // pcout<<"GGGG"<<std::endl;
                Assert(foo_dim != numbers::invalid_unsigned_int, ExcMessage("A wall needs a zero dimension"))

                vertices[1][true_dim[0]]+=wall_spans[i_wall][true_dim[0]];
                vertices[1][true_dim[1]]-=wall_spans[i_wall][true_dim[1]];
                vertices[3][true_dim[0]]-=wall_spans[i_wall][true_dim[0]];
                vertices[3][true_dim[1]]+=wall_spans[i_wall][true_dim[1]];

                // pcout<<P1<<" "<<P2<<" "<<P3<<" "<<P4<<" "<<std::endl;
                // std::vector<unsigned int> repetitions(dim-1);
                // for(unsigned int i=0; i<dim-1; ++i)
                //   repetitions[i]=2;
                // repetitions[1]=2;

                std::vector<CellData<dim-1> > cells(1);
                SubCellData subcelldata;

                if (foo_dim == 1)
                  {
                    // pos_y>0=>ok
                    if (wall_positions[i_wall][foo_dim]>0)
                      {
                        cells[0].vertices[0]=0;
                        cells[0].vertices[1]=3;
                        cells[0].vertices[2]=1;
                        cells[0].vertices[3]=2;
                      }
                    else
                      {
                        cells[0].vertices[0]=0;
                        cells[0].vertices[1]=1;
                        cells[0].vertices[2]=3;
                        cells[0].vertices[3]=2;
                      }
                  }
                else
                  {
                    if (wall_positions[i_wall][foo_dim]>0)
                      {
                        cells[0].vertices[0]=0;
                        cells[0].vertices[1]=1;
                        cells[0].vertices[2]=3;
                        cells[0].vertices[3]=2;
                      }
                    else
                      {
                        cells[0].vertices[0]=0;
                        cells[0].vertices[1]=3;
                        cells[0].vertices[2]=1;
                        cells[0].vertices[3]=2;
                      }

                  }
                // pcout<<"GGGG"<<std::endl;

                const std::vector<Point<dim> > dummy_vertices(vertices);
                const std::vector<CellData<dim-1> > dummy_cells(cells);
                triangulation_wall.clear();
                triangulation_wall.create_triangulation(dummy_vertices, dummy_cells, subcelldata);
                triangulation_wall.refine_global(n_rep_ext_wall_ref);
                // pcout<<"GGGG"<<std::endl;

                if (this_mpi_process == 0)
                  {
                    std::string filename = "post_process_wall_"+Utilities::int_to_string(i_wall)+".inp";
                    std::ofstream wall_ofs;
                    wall_ofs.open(filename, std::ofstream::out);
                    GridOut go;
                    go.write_ucd(triangulation_wall,wall_ofs);
                  }
                // pcout<<"GGGG"<<std::endl;

                MPI_Barrier(mpi_communicator);
                // go.write_eps(triangulation_wall,wall_ofs);
                pcout<<"Adding the wall to the tria"<<std::endl;
                add_post_process_wall_to_tria(ext_tria,i_wall);
              }
          }

      }
    // std::vector<unsigned int> repetition(2);
    // repetition[0]=40;
    // repetition[1]=40;

    // FESystem<2,dim> gridfe(FE_Q<2,dim> (1),dim);
    // DoFHandler<2,dim> grid_dh(ext_tria);
    grid_dh.distribute_dofs(*grid_fe);
    // pcout<<"BUBU "<<grid_dh.n_dofs()<<std::endl;

    DoFRenumbering::component_wise(grid_dh);
    external_grid_dimension = grid_dh.n_dofs()/dim;
    external_velocities.reinit(dim*external_grid_dimension);
    mean_external_velocities.reinit(dim*external_grid_dimension);

    std::vector<Point<dim> > grid_support_points(grid_dh.n_dofs());
    ext_grid.resize(grid_dh.n_dofs()/dim);
    DoFTools::map_dofs_to_support_points<2,dim>( StaticMappingQ1<2,dim>::mapping, grid_dh, grid_support_points);

    for (unsigned int i=0; i<ext_grid.size(); ++i)
      ext_grid[i]=grid_support_points[i];
    pcout<<"External grid, cells "<<ext_tria.n_active_cells()<<" , dofs "<<ext_grid.size()<<std::endl;

  }

  template<int dim>
  void PostProcessBEMStokes<dim>::reduce_output_grid_result(const unsigned int frame)
  {
    Vector<double> ext_red_vel(external_velocities.size());
    MPI::COMM_WORLD.Reduce(&external_velocities(0), &ext_red_vel(0),
                           external_velocities.size(), MPI_DOUBLE, MPI_SUM,
                           0);


    if (rank==0)
      {
        std::string file_name_vel;
        file_name_vel = "stokes_exterior_" + Utilities::int_to_string(frame) + ".bin";
        std::ofstream rvel (file_name_vel.c_str());
        ext_red_vel.block_write(rvel);

        std::ofstream ofs_mean;
        std::string filename_mean;

        filename_mean="exterior_velocity_at_frame_"+Utilities::int_to_string(frame)+".txt";
        ofs_mean.open (filename_mean, std::ofstream::out | std::ofstream::app);


        for (unsigned int i = 0; i<external_grid_dimension; ++i)
          {
            for (unsigned int j = 0; j<dim ; ++j)
              {
                ofs_mean<<" "<<ext_red_vel[i+j*external_grid_dimension];
              }
            ofs_mean << std::endl;
          }
        ofs_mean.close();

        // std::vector<std::vector<double> > vv(dim, std::vector<double> (external_grid_dimension));
        //
        // for(unsigned int i = 0; i<external_grid_dimension; ++i)
        // {
        //   for(unsigned int j = 0; j<dim ; ++j)
        //   {
        //     vv[j][i]=ext_red_vel[i*dim+j];
        //   }
        // }
        //
        // Table<2, double> my_table_x(41, 41, vv[0].begin());
        // Table<2, double> my_table_y(41, 41, vv[1].begin());
        //
        // std_cxx11::array<std::pair<double,double>,2> endpoints;
        // endpoints[1] = std::make_pair (-31.04678211, 35.04678211);
        // endpoints[0] = std::make_pair (-33.34678211, 32.74678211);
        //
        // std_cxx11::array<unsigned int,2>             n_intervals;
        // n_intervals[0] = 40;
        // n_intervals[1] = 40;
        //
        // Functions::InterpolatedUniformGridData<2> my_int_func_on_grid_x(endpoints,n_intervals,my_table_x);
        // Functions::InterpolatedUniformGridData<2> my_int_func_on_grid_y(endpoints,n_intervals,my_table_y);
        //
        // Triangulation<2> ext_tria;
        // Point<2> P1(-31.0467821,-33.3467821);
        // Point<2> P2(35.0467821,32.7467821);
        // // if(dim == 2)
        // // {
        // //   P1[0]=-31.0467821;
        // //   P1[1]=-33.3467821;
        // //
        // //   P2[0]=35.0467821;
        // //   P2[1]=32.7467821;
        // //
        // // }
        // // else if(dim == 3)
        // // {
        // //   P1[0]=-31.0467821;
        // //   P1[1]=-33.3467821;
        // //
        // //   P2[2]=35.0467821;
        // //   P2[2]=32.7467821;
        // //
        // // }
        //
        // std::vector<unsigned int> repetition(2);
        // repetition[0]=40;
        // repetition[1]=40;
        // GridGenerator::subdivided_hyper_rectangle<2> (ext_tria, repetition, P1, P2);
        // FESystem<2> gridfe(FE_Q<2> (1),2);
        // DoFHandler<2> grid_dh(ext_tria);
        // grid_dh.distribute_dofs(gridfe);
        // std::vector<Point<2> > grid_support_points;
        // grid_support_points.resize(grid_dh.n_dofs());
        // Vector<double> composed_vel(grid_support_points.size());
        // DoFTools::map_dofs_to_support_points<2>( StaticMappingQ1<2>::mapping, grid_dh, grid_support_points);
        //
        // std::vector<double> vel_x(grid_support_points.size());
        // std::vector<double> vel_y(grid_support_points.size());
        //
        // my_int_func_on_grid_x.value_list(grid_support_points,vel_x);
        // my_int_func_on_grid_y.value_list(grid_support_points,vel_y);
        //
        // for(types::global_dof_index i =0; i<ext_red_vel.size()/dim; ++i)
        //   ext_red_vel[i] += rigid_puntual_velocities[0];
        pcout<<"done txt"<<std::endl;
        if (create_grid_in_deal)
          {
            // if(dim==3)
            //   for(unsigned int i=0; i<ext_red_vel.size()/dim; ++i)
            //     ext_red_vel(i+ext_red_vel.size()/dim)=0;

            std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation
            (dim, DataComponentInterpretation::component_is_part_of_vector);

            DataOut<2, DoFHandler<2, dim> > dataout;

            dataout.attach_dof_handler(grid_dh);
            dataout.add_data_vector(ext_red_vel, std::vector<std::string > (dim,"ext_vel"), DataOut<2, DoFHandler<2, dim> >::type_dof_data, data_component_interpretation);
            dataout.build_patches();

            std::string filename;
            filename="exterior_velocity_at_frame_"+Utilities::int_to_string(frame)+".vtu";
            std::ofstream file_vector(filename.c_str());
            dataout.write_vtu(file_vector);

            DataOut<dim-1, DoFHandler<dim-1, dim> > dataout_data;

            dataout_data.attach_dof_handler(dh_stokes);
            dataout_data.add_data_vector(real_stokes_forces, std::vector<std::string > (dim,"stokes_forces"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);

            // dataout_data.add_data_vector(rigid_puntual_velocities, std::vector<std::string > (dim,"rigid_vel"), DataOut<2, DoFHandler<2, dim> >::type_dof_data, data_component_interpretation);
            // dataout_data.add_data_vector(shape_velocities, std::vector<std::string > (dim,"shape_velocities"), DataOut<2, DoFHandler<2, dim> >::type_dof_data, data_component_interpretation);
            dataout_data.add_data_vector(real_velocities, std::vector<std::string > (dim,"real_velocities"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
            dataout_data.build_patches();

            std::string filename_data;
            filename_data="original_data_at_frame_"+Utilities::int_to_string(frame)+".vtu";
            std::ofstream file_vector_data(filename_data.c_str());
            dataout_data.write_vtu(file_vector_data);

          }

      }



  }

  // In this function we simply prepare all our vectors-matrices for a
  // new cycle. We can't simply reinit every unkwown. We update also
  // the euler_vec vector.
  template<int dim>
  void PostProcessBEMStokes<dim>::reinit_for_new_frame(unsigned int frame)
  {
    compute_euler_vector(euler_vec,(frame+1)%n_frames);
    external_velocities = 0.;

  }


  template<int dim>
  void PostProcessBEMStokes<dim>::compute_average(unsigned int start_frame, unsigned int final_frame)
  {
    Vector<double> mean_red_vel(external_grid_dimension*dim);
    Vector<double> norm_mean_vel(external_grid_dimension);
    MPI::COMM_WORLD.Reduce(&mean_external_velocities(0), &mean_red_vel(0),
                           mean_external_velocities.size(), MPI_DOUBLE, MPI_SUM,
                           0);
    mean_red_vel /= (final_frame-start_frame+1);
    if (rank==0)
      {

        for (unsigned int i=0; i<external_grid_dimension; ++i)
          {
            //norm_mean_vel(i)=0;
            if (dim==2)
              {
                norm_mean_vel(i)=sqrt(mean_red_vel(i)*mean_red_vel(i)+
                                      mean_red_vel(i+external_grid_dimension)*mean_red_vel(i+external_grid_dimension));
              }
            else if (dim==3)
              {
                norm_mean_vel(i)=sqrt(mean_red_vel(i)*mean_red_vel(i)+
                                      mean_red_vel(i+external_grid_dimension)*mean_red_vel(i+external_grid_dimension)+
                                      mean_red_vel(i+2*external_grid_dimension)*mean_red_vel(i+2*external_grid_dimension));
              }
          }

        std::ofstream ofs_mean;
        std::string filename_mean;

        filename_mean="exterior_mean_from_"+Utilities::int_to_string(start_frame)+"_to_"+Utilities::int_to_string(final_frame)+".txt";
        ofs_mean.open (filename_mean, std::ofstream::out | std::ofstream::app);

        std::cout<<mean_red_vel.size()<<std::endl;
        for (unsigned int i = 0; i<norm_mean_vel.size(); ++i)
          {
            ofs_mean << norm_mean_vel(i);
            for (unsigned int j = 0; j<dim ; ++j)
              ofs_mean<<" "<<mean_red_vel[i+j*norm_mean_vel.size()];

            for (unsigned int j = 0; j<dim ; ++j)
              ofs_mean<<" "<<external_grid[i][j];
            ofs_mean << std::endl;
          }
        ofs_mean.close();

        std::string file_name_mean_bin;
        file_name_mean_bin = "stokes_exterior_mean_from_" + Utilities::int_to_string(start_frame) +"_to_"+Utilities::int_to_string(final_frame)+ ".bin";
        std::ofstream rmean (file_name_mean_bin.c_str());
        mean_red_vel.block_write(rmean);
      }
    if (create_grid_in_deal)
      {
        if (dim==3)
          for (unsigned int i=0; i<mean_red_vel.size()/dim; ++i)
            mean_red_vel(i+mean_red_vel.size()/dim)=0;

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation
        (dim, DataComponentInterpretation::component_is_part_of_vector);

        DataOut<2, DoFHandler<2, dim> > dataout;

        dataout.attach_dof_handler(grid_dh);
        dataout.add_data_vector(mean_red_vel, std::vector<std::string > (dim,"ext_vel"), DataOut<2, DoFHandler<2, dim> >::type_dof_data, data_component_interpretation);
        dataout.build_patches();

        std::string filename;
        filename="exterior_mean_velocity_frame_"+Utilities::int_to_string(start_frame)+"_"+Utilities::int_to_string(final_frame)+".vtu";
        std::ofstream file_vector(filename.c_str());

        dataout.write_vtu(file_vector);
      }

  }



}

template class PostProcess::PostProcessBEMStokes<2>;
template class PostProcess::PostProcessBEMStokes<3>;
