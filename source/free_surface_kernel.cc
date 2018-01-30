#include "free_surface_kernel.h"

DEAL_II_NAMESPACE_OPEN

template <int dim>
FreeSurfaceStokesKernel<dim>::FreeSurfaceStokesKernel(const double eps)
  :
  epsilon(eps)
{}


template <>
Tensor<2, 2, double>
FreeSurfaceStokesKernel<2>::value_tens_image(const Tensor<1,2,double> &, const Tensor<1,2,double> &) const
{
  AssertThrow(false, ExcImpossibleInDim(2));
  return Tensor<2, 2, double> ();
}
template <int dim>
Tensor<2, dim, double>
FreeSurfaceStokesKernel<dim>::value_tens_image(const Tensor<1,dim,double> &p, const Tensor<1,dim,double> &p_image) const
{

  double R, R_image;
  R = 0.;
  R_image = 0.;
  for (unsigned int i=0; i<dim; ++i)
    {
      R += p[i]*p[i];
      R_image += p_image[i]*p_image[i];
    }
  R = sqrt(R) + epsilon;
  R_image = sqrt(R_image) + epsilon;

  Tensor<2,dim, double> G;
  for (unsigned int i=0; i<dim; ++i)
    {
      if (i==wall_orientation)
        {
          for (unsigned int j=0; j<dim; ++j)
            {
              double delta = 1.0*(i==j);
              if (dim ==2)
                G[i][j] = (p[i]*p[j]/(R*R)-delta*std::log(R)) -
                          (p_image[i]*p_image[j]/(R_image*R_image)-delta*std::log(R_image));
              else
                G[i][j] = (p[i]*p[j]/(R*R*R)+delta/R) -
                          (p_image[i]*p_image[j]/(R_image*R_image*R_image)+delta/R_image);

              G[i][j] /= (4*numbers::PI*(dim-1));

            }
        }
      else
        {
          for (unsigned int j=0; j<dim; ++j)
            {
              double delta = 1.0*(i==j);
              if (dim ==2)
                G[i][j] = (p[i]*p[j]/(R*R)-delta*std::log(R)) +
                          (p_image[i]*p_image[j]/(R_image*R_image)-delta*std::log(R_image));
              else
                G[i][j] = (p[i]*p[j]/(R*R*R)+delta/R) +
                          (p_image[i]*p_image[j]/(R_image*R_image*R_image)+delta/R_image);
              G[i][j] /= (4*numbers::PI*(dim-1));

            }
        }
    }
  return G;

}
template <>
Tensor<2, 2, double>
FreeSurfaceStokesKernel<2>::value_tens_image_old(const Tensor<1,2,double> &, const Tensor<1,2,double> &) const
{
  AssertThrow(false, ExcImpossibleInDim(2));
  return Tensor<2, 2, double> ();
}
template <int dim>
Tensor<2, dim, double>
FreeSurfaceStokesKernel<dim>::value_tens_image_old(const Tensor<1,dim,double> &p, const Tensor<1,dim,double> &p_image) const
{

  double R, R_image;
  R = 0.;
  R_image = 0.;
  for (unsigned int i=0; i<dim; ++i)
    {
      R += p[i]*p[i];
      R_image += p_image[i]*p_image[i];
    }
  R = sqrt(R) + epsilon;
  R_image = sqrt(R_image) + epsilon;

  Tensor<2,dim, double> G;

  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      {
        double delta = 1.0*(i==j);
        if (j==wall_orientation)
          {
            if (dim ==2)
              G[i][j] = (p[i]*p[j]/(R*R)-delta*std::log(R)) -
                        (p_image[i]*p_image[j]/(R_image*R_image)-delta*std::log(R_image));
            else
              G[i][j] = (p[i]*p[j]/(R*R*R)+delta/R) -
                        (p_image[i]*p_image[j]/(R_image*R_image*R_image)+delta/R_image);


          }
        else
          {
            if (dim ==2)
              G[i][j] = (p[i]*p[j]/(R*R)-delta*std::log(R)) +
                        (p_image[i]*p_image[j]/(R_image*R_image)-delta*std::log(R_image));
            else
              G[i][j] = (p[i]*p[j]/(R*R*R)+delta/R) +
                        (p_image[i]*p_image[j]/(R_image*R_image*R_image)+delta/R_image);
          }
        G[i][j] /= (4*numbers::PI*(dim-1));
      }
  return G;

}

template <>
Tensor<3, 2, double>
FreeSurfaceStokesKernel<2>::value_tens_image2(const Tensor<1,2,double> &, const Tensor<1,2,double> &) const
{
  AssertThrow(false, ExcImpossibleInDim(2));
  return Tensor<3, 2, double> ();
}
template <int dim>
Tensor<3, dim, double>
FreeSurfaceStokesKernel<dim>::value_tens_image2(const Tensor<1,dim,double> &p, const Tensor<1,dim,double> &p_image) const
{

  double R, R_image;
  R = 0.;
  R_image = 0.;
  for (unsigned int i=0; i<dim; ++i)
    {
      R += p[i]*p[i];
      R_image += p_image[i]*p_image[i];
    }
  R = sqrt(R) + epsilon;
  R_image = sqrt(R_image) + epsilon;

  Tensor<3,dim, double> W;
  for (unsigned int i=0; i<dim; ++i)
    {
      if (i==wall_orientation)
        {
          for (unsigned int j=0; j<dim; ++j)
            {
              for (unsigned int k=0; k<dim; ++k)
                {

                  W[i][j][k] = -dim*p[i]*p[j]*p[k]/(std::pow(R, dim+2))/(2*numbers::PI*(dim-1));
                  W[i][j][k] -= -dim*p_image[i]*p_image[j]*p_image[k]/(std::pow(R_image, dim+2))/(2*numbers::PI*(dim-1));
                }
            }
        }
      else
        {
          for (unsigned int j=0; j<dim; ++j)
            {
              for (unsigned int k=0; k<dim; ++k)
                {

                  W[i][j][k] = -dim*p[i]*p[j]*p[k]/(std::pow(R, dim+2))/(2*numbers::PI*(dim-1));
                  W[i][j][k] += -dim*p_image[i]*p_image[j]*p_image[k]/(std::pow(R_image, dim+2))/(2*numbers::PI*(dim-1));
                }
            }
        }
    }
  //   for(unsigned int i=0; i<dim; ++i)
  // for(unsigned int j=0; j<dim; ++j) {
  //     if(j==wall_orientation)
  //     {
  //       for(unsigned int k=0; k<dim; ++k)
  //       {
  //
  //         W[i][j][k] = -dim*p[i]*p[j]*p[k]/(std::pow(R, dim+2))/(2*numbers::PI*(dim-1));
  //         if(k==wall_orientation)
  //           W[i][j][k] += -dim*p_image[i]*p_image[j]*p_image[k]/(std::pow(R_image, dim+2))/(2*numbers::PI*(dim-1));
  //         else
  //           W[i][j][k] -= -dim*p_image[i]*p_image[j]*p_image[k]/(std::pow(R_image, dim+2))/(2*numbers::PI*(dim-1));
  //       }
  //
  //     }
  //     else
  //     {
  //       for(unsigned int k=0; k<dim; ++k)
  //       {
  //
  //         W[i][j][k] = -dim*p[i]*p[j]*p[k]/(std::pow(R, dim+2))/(2*numbers::PI*(dim-1));
  //         if(k==wall_orientation)
  //           W[i][j][k] -= -dim*p_image[i]*p_image[j]*p_image[k]/(std::pow(R_image, dim+2))/(2*numbers::PI*(dim-1));
  //         else
  //           W[i][j][k] += -dim*p_image[i]*p_image[j]*p_image[k]/(std::pow(R_image, dim+2))/(2*numbers::PI*(dim-1));
  //       }
  //     }
  // }
  return W;

}

template <>
Tensor<2, 2, double>
FreeSurfaceStokesKernel<2>::value_tens_image_pimponi(const Tensor<1,2,double> &, const Tensor<1,2,double> &) const
{
  AssertThrow(false, ExcImpossibleInDim(2));
  return Tensor<2, 2, double> ();
}
template <int dim>
Tensor<2, dim, double>
FreeSurfaceStokesKernel<dim>::value_tens_image_pimponi(const Tensor<1,dim,double> &p, const Tensor<1,dim,double> &p_image) const
{

  double R, R_image;
  R = 0.;
  R_image = 0.;
  for (unsigned int i=0; i<dim; ++i)
    {
      R += p[i]*p[i];
      R_image += p_image[i]*p_image[i];
    }
  R = sqrt(R) + epsilon;
  R_image = sqrt(R_image) + epsilon;

  Tensor<2,dim, double> G;
  for (unsigned int i=0; i<dim; ++i)
    {
      for (unsigned int j=0; j<dim; ++j)
        {
          double delta = 1.0*(i==j);
          if (j==wall_orientation)
            {
              if (dim ==2)
                G[i][j] = (p[i]*p[j]/(R*R)-delta*std::log(R)) -
                          (p_image[i]*p_image[j]/(R_image*R_image)-delta*std::log(R_image));
              else
                G[i][j] = (p[i]*p[j]/(R*R*R)+delta/R) -
                          (p_image[i]*p_image[j]/(R_image*R_image*R_image)+delta/R_image);
            }
          else
            {
              if (dim ==2)
                G[i][j] = (p[i]*p[j]/(R*R)-delta*std::log(R)) +
                          (p_image[i]*p_image[j]/(R_image*R_image)-delta*std::log(R_image));
              else
                G[i][j] = (p[i]*p[j]/(R*R*R)+delta/R) +
                          (p_image[i]*p_image[j]/(R_image*R_image*R_image)+delta/R_image);

            }


          G[i][j] /= (4*numbers::PI*(dim-1));

        }
    }

  return G;

}

template <>
Tensor<3, 2, double>
FreeSurfaceStokesKernel<2>::value_tens_image2_pimponi(const Tensor<1,2,double> &, const Tensor<1,2,double> &) const
{
  AssertThrow(false, ExcImpossibleInDim(2));
  return Tensor<3, 2, double> ();
}
template <int dim>
Tensor<3, dim, double>
FreeSurfaceStokesKernel<dim>::value_tens_image2_pimponi(const Tensor<1,dim,double> &p, const Tensor<1,dim,double> &p_image) const
{

  double R, R_image;
  R = 0.;
  R_image = 0.;
  for (unsigned int i=0; i<dim; ++i)
    {
      R += p[i]*p[i];
      R_image += p_image[i]*p_image[i];
    }
  R = sqrt(R) + epsilon;
  R_image = sqrt(R_image) + epsilon;

  Tensor<3,dim, double> W;
  for (unsigned int i=0; i<dim; ++i)
    {
      for (unsigned int j=0; j<dim; ++j)
        {
          if (j==wall_orientation)
            {
              for (unsigned int k=0; k<dim; ++k)
                {

                  W[i][j][k] = -dim*p[i]*p[j]*p[k]/(std::pow(R, dim+2))/(2*numbers::PI*(dim-1));
                  W[i][j][k] -= -dim*p_image[i]*p_image[j]*p_image[k]/(std::pow(R_image, dim+2))/(2*numbers::PI*(dim-1));
                }
            }
          else
            {
              for (unsigned int k=0; k<dim; ++k)
                {

                  W[i][j][k] = -dim*p[i]*p[j]*p[k]/(std::pow(R, dim+2))/(2*numbers::PI*(dim-1));
                  W[i][j][k] += -dim*p_image[i]*p_image[j]*p_image[k]/(std::pow(R_image, dim+2))/(2*numbers::PI*(dim-1));
                }
            }
        }
    }
  return W;

}

template <>
Tensor<3, 2, double>
FreeSurfaceStokesKernel<2>::value_tens_image2_old(const Tensor<1,2,double> &, const Tensor<1,2,double> &) const
{
  AssertThrow(false, ExcImpossibleInDim(2));
  return Tensor<3, 2, double> ();
}
template <int dim>
Tensor<3, dim, double>
FreeSurfaceStokesKernel<dim>::value_tens_image2_old(const Tensor<1,dim,double> &p, const Tensor<1,dim,double> &p_image) const
{

  double R, R_image;
  R = 0.;
  R_image = 0.;
  for (unsigned int i=0; i<dim; ++i)
    {
      R += p[i]*p[i];
      R_image += p_image[i]*p_image[i];
    }
  R = sqrt(R) + epsilon;
  R_image = sqrt(R_image) + epsilon;

  Tensor<3,dim, double> W;
  // for(unsigned int i=0; i<dim; ++i)
  // {
  //   if(i==wall_orientation)
  //   {
  //     for(unsigned int j=0; j<dim; ++j)
  //     {
  //           for(unsigned int k=0; k<dim; ++k)
  //           {
  //
  //             W[i][j][k] = -dim*p[i]*p[j]*p[k]/(std::pow(R, dim+2))/(2*numbers::PI*(dim-1));
  //             W[i][j][k] -= -dim*p_image[i]*p_image[j]*p_image[k]/(std::pow(R_image, dim+2))/(2*numbers::PI*(dim-1));
  //           }
  //     }
  //   }
  //   else
  //   {
  //     for(unsigned int j=0; j<dim; ++j)
  //     {
  //           for(unsigned int k=0; k<dim; ++k)
  //           {
  //
  //             W[i][j][k] = -dim*p[i]*p[j]*p[k]/(std::pow(R, dim+2))/(2*numbers::PI*(dim-1));
  //             W[i][j][k] += -dim*p_image[i]*p_image[j]*p_image[k]/(std::pow(R_image, dim+2))/(2*numbers::PI*(dim-1));
  //           }
  //     }
  //   }
  // }
  //
  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      {
        if (j==wall_orientation)
          {
            for (unsigned int k=0; k<dim; ++k)
              {

                W[i][j][k] = -dim*p[i]*p[j]*p[k]/(std::pow(R, dim+2))/(2*numbers::PI*(dim-1));
                if (k==wall_orientation)
                  W[i][j][k] += -dim*p_image[i]*p_image[j]*p_image[k]/(std::pow(R_image, dim+2))/(2*numbers::PI*(dim-1));
                else
                  W[i][j][k] -= -dim*p_image[i]*p_image[j]*p_image[k]/(std::pow(R_image, dim+2))/(2*numbers::PI*(dim-1));
              }

          }
        else
          {
            for (unsigned int k=0; k<dim; ++k)
              {

                W[i][j][k] = -dim*p[i]*p[j]*p[k]/(std::pow(R, dim+2))/(2*numbers::PI*(dim-1));
                if (k==wall_orientation)
                  W[i][j][k] -= -dim*p_image[i]*p_image[j]*p_image[k]/(std::pow(R_image, dim+2))/(2*numbers::PI*(dim-1));
                else
                  W[i][j][k] += -dim*p_image[i]*p_image[j]*p_image[k]/(std::pow(R_image, dim+2))/(2*numbers::PI*(dim-1));
              }
          }
      }
  return W;

}

template <int dim>
void
FreeSurfaceStokesKernel<dim>::set_wall_orientation(const unsigned int orientation_in)
{
  wall_orientation = orientation_in;
}

template <int dim>
unsigned int
FreeSurfaceStokesKernel<dim>::get_wall_orientation()
{
  return wall_orientation;
}
template class FreeSurfaceStokesKernel<2>;
template class FreeSurfaceStokesKernel<3>;


DEAL_II_NAMESPACE_CLOSE
