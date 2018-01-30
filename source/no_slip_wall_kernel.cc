#include "no_slip_wall_kernel.h"


DEAL_II_NAMESPACE_OPEN

template <int dim>
NoSlipWallStokesKernel<dim>::NoSlipWallStokesKernel(const double eps)
  :
  epsilon(eps)
{}



template <>
Tensor<2, 2, double>
NoSlipWallStokesKernel<2>::value_tens_image(const Tensor<1,2,double> &, const Tensor<1,2,double> &) const
{
  AssertThrow(false, ExcImpossibleInDim(2));
  return Tensor<2, 2, double> ();
}


template <int dim>
Tensor<2, dim, double>
NoSlipWallStokesKernel<dim>::value_tens_image(const Tensor<1,dim,double> &p, const Tensor<1,dim,double> &p_image) const
{

  double R, R_image;
  double h_0 = 0.5*(p_image[wall_orientation]-p[wall_orientation]);
  // std::cout<<wall_orientation<<std::endl;
  // std::cout<<p[wall_orientation]<<" "<<p_image[wall_orientation]<<" "<<h_0<<std::endl;
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
  double X3=h_0;
  double x3=p[wall_orientation]+h_0;

  for (unsigned int i=0; i<dim; ++i)
    {
      double delta_i1 = 1.0*(i==wall_orientation);
      for (unsigned int j=0; j<dim; ++j)
        {
          double delta = 1.0*(i==j);
          double delta_j1 = 1.0*(j==wall_orientation);
          G[i][j]=1./8./numbers::PI*(delta*(1/R-1/R_image)+p[i]*p[j]/(R*R*R)-p_image[i]*p_image[j]/(R_image*R_image*R_image)
                                     +2.*X3/(R_image*R_image*R_image)*(delta_j1*p_image[i]+delta_i1*p_image[j]-2*delta_i1*delta_j1*p_image[wall_orientation]+
                                         x3*(2*delta_i1*delta_j1-delta+3.*p_image[i]/(R_image*R_image)*(p_image[j]-2*delta_j1*p_image[wall_orientation]))));
        }
    }
  // for(unsigned int i=0; i<dim; ++i)
  // {
  //   if(i==wall_orientation)
  //   {
  //     for(unsigned int j=0; j<dim; ++j)
  //     {
  //         double delta = 1.0*(i==j);
  //         double delta_i1 = 1.0*(i==wall_orientation);
  //         double delta_j1 = 1.0*(j==wall_orientation);
  //         if(dim ==2)
  //             G[i][j] = (p[i]*p[j]/(R*R)-delta*std::log(R)) -
  //                       (p_image[i]*p_image[j]/(R_image*R_image)-delta*std::log(R_image));
  //         else
  //             G[i][j] = (p[i]*p[j]/(R*R*R)+delta/R) -
  //                       (p_image[i]*p_image[j]/(R_image*R_image*R_image)+delta/R_image) -
  //                       2.*h_0*h_0*(-3*p_image[i]*p_image[j]/(R_image*R_image*R_image*R_image*R_image)+delta/(R_image*R_image*R_image)) +
  //                       2.*h_0*(p_image[wall_orientation]*(-3*p_image[i]*p_image[j]/(R_image*R_image*R_image*R_image*R_image)+delta/(R_image*R_image*R_image))+
  //                            ((delta_i1*p_image[j]-delta_j1*p_image[i])/(R_image*R_image*R_image)));
  //
  //                           //  G[i][j] = (p[i]*p[j]/(R*R*R)+delta/R) -
  //                           //            (p_image[i]*p_image[j]/(R_image*R_image*R_image)+delta/R_image) +
  //                           //            2.*h_0*h_0*                       (+3*p_image[i]*p_image[j]/(R_image*R_image*R_image*R_image*R_image)-delta/(R_image*R_image*R_image))+
  //                           //            2.*h_0*(p_image[wall_orientation]*(+3*p_image[i]*p_image[j]/(R_image*R_image*R_image*R_image*R_image)-delta/(R_image*R_image*R_image))+
  //                           //                    ((-(delta_i1*p_image[j]+delta_j1*p_image[i])*(1-delta))/(R_image*R_image*R_image)));//(delta_i1*p_image[j]+delta_j1*p_image[i])
  //        G[i][j] /= (4*numbers::PI*(dim-1));
  //
  //     }
  //   }
  //   else
  //   {
  //     for(unsigned int j=0; j<dim; ++j)
  //     {
  //       double delta = 1.0*(i==j);
  //       double delta_i1 = 1.0*(i==wall_orientation);
  //       double delta_j1 = 1.0*(j==wall_orientation);
  //       if(dim ==2)
  //           G[i][j] = (p[i]*p[j]/(R*R)-delta*std::log(R)) +
  //                     (p_image[i]*p_image[j]/(R_image*R_image)-delta*std::log(R_image));
  //       else
  //       G[i][j] = (p[i]*p[j]/(R*R*R)+delta/R) -
  //                 (p_image[i]*p_image[j]/(R_image*R_image*R_image)+delta/R_image) +
  //                 2.*h_0*h_0*(-3*p_image[i]*p_image[j]/(R_image*R_image*R_image*R_image*R_image)+delta/(R_image*R_image*R_image))-
  //                 2.*h_0*(p_image[wall_orientation]*(-3*p_image[i]*p_image[j]/(R_image*R_image*R_image*R_image*R_image)+delta/(R_image*R_image*R_image))+
  //                         ((delta_i1*p_image[j]-delta_j1*p_image[i])/(R_image*R_image*R_image)));// CHECK SEGNI!!!
  //
  //                         // G[i][j] = (p[i]*p[j]/(R*R*R)+delta/R) -
  //                         //           (p_image[i]*p_image[j]/(R_image*R_image*R_image)+delta/R_image) +
  //                         //           2.*h_0*h_0*                       (-3*p_image[i]*p_image[j]/(R_image*R_image*R_image*R_image*R_image)+delta/(R_image*R_image*R_image))+
  //                         //           2.*h_0*(p_image[wall_orientation]*(-3*p_image[i]*p_image[j]/(R_image*R_image*R_image*R_image*R_image)+delta/(R_image*R_image*R_image))+
  //                         //                   ((-(delta_i1*p_image[j]+delta_j1*p_image[i])*(1-delta))/(R_image*R_image*R_image)));
  //
  //       G[i][j] /= (4*numbers::PI*(dim-1));
  //
  //     }
  //   }
  // }
  return G;

}


template <>
Tensor<3, 2, double>
NoSlipWallStokesKernel<2>::value_tens_image2(const Tensor<1,2,double> &, const Tensor<1,2,double> &) const
{
  AssertThrow(false, ExcImpossibleInDim(2));
  return Tensor<3, 2, double> ();
}

template <int dim>
Tensor<3, dim, double>
NoSlipWallStokesKernel<dim>::value_tens_image2(const Tensor<1,dim,double> &p, const Tensor<1,dim,double> &p_image) const
{

  double R, R_image;
  R = 0.;
  R_image = 0.;
  double h_0 = 0.5*(p_image[wall_orientation]-p[wall_orientation]);
  // std::cout<<h_0<<std::endl;
  for (unsigned int i=0; i<dim; ++i)
    {
      R += p[i]*p[i];
      R_image += p_image[i]*p_image[i];
    }
  R = sqrt(R) + epsilon;
  R_image = sqrt(R_image) + epsilon;

  Tensor<3,dim, double> W;
  Tensor<3,dim, double> W2;
  double X3=h_0;
  double x3=p[wall_orientation]+h_0;
  X3+=0.;
  x3+=0.;
  for (unsigned int i=0; i<dim; ++i)
    {
      double delta_i1 = 1.0*(i==wall_orientation);

      if (i==wall_orientation)
        {

          for (unsigned int j=0; j<dim; ++j)
            {
              double delta_ij = 1.0*(i==j);
              for (unsigned int k=0; k<dim; ++k)
                {
                  double delta_jk = 1.0*(k==j);
                  double delta_ik = 1.0*(i==k);

                  W[i][j][k] = -1.*p[i]*p[j]*p[k]/(std::pow(R, dim+2));
                  W[i][j][k] -= -1.*p_image[i]*p_image[j]*p_image[k]/(std::pow(R_image, dim+2));
                  W[i][j][k] -= 2*h_0*h_0*(-(delta_ik*p_image[j]+delta_ij*p_image[k]*delta_jk*p_image[i])/(std::pow(R_image, dim+2))+5.*(p_image[i]*p_image[j]*p_image[k])/(std::pow(R_image, dim+4)));
                  W[i][j][k] -= (-2*h_0)*(p_image[wall_orientation]*(-(delta_ik*p_image[j]+delta_ij*p_image[k]*delta_jk*p_image[i])/(std::pow(R_image, dim+2))+5.*(p_image[i]*p_image[j]*p_image[k])/(std::pow(R_image, dim+4))) +
                                          (delta_jk*p_image[i]*p_image[wall_orientation]-delta_i1*p_image[j]*p_image[k])/(std::pow(R_image, dim+2)));
                  W[i][j][k] *= dim/(2*numbers::PI*(dim-1));
                }
            }
        }
      else
        {
          for (unsigned int j=0; j<dim; ++j)
            {
              double delta_ij = 1.0*(i==j);

              for (unsigned int k=0; k<dim; ++k)
                {
                  double delta_jk = 1.0*(k==j);
                  double delta_ik = 1.0*(i==k);

                  W[i][j][k] = -1.*p[i]*p[j]*p[k]/(std::pow(R, dim+2));
                  W[i][j][k] -= -1.*p_image[i]*p_image[j]*p_image[k]/(std::pow(R_image, dim+2));
                  W[i][j][k] += 2*h_0*h_0*(-(delta_ik*p_image[j]+delta_ij*p_image[k]*delta_jk*p_image[i])/(std::pow(R_image, dim+2))+5.*(p_image[i]*p_image[j]*p_image[k])/(std::pow(R_image, dim+4)));
                  W[i][j][k] += (-2*h_0)*(p_image[wall_orientation]*
                                          (-(delta_ik*p_image[j]+delta_ij*p_image[k]*delta_jk*p_image[i])/(std::pow(R_image, dim+2))+5.*(p_image[i]*p_image[j]*p_image[k])/(std::pow(R_image, dim+4))) +
                                          (delta_jk*p_image[i]*p_image[wall_orientation]-delta_i1*p_image[j]*p_image[k])/(std::pow(R_image, dim+2)));
                  W[i][j][k] *= dim/(2*numbers::PI*(dim-1));
                }
            }
        }
    }
  return W;

}


template <>
Tensor<2, 2, double>
NoSlipWallStokesKernel<2>::value_tens_image3(const Tensor<1,2,double> &, const Tensor<1,2,double> &, const Tensor<1,2,double> &) const
{
  AssertThrow(false, ExcImpossibleInDim(2));
  return Tensor<2, 2, double> ();
}

template <int dim>
Tensor<2, dim, double>
NoSlipWallStokesKernel<dim>::value_tens_image3(const Tensor<1,dim,double> &p, const Tensor<1,dim,double> &p_image, const Tensor<1,dim,double> &normal) const
{

  double R, R_image;
  R = 0.;
  R_image = 0.;
  double h_0 = 0.5*(p_image[wall_orientation]-p[wall_orientation]);
  // std::cout<<h_0<<std::endl;
  StokesKernel<dim> stokes_kernel;
  Tensor<3,dim> W_1=stokes_kernel.value_tens2(p);
  Tensor<3,dim> W_2=stokes_kernel.value_tens2(p_image);
  for (unsigned int i=0; i<dim; ++i)
    {
      R += p[i]*p[i];
      R_image += p_image[i]*p_image[i];
    }
  R = sqrt(R) + epsilon;
  R_image = sqrt(R_image) + epsilon;

  Tensor<2,dim, double> singular_kernel;
  double RmNm=0.;
  double X3=h_0;
  double x3=p[wall_orientation]+h_0;

  for (unsigned  int i=0; i<dim; ++i)
    {
      RmNm += normal[i]*p_image[i];
      for (unsigned  int j=0; j<dim; ++j)
        for (unsigned  int k=0; k<dim; ++k)
          {
            singular_kernel[i][j] += W_1[i][j][k] * normal[k] - W_2[i][j][k] * normal[k];
          }
    }
  for (unsigned  int i=0; i<dim; ++i)
    {
      double delta_i1 = 1.0*(i==wall_orientation);

      for (unsigned  int j=0; j<dim; ++j)
        {
          double delta_j1 = 1.0*(j==wall_orientation);
          double delta_ij = 1.0*(i==j);
          // std::cout<<dim<<" "<<i<<" "<<j<<" "<<wall_orientation<<std::endl;
          singular_kernel[i][j]+=(-delta_j1*p_image[i]*RmNm+p_image[wall_orientation]*normal[i]*(2*delta_j1*p_image[wall_orientation]-p_image[j])
                                  +x3*(delta_ij*RmNm+p_image[i]*normal[j] - 2 * delta_j1 *(delta_i1*RmNm+p_image[i]*normal[wall_orientation])
                                       +(5.*(p_image[i]*RmNm)/(R*R)-normal[i])*(2.*delta_j1*p_image[wall_orientation]-p_image[j])))
                                 // +2./3.*delta_j1*(delta_i1*RmNm+p_image[i]*normal[wall_orientation])
                                 // +(5.*(p_image[i]*RmNm)/(R*R)-normal[i])*(2*delta_j1*p_image[wall_orientation]-p_image[j]))
                                 *.3*X3/2./numbers::PI/std::pow(R_image,dim+2);
        }
    }

  return singular_kernel;

}
template <int dim>
void
NoSlipWallStokesKernel<dim>::set_wall_orientation(const unsigned int orientation_in)
{
  wall_orientation = orientation_in;
}

template <int dim>
unsigned int
NoSlipWallStokesKernel<dim>::get_wall_orientation()
{
  return wall_orientation;
}
template class NoSlipWallStokesKernel<2>;
template class NoSlipWallStokesKernel<3>;


DEAL_II_NAMESPACE_CLOSE
