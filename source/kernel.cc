#include "../include/kernel.h"
#include <cmath>
#include <Sacado.hpp>
#include <typeinfo>


DEAL_II_NAMESPACE_OPEN
typedef Sacado::Fad::DFad<double> SacadoDouble;
typedef Sacado::Fad::DFad<SacadoDouble> SacadoSacadoDouble;
namespace
{
  void interpret(double &out, const Sacado::Fad::DFad<double> &in)
  {
    out = in.val();
  }

  void interpret(Sacado::Fad::DFad<double> &out, const double &in)
  {
    out = in;
  }

  template <int dim, typename outtype, typename intype>
  void interpret(Tensor<0, dim, outtype> &out, const Tensor<0, dim, intype> &in)
  {
    interpret(static_cast<outtype &>(out), static_cast<const intype &>(in));
  }

  template <int rank, int dim, typename outtype, typename intype>
  void interpret(Tensor<rank, dim, outtype> &out, const Tensor<rank, dim, intype> &in)
  {
    for (unsigned int i=0; i<dim; ++i)
      interpret(out[i], in[i]);
  }
}


template <int rank, int dim>
Tensor<rank,dim,double>
SacadoKernel<rank, dim>::value_tens(const Tensor<1, dim, double > &p) const
{
  Tensor<1, dim, SacadoDouble> sac_p;
  interpret(sac_p, p);
  Tensor<rank, dim, SacadoDouble> sac_res = sacado_value(sac_p);
  Tensor<rank, dim, double> res;
  interpret(res, sac_res);
  return res;
}

template <int rank, int dim>
Tensor<rank+1,dim,double>
SacadoKernel<rank, dim>::value_tens2(const Tensor<1, dim, double > &p) const
{
  Tensor<1, dim, SacadoDouble> sac_p;
  interpret(sac_p, p);
  Tensor<rank+1, dim, SacadoDouble> sac_res = sacado_value2(sac_p);
  Tensor<rank+1, dim, double> res;
  interpret(res, sac_res);
  return res;
}

template <int dim>
Tensor<2,dim,double>
StokesKernel<dim>::value_tens(const Tensor<1, dim, double > &p) const
{
  double R = 0.;
  for (unsigned int i=0; i<dim; ++i)
    R += p[i]*p[i];
  R = sqrt(R) + epsilon;

  Tensor<2,dim, double> G;
  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      {
        double delta = 1.0*(i==j);
        if (dim ==2)
          G[i][j] = p[i]*p[j]/(R*R)-delta*std::log(R);
        else
          G[i][j] = p[i]*p[j]/(R*R*R)+delta/R;
        G[i][j] /= (4*numbers::PI*(dim-1));
      }
  return G;

}

template <int dim>
Tensor<3,dim,double>
StokesKernel<dim>::value_tens2(const Tensor<1, dim, double > &p) const
{
  double R = 0.;
  for (unsigned int i=0; i<dim; ++i)
    R += p[i]*p[i];
  R = sqrt(R) + epsilon;

  Tensor<3,dim, double> W;
  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      {
        for (unsigned int k=0; k<dim; ++k)
          {
            W[i][j][k] = -dim*p[i]*p[j]*p[k]/(std::pow(R, dim+2))/(2*numbers::PI*(dim-1));
          }
      }
  return W;
}


template <int rank, int dim>
Tensor<rank+1,dim,double>
SacadoKernel<rank, dim>::gradient_tens(const Tensor<1, dim, double > &p) const
{
  Tensor<1, dim, SacadoDouble> sac_p;
  for (unsigned int i=0; i<dim; ++i)
    {
      sac_p[i] = p[i];
      sac_p[i].diff(i, dim);
    }
  Tensor<rank, dim, SacadoDouble> sac_res = sacado_value(sac_p);
  Tensor<rank+1, dim, double> res;
  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      for (unsigned int k=0; k<dim; ++k)
        res[i][j][k] = sac_res[i][j].fastAccessDx(k);
  return res;
}


template <int dim>
StokesKernel<dim>::StokesKernel(const double eps)
  :
  epsilon(eps)
{}

/// The implementation of the free space Green function.
// We have checked it many times
// and it seems OK so far.

template <int dim>
Tensor<2, dim, SacadoDouble>
StokesKernel<dim>::sacado_value(const Tensor<1,dim,SacadoDouble> &p) const
{
  SacadoDouble R;
  R = 0.;
  for (unsigned int i=0; i<dim; ++i)
    R += p[i]*p[i];
  R = sqrt(R) + epsilon;

  Tensor<2,dim, SacadoDouble> G;
  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      {
        double delta = 1.0*(i==j);
        if (dim ==2)
          G[i][j] = p[i]*p[j]/(R*R)-delta*std::log(R);
        else
          G[i][j] = p[i]*p[j]/(R*R*R)+delta/R;
        G[i][j] /= (4*numbers::PI*(dim-1));
      }
  return G;
}

/// The implementation of the three indices tensor.
// We have checked it many times
// and it seems OK so far.
template <int dim>
Tensor<3, dim, SacadoDouble>
StokesKernel<dim>::sacado_value2(const Tensor<1,dim,SacadoDouble> &p) const
{
  SacadoDouble R;
  R = 0.;
  for (unsigned int i=0; i<dim; ++i)
    R += p[i]*p[i];
  R = sqrt(R) + epsilon;

  Tensor<3,dim, SacadoDouble> W;
  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      {
        for (unsigned int k=0; k<dim; ++k)
          {
            W[i][j][k] = -dim*p[i]*p[j]*p[k]/(std::pow(R, dim+2))/(2*numbers::PI*(dim-1));
          }
      }
  return W;
}

/// The implementation of the three indices tensor.
// We have checked it many times
// and it seems OK so far.
template <int dim>
Tensor<4, dim, double>
StokesKernel<dim>::value_tens3(const Tensor<1,dim,double> &p) const
{
  double R;
  R = 0.;
  for (unsigned int i=0; i<dim; ++i)
    R += p[i]*p[i];
  R = sqrt(R) + epsilon;

  Tensor<4,dim, double> L;
  for (unsigned int i=0; i<dim; ++i)
    {

      for (unsigned int j=0; j<dim; ++j)
        {
          double delta_ij = 1.0*(i==j);

          for (unsigned int k=0; k<dim; ++k)
            {
              double delta_jk = 1.0*(j==k);
              double delta_ik = 1.0*(i==k);

              for (unsigned int m=0; m<dim; ++m)
                {
                  double delta_im = 1.0*(i==m);
                  double delta_jm = 1.0*(j==m);
                  double delta_mk = 1.0*(k==m);

                  L[i][j][k][m] = -4. * delta_im * delta_jk / std::pow(R,3) +
                                  -6. * p[k] * (delta_jm * p[i] + delta_ij * p[m]) / std::pow(R,5) +
                                  -6. * p[j] * (delta_mk * p[i] + delta_ik * p[m]) / std::pow(R,5) +
                                  +60. * p[i] * p[j] * p[k] * p[m] / std::pow(R,7);
                  L[i][j][k][m]/= (-4*numbers::PI*(dim-1));
                }
            }
        }
    }
  return L;
}


template class SacadoKernel<2,2>;
template class SacadoKernel<2,3>;
template class StokesKernel<2>;
template class StokesKernel<3>;


DEAL_II_NAMESPACE_CLOSE

// template <int dim, class Type>
// double Kernel<dim,Type>::value_tens (const Tensor<1,dim> &R,
//                                      const unsigned int component) const
// {
//   Tensor<1, dim, Type> TypeR;
//   for (unsigned int i=0; i<dim; ++i)
//     TypeR[i] = convert_to_type(R[i]);
//   return (convert_to_double(t_value(TypeR,component)));
// }
//
//
//
// template <int dim, class Type>
// Type Kernel<dim,Type>::t_value (const Tensor<1,dim, Type> &R,
//                                 const unsigned int component) const
// {
//   Assert(false, ExcPureFunctionCalled());
//   return Type();
// }
//
//
// template <int dim, class Type>
// Type Kernel<dim,Type>::convert_to_type(double d) const
// {
//   if (typeid(Type) == typeid(sacado_double) )
//     {
//       sacado_double sd(d);
//       return *reinterpret_cast<Type *>(&sd);
//     }
//   else
//     {
//       return static_cast<Type>(d);
//     }
// }
//
// template <int dim, class Type>
// double Kernel<dim,Type>::convert_to_double(Type d) const
// {
//   if (typeid(Type) == typeid(sacado_double) )
//     {
//       sacado_double *sd = reinterpret_cast<sacado_double *>(&d);
//       return sd->val();
//     }
//   else
//     {
//       Assert(false, ExcNotImplemented());
//       //return static_cast<double>(d);
//       return 0.;
//     }
// }
//
//
// template <int dim, class Type>
// Tensor<1,dim> Kernel<dim,Type>::gradient_tens(const Tensor<1,dim> &R, const unsigned int component) const
// {
//   //if Type is sacado_double we convert Tensor<1, dim, Type> into a Point<dim, sacado_double>
//   if (typeid(sacado_double) == typeid(Type))
//     {
//       Tensor<1,dim, sacado_double> Rsac;
//       for (unsigned int i=0; i<dim; ++i)
//         {
//           Rsac[i] = R[i];
//           Rsac[i].diff(i, dim);
//         }
//
//       const Tensor<1, dim, Type> *TypeR =
//         reinterpret_cast<const Tensor<1, dim, Type>*>(&Rsac);
//
//       Type v=t_value(*TypeR, component);
//
//       Tensor<1,dim> grad;
//       for (unsigned int i=0; i<dim; ++i)
//         grad[i] = v.fastAccessDx(i);
//       return grad;
//     }
//   else
//     {
//       Assert(false, ExcNotImplemented());
//       return Tensor<1,dim>();
//     }
// }
//
// /*
// template <int dim>
// void kernels(const Tensor<1, dim, Type> &R, Tensor<1, dim, Type> &D, double &d)
// {
//     double r = R.norm();
//     double p[2] = r*r;
//     switch(dim)
//     {
//         case 2:
//             d = -std::log(r) / (2*numbers::PI);
//             D = R / ( -2*numbers::PI * p[2]);
//             break;
//         case 3:
//             d = (1./( r*4*numbers::PI ) );
//             D = R / ( -4*numbers::PI * p[2]*r );
//             break;
//         default:
//             Assert(false, ExcInternalError());
//     }
// }
// */
// typedef Sacado::Fad::DFad<double> sacado_double;
// typedef Sacado::Fad::DFad<sacado_double> sacado_sacado_double;
// // template class Kernel<2>;
// // template class Kernel<3>;
// template class Kernel<2, sacado_double>;
// template class Kernel<3, sacado_double>;
