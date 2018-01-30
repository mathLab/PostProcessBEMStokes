#include "repeated_kernel.h"

DEAL_II_NAMESPACE_OPEN

template <int dim>
RepeatedStokesKernel<dim>::RepeatedStokesKernel(const double eps, const unsigned int rep, const double h_in)
  :
  epsilon(eps),
  repetitions(rep),
  h(h_in)
{}

template <int dim>
Tensor<2, dim, double>
RepeatedStokesKernel<dim>::value_tens(const Tensor<1,dim,double> &p) const
{
  // SacadoDouble R;
  // R = 0.;
  // for(unsigned int i=0; i<dim; ++i)
  //     R += p[i]*p[i];
  // R = sqrt(R) + epsilon;

  Tensor<2,dim, double> G({{
      (2*std::pow(p[0],2) +
      std::pow(h - p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(h - p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(-10*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-10*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(-9*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(-8*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(-7*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(-6*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(-5*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(-4*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(-3*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(-2*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(2*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(3*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(4*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(5*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(6*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(7*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(8*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(9*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (2*std::pow(p[0],2) +
      std::pow(10*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(10*h + p[1],2) +
      std::pow(p[2],2),1.5),
      (p[0]*(-h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(h - p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[1])/
      std::pow(std::pow(p[0],2) +
      std::pow(p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-10*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-10*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-9*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-8*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-7*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-6*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-5*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-4*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-3*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-2*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(2*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(3*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(4*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(5*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(6*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(7*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(8*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(9*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(10*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(10*h + p[1],2) +
      std::pow(p[2],2),1.5),
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(h - p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-10*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(10*h + p[1],2) +
      std::pow(p[2],2),1.5)
    },
    {
      (p[0]*(-h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(h - p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[1])/
      std::pow(std::pow(p[0],2) +
      std::pow(p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-10*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-10*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-9*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-8*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-7*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-6*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-5*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-4*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-3*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(-2*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(-2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(2*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(3*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(4*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(5*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(6*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(7*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(8*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(9*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*(10*h + p[1]))/
      std::pow(std::pow(p[0],2) +
      std::pow(10*h + p[1],2) +
      std::pow(p[2],2),1.5),
      (std::pow(p[0],2) +
      2*std::pow(h - p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(h - p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(-10*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-10*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(-9*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(-8*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(-7*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(-6*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(-5*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(-4*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(-3*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(-2*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(2*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(3*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(4*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(5*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(6*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(7*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(8*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(9*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      2*std::pow(10*h + p[1],2) +
      std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(10*h + p[1],2) +
      std::pow(p[2],2),1.5),
      ((-h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(h - p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[1]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-10*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-10*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-9*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-8*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-7*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-6*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-5*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-4*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-3*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-2*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((2*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((3*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((4*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((5*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((6*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((7*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((8*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((9*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((10*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(10*h + p[1],2) +
      std::pow(p[2],2),1.5)
    },
    {
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(h - p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-10*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[0]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(10*h + p[1],2) +
      std::pow(p[2],2),1.5),
      ((-h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(h - p[1],2) +
      std::pow(p[2],2),1.5) +
      (p[1]*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-10*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-10*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-9*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-8*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-7*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-6*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-5*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-4*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-3*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((-2*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(-2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((2*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((3*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((4*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((5*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((6*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((7*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((8*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((9*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      ((10*h + p[1])*p[2])/
      std::pow(std::pow(p[0],2) +
      std::pow(10*h + p[1],2) +
      std::pow(p[2],2),1.5),
      (std::pow(p[0],2) +
      std::pow(h - p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(h - p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(-10*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-10*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(-9*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(-8*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(-7*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(-6*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(-5*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(-4*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(-3*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(-2*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(-2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(2*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(2*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(3*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(3*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(4*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(4*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(5*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(5*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(6*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(6*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(7*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(7*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(8*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(8*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(9*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(9*h + p[1],2) +
      std::pow(p[2],2),1.5) +
      (std::pow(p[0],2) +
      std::pow(10*h + p[1],2) +
      2*std::pow(p[2],2))/
      std::pow(std::pow(p[0],2) +
      std::pow(10*h + p[1],2) +
      std::pow(p[2],2),1.5)
    }
  });

  return G;
}

template <int dim>
Tensor<3, dim, double>
RepeatedStokesKernel<dim>::value_tens2(const Tensor<1,dim,double> &p) const
{
  Tensor<3,dim, double> W({{{
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        std::pow(p[0],2)*(-h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[1]*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-10*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-9*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-8*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-7*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-6*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-5*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-4*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-3*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-2*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(2*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(3*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(4*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(5*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(6*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(7*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(8*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(9*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(10*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.)
      },
      {
        std::pow(p[0],2)*(-h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[1]*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-10*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-9*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-8*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-7*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-6*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-5*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-4*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-3*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-2*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(2*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(3*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(4*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(5*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(6*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(7*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(8*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(9*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(10*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        p[0]*std::pow(h - p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-10*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-9*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-8*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-7*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-6*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-5*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-4*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-3*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-2*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(2*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(3*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(4*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(5*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(6*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(7*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(8*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(9*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(10*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        p[0]*(-h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*p[1]*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-10*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-9*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-8*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-7*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-6*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-5*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-4*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-3*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-2*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(2*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(3*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(4*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(5*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(6*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(7*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(8*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(9*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(10*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.)
      },
      {
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        p[0]*(-h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*p[1]*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-10*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-9*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-8*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-7*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-6*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-5*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-4*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-3*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-2*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(2*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(3*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(4*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(5*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(6*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(7*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(8*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(9*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(10*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.)
      }
    },
    { {
        std::pow(p[0],2)*
        (-h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[1]*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-10*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-9*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-8*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-7*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-6*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-5*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-4*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-3*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(-2*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(2*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(3*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(4*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(5*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(6*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(7*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(8*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(9*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*(10*h + p[1])*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        p[0]*std::pow(h - p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-10*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-9*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-8*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-7*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-6*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-5*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-4*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-3*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-2*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(2*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(3*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(4*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(5*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(6*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(7*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(8*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(9*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(10*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        p[0]*(-h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*p[1]*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-10*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-9*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-8*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-7*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-6*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-5*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-4*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-3*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-2*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(2*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(3*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(4*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(5*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(6*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(7*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(8*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(9*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(10*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.)
      },
      {
        p[0]*std::pow(h - p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-10*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-9*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-8*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-7*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-6*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-5*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-4*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-3*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(-2*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(2*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(3*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(4*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(5*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(6*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(7*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(8*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(9*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(10*h + p[1],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        std::pow(-h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-10*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-9*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-8*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-7*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-6*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-5*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-4*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-3*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-2*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(2*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(3*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(4*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(5*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(6*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(7*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(8*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(9*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(10*h + p[1],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        std::pow(h - p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-10*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-9*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-8*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-7*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-6*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-5*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-4*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-3*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-2*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(2*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(3*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(4*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(5*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(6*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(7*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(8*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(9*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(10*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.)
      },
      {
        p[0]*(-h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*p[1]*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-10*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-9*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-8*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-7*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-6*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-5*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-4*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-3*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-2*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(2*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(3*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(4*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(5*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(6*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(7*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(8*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(9*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(10*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        std::pow(h - p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-10*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-9*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-8*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-7*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-6*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-5*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-4*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-3*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-2*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(2*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(3*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(4*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(5*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(6*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(7*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(8*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(9*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(10*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        (-h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[1]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-10*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-9*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-8*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-7*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-6*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-5*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-4*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-3*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-2*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (2*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (3*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (4*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (5*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (6*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (7*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (8*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (9*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (10*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.)
      }
    },
    { {
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[0],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        p[0]*(-h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*p[1]*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-10*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-9*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-8*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-7*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-6*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-5*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-4*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-3*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-2*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(2*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(3*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(4*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(5*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(6*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(7*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(8*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(9*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(10*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.)
      },
      {
        p[0]*(-h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*p[1]*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-10*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-9*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-8*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-7*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-6*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-5*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-4*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-3*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(-2*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(2*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(3*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(4*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(5*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(6*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(7*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(8*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(9*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*(10*h + p[1])*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        std::pow(h - p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-10*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-9*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-8*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-7*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-6*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-5*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-4*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-3*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(-2*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(2*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(3*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(4*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(5*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(6*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(7*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(8*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(9*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(10*h + p[1],2)*p[2]*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        (-h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[1]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-10*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-9*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-8*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-7*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-6*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-5*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-4*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-3*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-2*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (2*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (3*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (4*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (5*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (6*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (7*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (8*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (9*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (10*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.)
      },
      {
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[0]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        (-h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        p[1]*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-10*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-9*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-8*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-7*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-6*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-5*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-4*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-3*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (-2*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (2*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (3*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (4*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (5*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (6*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (7*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (8*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (9*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        (10*h + p[1])*std::pow(p[2],2)*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.),
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(h - p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(-2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(2*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(3*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(4*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(5*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(6*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(7*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(8*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(9*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.) +
        std::pow(p[2],3)*
        std::pow(std::pow(p[0],2) +
        std::pow(10*h + p[1],2) +
        std::pow(p[2],2),
        -1 - dim/2.)
      }
    }
  });
  W *= -dim/(2*(dim-1)*numbers::PI);
  return W;
}

template class RepeatedStokesKernel<3>;

DEAL_II_NAMESPACE_CLOSE
