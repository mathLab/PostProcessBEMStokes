#ifndef __mathlab__repeated_kernel_h // Avoid double definitions
#define __mathlab__repeated_kernel_h


#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
#include <cmath>
#include <kernel.h>

DEAL_II_NAMESPACE_OPEN

template <int dim>
class RepeatedStokesKernel : public StokesKernel<dim>
{
public:
  RepeatedStokesKernel(const double eps = 0., const unsigned int rep=10, const double h_in=10.);

  virtual Tensor<2, dim, double>
  value_tens(const Tensor<1,dim,double> &p) const;
  virtual Tensor<3, dim, double>
  value_tens2(const Tensor<1,dim,double> &p) const;
private:
  const double epsilon;
  const unsigned int repetitions;
  const double h;
};

DEAL_II_NAMESPACE_CLOSE

#endif
