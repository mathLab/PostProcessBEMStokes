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
