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

#ifndef __mathlab__no_slip_wall_kernel_h // Avoid double definitions
#define __mathlab__no_slip_wall_kernel_h

#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
#include <cmath>
#include <kernel.h>

DEAL_II_NAMESPACE_OPEN

template <int dim>
class NoSlipWallStokesKernel : public StokesKernel<dim>
{
public:
  NoSlipWallStokesKernel(const double eps = 0.);

  virtual Tensor<2, dim, double>
  value_tens_image(const Tensor<1,dim,double> &p, const Tensor<1,dim,double> &p_image) const;
  virtual Tensor<3, dim, double>
  value_tens_image2(const Tensor<1,dim,double> &p, const Tensor<1,dim,double> &p_image) const;

  Tensor<2,dim>value_tens_image3(const Tensor<1,dim,double> &p, const Tensor<1,dim,double> &p_image,  const Tensor<1,dim,double> &normal) const;

  void set_wall_orientation(const unsigned int orientation);

  unsigned int  get_wall_orientation();

private:
  const double epsilon;
  double wall_position;
  unsigned int wall_orientation;
};

DEAL_II_NAMESPACE_CLOSE

#endif
