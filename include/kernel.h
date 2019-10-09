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

#ifndef __mathlab__kernel_h // Avoid double definitions
#define __mathlab__kernel_h

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <Sacado.hpp>
#include <Sacado_cmath.hpp>
//#include <>
typedef Sacado::Fad::DFad<double> SacadoDouble;
typedef Sacado::Fad::DFad<SacadoDouble> SacadoSacadoDouble;

DEAL_II_NAMESPACE_OPEN

template <int rank, int dim>
class SacadoKernel : public  TensorFunction<rank, dim, double>
{
public:
  virtual Tensor<rank,dim,double>
  value_tens (const Tensor<1, dim, double > &p) const;

  virtual Tensor<rank+1,dim,double>
  value_tens2 (const Tensor<1, dim, double > &p) const;

//  virtual Tensor<rank+2,dim,double>
// value_tens3 (const Tensor<1, dim, double > &p) const;

  virtual Tensor<rank+1,dim,double>
  gradient_tens (const Tensor<1, dim, double > &p) const;

  typedef Sacado::Fad::DFad<double> SacadoDouble;
  typedef Sacado::Fad::DFad<SacadoDouble> SacadoSacadoDouble;

private:
  virtual Tensor<rank, dim, SacadoDouble>
  sacado_value(const Tensor<1, dim, SacadoDouble> &p) const = 0;
  virtual Tensor<rank+1, dim, SacadoDouble>
  sacado_value2(const Tensor<1, dim, SacadoDouble> &p) const = 0;
  // virtual Tensor<rank+2, dim, SacadoDouble>
  // sacado_value3(const Tensor<1, dim, SacadoDouble> &p) const;

};


template <int dim>
class StokesKernel : public SacadoKernel<2,dim>
{
public:
  StokesKernel(const double eps = 0.);

  virtual Tensor<2, dim, SacadoDouble>
  sacado_value(const Tensor<1,dim,SacadoDouble> &p) const;
  virtual Tensor<3, dim, SacadoDouble>
  sacado_value2(const Tensor<1,dim,SacadoDouble> &p) const;

  virtual Tensor<2, dim, double>
  value_tens(const Tensor<1,dim,double> &p) const;
  virtual Tensor<3, dim, double>
  value_tens2(const Tensor<1,dim,double> &p) const;
  virtual Tensor<4, dim, double>
  value_tens3(const Tensor<1,dim,double> &p) const;

private:
  const double epsilon;
};



DEAL_II_NAMESPACE_CLOSE

// using namespace dealii;
//
// template <int dim, class Type=double>
// class Kernel: public Function<dim>
// {
// public:
//   Kernel(const unsigned int n_components=1) :
//     Function<dim>(n_components) {}
//
//   typedef Sacado::Fad::DFad<double> sacado_double;
//   typedef Sacado::Fad::DFad<sacado_double> sacado_sacado_double;
//
//   virtual Type t_value (const Tensor<1, dim, Type>   &p,
//                         const unsigned int  dimension = 0) const;
//
//   virtual double value_tens (const Tensor<1, dim>   &p,
//                              const unsigned int  dimension = 0) const;
//
//   virtual Tensor<1,dim> gradient_tens(const Tensor<1, dim>  &p,
//                                       const unsigned int   component = 0) const;
//
// private:
//   virtual
//   Type convert_to_type(double d) const;
//
//   virtual
//   double convert_to_double(Type d) const;
//
//   // virtual double Hessian (const Tensor<1, dim>   &p,
//
//   //const unsigned int n_components;
// };
//
#endif
