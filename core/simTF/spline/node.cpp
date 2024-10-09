#include "simTF/spline/node.h"

namespace aphys {

Vec4d get_bezier_coeff(const double t) {
  return Vec4d((t - 1.) * (t - 1.) * (1. + 2. * t),  //
               (3. - 2. * t) * t * t,                //
               3. * (t - 1.) * (t - 1.) * t,         //
               3. * (t - 1.) * t * t);
}

Vec4d get_bezier_d_coeff(const double t) {
  return Vec4d(6. * t * (t - 1.),              //
               6. * (1. - t) * t,              //
               3. * (t - 1.) * (3. * t - 1.),  //
               3. * t * (3. * t - 2.));
}

Vec4d get_bezier_dd_coeff(const double t) {
  return Vec4d(6. * (2 * t - 1),    //
               6. * (1. - 2. * t),  //
               6. * (3. * t - 2.),  //
               6. * (3 * t - 1.));
}

}  // namespace aphys
