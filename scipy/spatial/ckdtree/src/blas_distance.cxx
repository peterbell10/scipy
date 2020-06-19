
#include <vector>
#include <algorithm>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "numpy/arrayobject.h"
#include "npy_cblas.h"

// blas
extern "C"
{
    void BLAS_FUNC(daxpy)(CBLAS_INT *n, double *alpha, double *x, CBLAS_INT *incx, double *y, CBLAS_INT *incy);
    double BLAS_FUNC(ddot)(CBLAS_INT *n, double *x, CBLAS_INT *incx, double *y, CBLAS_INT *incy);
}

#define CKDTREE_BLAS_DIST

#include "ckdtree_decl.h"

double
sqeuclidean_distance_double_blas(const double *u, const double *v, ckdtree_intp_t n)
{
    double *du = const_cast<double*>(u);
    double *dv = const_cast<double*>(v);
    CBLAS_INT one = 1;
    CBLAS_INT bn = static_cast<CBLAS_INT>(n); // this is safe because n is the number of dimensions
    double minus_one = -1;
    if (CKDTREE_LIKELY(n <= 64)) {
        double tmp[64];
        std::copy(dv, dv+n, &tmp[0]);
        BLAS_FUNC(daxpy)(&bn, &minus_one, du, &one, &tmp[0], &one);
        return BLAS_FUNC(ddot)(&bn, &tmp[0], &one, &tmp[0], &one);
    } else {
        std::vector<double> tmp(n);
        std::copy(dv, dv+n, &tmp[0]);
        BLAS_FUNC(daxpy)(&bn, &minus_one, du, &one, &tmp[0], &one);
        return BLAS_FUNC(ddot)(&bn, &tmp[0], &one, &tmp[0], &one);
    }
}


