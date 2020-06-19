#include "distance_base.h"

struct PlainDist1D {
    static inline const double side_distance_from_min_max(
        const ckdtree * tree, const double x,
        const double min,
        const double max,
        const ckdtree_intp_t k
        )
    {
        double s, t;
        s = 0;
        t = x - max;
        if (t > s) {
            s = t;
        } else {
            t = min - x;
            if (t > s) s = t;
        }
        return s;
    }
    static inline void
    interval_interval(const ckdtree * tree,
                        const Rectangle& rect1, const Rectangle& rect2,
                        const ckdtree_intp_t k,
                        double *min, double *max)
    {
        /* Compute the minimum/maximum distance along dimension k between points in
         * two hyperrectangles.
         */
        *min = std::fmax(0., std::fmax(rect1.mins()[k] - rect2.maxes()[k],
                              rect2.mins()[k] - rect1.maxes()[k]));
        *max = std::fmax(rect1.maxes()[k] - rect2.mins()[k],
                              rect2.maxes()[k] - rect1.mins()[k]);
    }

    static inline double
    point_point(const ckdtree * tree,
               const double *x, const double *y,
                 const ckdtree_intp_t k) {
        return std::fabs(x[k] - y[k]);
    }
};

typedef BaseMinkowskiDistPp<PlainDist1D> MinkowskiDistPp;
typedef BaseMinkowskiDistPinf<PlainDist1D> MinkowskiDistPinf;
typedef BaseMinkowskiDistP1<PlainDist1D> MinkowskiDistP1;
typedef BaseMinkowskiDistP2<PlainDist1D> NonOptimizedMinkowskiDistP2;

/*
 * Measuring distances
 * ===================
 */


#ifndef CKDTREE_BLAS_DIST
#define CKDTREE_BLAS_DIST

double
sqeuclidean_distance_double_blas(const double *u, const double *v, ckdtree_intp_t n);

#endif


#ifdef USE_GNU_EXT

inline static double
sqeuclidean_distance_double(const double *u, const double *v, ckdtree_intp_t n)
{
    // Faster than MKL daxpy+ddot up to about 16 dimensional space.
    // About 3x faster for n < 8.

    typedef double vec4d __attribute__ ((vector_size (4*sizeof(double))));
    typedef double vec2d __attribute__ ((vector_size (2*sizeof(double))));     

    double s = 0.0;

    // manually unrolled loop using GNU vector extensions

    const ckdtree_uintp_t un = static_cast<const ckdtree_uintp_t>(n);

    switch(un) {

        case 0:
        break; // help the compiler make a fast jump tab

        case 1:
        {
            double d = u[0] - v[0];
            s = d*d; 
        }
        break;

        case 2:
        {
            vec2d _u = {u[0], u[1]};
            vec2d _v = {v[0], v[1]};
            vec2d diff = _u - _v;
            vec2d acc = diff * diff;
            s = acc[0] + acc[1];
        }
        break;

        case 3:
        {
            vec4d _u = {u[0], u[1], u[2], 0.0};
            vec4d _v = {v[0], v[1], v[2], 0.0};
            vec4d diff = _u - _v;
            vec4d acc = diff * diff;
            s = acc[0] + acc[1] + acc[2] + acc[3];
        }
        break;

        case 4:
        {
            vec4d _u = {u[0], u[1], u[2], u[3]};
            vec4d _v = {v[0], v[1], v[2], v[3]};
            vec4d diff = _u - _v;
            vec4d acc = diff * diff;
            s = acc[0] + acc[1] + acc[2] + acc[3];
        }
        break;

        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        {
            ckdtree_intp_t i;
            vec4d acc = {0., 0., 0., 0.};
            for (i = 0; i < n/4; i += 4) {
                vec4d _u = {u[i], u[i + 1], u[i + 2], u[i + 3]};
                vec4d _v = {v[i], v[i + 1], v[i + 2], v[i + 3]};
                vec4d diff = _u - _v;
                acc += diff * diff;
            }
            s = acc[0] + acc[1] + acc[2] + acc[3];
            if (i < n) {
                for(; i<n; ++i) {
                    double d = u[i] - v[i];
                    s = std::fma(d, d, s);
                }
            }
        }
        break;

        default:
        s = sqeuclidean_distance_double_blas(u, v, n);

    }
    return s;
}

#else // use std::fma on compilers that do not have GNU simd

inline static double
sqeuclidean_distance_double(const double *u, const double *v, ckdtree_intp_t n)
{
    // Faster than MKL daxpy+ddot up to about 12 dimensional space.
    // About 2x faster for n < 6.

    double s = 0.0;
    // manually unrolled loop using GNU vector extensions

    const ckdtree_uintp_t un = static_cast<const ckdtree_uintp_t>(n);

    switch(un) {

        case 0:
        break; // help the compiler make a fast jump tab

        case 1:
        {
            double d = u[0] - v[0];
            s = d*d; 
        }
        break;

        case 2:
        {
            double _u[2] = {u[0], u[1]};
            double _v[2] = {v[0], v[1]};
            double diff[2] = {_u[0] - _v[0], 
                              _u[1] - _v[1]};
            s = diff[0] * diff[0];
            s = std::fma(diff[1], diff[1], s);
        }
        break;

        case 3:
        {
            double _u[3] = {u[0], u[1], u[2]};
            double _v[3] = {v[0], v[1], v[2]};
            double diff[3] = {_u[0] - _v[0],
                              _u[1] - _v[1],
                              _u[2] - _v[2]};
            s = diff[0] * diff[0];
            s = std::fma(diff[1], diff[1], s);
            s = std::fma(diff[2], diff[2], s);
        }
        break;

        case 4:
        {
            double _u[4] = {u[0], u[1], u[2], u[3]};
            double _v[4] = {v[0], v[1], v[2], v[3]};
            double diff[4] = {_u[0] - _v[0],
                              _u[1] - _v[1],
                              _u[2] - _v[2],
                              _u[3] - _v[3]};
            s = diff[0] * diff[0];
            s = std::fma(diff[1], diff[1], s);
            s = std::fma(diff[2], diff[2], s);
            s = std::fma(diff[3], diff[3], s);
        }
        break;

        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        {
            ckdtree_intp_t i;
            for (i = 0; i < n/4; i += 4) {
                double _u[4] = {u[i], u[i + 1], u[i + 2], u[i + 3]};
                double _v[4] = {v[i], v[i + 1], v[i + 2], v[i + 3]};
                double diff[4] = {_u[0] - _v[0],
                                  _u[1] - _v[1],
                                  _u[2] - _v[2],
                                  _u[3] - _v[3]};
                s = std::fma(diff[0], diff[0], s);
                s = std::fma(diff[1], diff[1], s);
                s = std::fma(diff[2], diff[2], s);
                s = std::fma(diff[3], diff[3], s);
            }
            if (i < n) {
                for(; i<n; ++i) {
                    double d = u[i] - v[i];
                    s = std::fma(d, d, s);
                }
            }
        }
        break;

        default:
        s = sqeuclidean_distance_double_blas(u, v, n);
    }   

    return s;
}

#endif


struct MinkowskiDistP2: NonOptimizedMinkowskiDistP2 {
    static inline double
    point_point_p(const ckdtree * tree,
               const double *x, const double *y,
               const double p, const ckdtree_intp_t k,
               const double upperbound)
    {
        return sqeuclidean_distance_double(x, y, k);
    }
};

struct BoxDist1D {
    static inline void _interval_interval_1d (
        double min, double max,
        double *realmin, double *realmax,
        const double full, const double half
    )
    {
        /* Minimum and maximum distance of two intervals in a periodic box
         *
         * min and max is the nonperiodic distance between the near
         * and far edges.
         *
         * full and half are the box size and 0.5 * box size.
         *
         * value is returned in realmin and realmax;
         *
         * This function is copied from kdcount, and the convention
         * of is that
         *
         * min = rect1.min - rect2.max
         * max = rect1.max - rect2.min = - (rect2.min - rect1.max)
         *
         * We will fix the convention later.
         * */
        if (CKDTREE_UNLIKELY(full <= 0)) {
            /* A non-periodic dimension */
            /* \/     */
            if(max <= 0 || min >= 0) {
                /* do not pass though 0 */
                min = std::fabs(min);
                max = std::fabs(max);
                if(min < max) {
                    *realmin = min;
                    *realmax = max;
                } else {
                    *realmin = max;
                    *realmax = min;
                }
            } else {
                min = std::fabs(min);
                max = std::fabs(max);
                *realmax = std::fmax(max, min);
                *realmin = 0;
            }
            /* done with non-periodic dimension */
            return;
        }
        if(max <= 0 || min >= 0) {
            /* do not pass through 0 */
            min = std::fabs(min);
            max = std::fabs(max);
            if(min > max) {
                double t = min;
                min = max;
                max = t;
            }
            if(max < half) {
                /* all below half*/
                *realmin = min;
                *realmax = max;
            } else if(min > half) {
                /* all above half */
                *realmax = full - min;
                *realmin = full - max;
            } else {
                /* min below, max above */
                *realmax = half;
                *realmin = std::fmin(min, full - max);
            }
        } else {
            /* pass though 0 */
            min = -min;
            if(min > max) max = min;
            if(max > half) max = half;
            *realmax = max;
            *realmin = 0;
        }
    }
    static inline void
    interval_interval(const ckdtree * tree,
                        const Rectangle& rect1, const Rectangle& rect2,
                        const ckdtree_intp_t k,
                        double *min, double *max)
    {
        /* Compute the minimum/maximum distance along dimension k between points in
         * two hyperrectangles.
         */
        _interval_interval_1d(rect1.mins()[k] - rect2.maxes()[k],
                    rect1.maxes()[k] - rect2.mins()[k], min, max,
                    tree->raw_boxsize_data[k], tree->raw_boxsize_data[k + rect1.m]);
    }

    static inline double
    point_point(const ckdtree * tree,
               const double *x, const double *y,
               const ckdtree_intp_t k)
    {
        double r1;
        r1 = wrap_distance(x[k] - y[k], tree->raw_boxsize_data[k + tree->m], tree->raw_boxsize_data[k]);
        r1 = std::fabs(r1);
        return r1;
    }

    static inline const double
    wrap_position(const double x, const double boxsize)
    {
        if (boxsize <= 0) return x;
        const double r = std::floor(x / boxsize);
        double x1 = x - r * boxsize;
        /* ensure result is within the box. */
        while(x1 >= boxsize) x1 -= boxsize;
        while(x1 < 0) x1 += boxsize;
        return x1;
    }

    static inline const double side_distance_from_min_max(
        const ckdtree * tree, const double x,
        const double min,
        const double max,
        const ckdtree_intp_t k
        )
    {
        double s, t, tmin, tmax;
        double fb = tree->raw_boxsize_data[k];
        double hb = tree->raw_boxsize_data[k + tree->m];

        if (fb <= 0) {
            /* non-periodic dimension */
            s = PlainDist1D::side_distance_from_min_max(tree, x, min, max, k);
            return s;
        }

        /* periodic */
        s = 0;
        tmax = x - max;
        tmin = x - min;
        /* is the test point in this range */
        if(CKDTREE_LIKELY(tmax < 0 && tmin > 0)) {
            /* yes. min distance is 0 */
            return 0;
        }

        /* no */
        tmax = std::fabs(tmax);
        tmin = std::fabs(tmin);

        /* make tmin the closer edge */
        if(tmin > tmax) { t = tmin; tmin = tmax; tmax = t; }

        /* both edges are less than half a box. */
        /* no wrapping, use the closer edge */
        if(tmax < hb) return tmin;

        /* both edge are more than half a box. */
        /* wrapping on both edge, use the
         * wrapped further edge */
        if(tmin > hb) return fb - tmax;

        /* the further side is wrapped */
        tmax = fb - tmax;
        if(tmin > tmax) return tmax;
        return tmin;
    }

    private:
    static inline double
    wrap_distance(const double x, const double hb, const double fb)
    {
        double x1;
        if (CKDTREE_UNLIKELY(x < -hb)) x1 = fb + x;
        else if (CKDTREE_UNLIKELY(x > hb)) x1 = x - fb;
        else x1 = x;
    #if 0
        printf("ckdtree_fabs_b x : %g x1 %g\n", x, x1);
    #endif
        return x1;
    }


};


typedef BaseMinkowskiDistPp<BoxDist1D> BoxMinkowskiDistPp;
typedef BaseMinkowskiDistPinf<BoxDist1D> BoxMinkowskiDistPinf;
typedef BaseMinkowskiDistP1<BoxDist1D> BoxMinkowskiDistP1;
typedef BaseMinkowskiDistP2<BoxDist1D> BoxMinkowskiDistP2;

