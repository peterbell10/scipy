#include "distance_base.h"
#include "vec2d.h"

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

        *max =  std::fmax(rect1.maxes()[k] - rect2.mins()[k],
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


inline double
sqeuclidean_distance_double(const double *u, const double *v, ckdtree_intp_t n)
{
    const ckdtree_uintp_t un = static_cast<const ckdtree_uintp_t>(n);
    using vec2d = ckdtree_vec2d;

    // Computes (a - b) ** 2 in a vec2d
    auto d2 = [](const double *a, const double *b) {
             auto diff = vec2d::loadu(a) - vec2d::loadu(b);
             return diff * diff;
         };
    auto d2_scalar = [](const double *a, const double *b) {
             auto diff = a[0] - b[0];
             return vec2d::from_scalar(diff * diff);
         };

    switch(un) {
        case 0: return 0.;
        case 1:
        {
            auto diff = u[0] - v[0];
            return diff * diff;
        }
        case 2:
        {
            return d2(u, v).sum();
        }
        case 3:
        {
            auto acc = d2(u, v) + d2_scalar(&u[2], &v[2]);
            return acc.sum();
        }
        case 4:
        {
            auto acc = d2(&u[0], &v[0]) + d2(&u[2], &v[2]);
            return acc.sum();
        }
        case 5:
        {
            auto acc = d2(&u[0], &v[0]) + d2(&u[2], &v[2]) + d2_scalar(&u[4], &v[4]);
            return acc.sum();
        }
        case 6:
        {
            auto acc = d2(&u[0], &v[0]) + d2(&u[2], &v[2]) + d2(&u[4], &v[4]);
            return acc.sum();
        }
    }

    constexpr int ilp_factor = 2;
    vec2d acc[ilp_factor] = { d2(&u[0], &v[0]), d2(&u[2], &v[2]) };
    ckdtree_intp_t i;
    for (i = 2 * ilp_factor; i + 2 * ilp_factor <= n; i += 2 * ilp_factor) {
        #pragma unroll
        for (int k = 0; k < ilp_factor; ++k) {
            acc[k] = acc[k] + d2(&u[i + 2 * k], &v[i + 2 * k]);
        }
    }

    for (; i + 2 <= n; i += 2) {
        acc[0] = acc[0] + d2(&u[i], &v[i]);
    }

    if (i < n) {
        acc[1] = acc[1] + d2_scalar(&u[i], &v[i]);
    }

    return (acc[0] + acc[1]).sum();
}





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

