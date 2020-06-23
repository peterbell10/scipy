
#ifdef __SSE2__
#include <emmintrin.h>

struct ckdtree_vec2d
{
    __m128d x;

    ckdtree_vec2d operator + (ckdtree_vec2d rhs) const
    {
        return {_mm_add_pd(x, rhs.x)};
    }
    ckdtree_vec2d operator - (ckdtree_vec2d rhs) const
    {
        return {_mm_sub_pd(x, rhs.x)};
    }
    ckdtree_vec2d operator * (ckdtree_vec2d rhs) const
    {
        return {_mm_mul_pd(x, rhs.x)};
    }

    static ckdtree_vec2d loadu(const double * data)
    {
        return {_mm_loadu_pd(data)};
    }

    static ckdtree_vec2d from_scalar(double data)
    {
        return {_mm_set_sd(data)};
    }

    static ckdtree_vec2d splat(double data)
    {
        return {_mm_set_pd1(data)};
    }

    double sum() const
    {
        return x[0] + x[1];
    }
};

#else // __SSE2__

struct ckdtree_vec2d
{
    double x[2];

    ckdtree_vec2d operator + (ckdtree_vec2d rhs) const
    {
        return {{x[0] + rhs.x[0], x[1] + rhs.x[1]}};
    }
    ckdtree_vec2d operator - (ckdtree_vec2d rhs) const
    {
        return {{x[0] - rhs.x[0], x[1] - rhs.x[1]}};
    }
    ckdtree_vec2d operator * (ckdtree_vec2d rhs) const
    {
        return {{x[0] * rhs.x[0], x[1] * rhs.x[1]}};
    }

    static ckdtree_vec2d loadu(const double * data)
    {
        return {{data[0], data[1]}};
    }

    static ckdtree_vec2d from_scalar(double data)
    {
        return {{data, 0.}};
    }

    static ckdtree_vec2d splat(double data)
    {
        return {{data, data}};
    }

    double sum() const
    {
        return x[0] + x[1];
    }
};

#endif
