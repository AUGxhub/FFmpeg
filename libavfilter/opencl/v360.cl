/*
 * Copyright (c) 2021 Paul B Mahol
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

const sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR);

static float2 scale(float2 x, int2 s)
{
    return (0.5f * x + 0.5f) * ((float2)(s.x, s.y) - 1.f);
}

static float2 rescale(int2 x, int2 s)
{
    return ((float2)(2.f, 2.f) * (float2)(x.x, x.y) + 1.f) / (float2)(s.x, s.y) - 1.f;
}

__kernel void equirect_to_xyz(global float3 *dst,
                              float2 flat_range)
{
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(get_global_size(0), get_global_size(1));
    float2 f = flat_range * rescale(p, size);

    float sin_phi   = sin(f.x);
    float cos_phi   = cos(f.x);
    float sin_theta = sin(f.y);
    float cos_theta = cos(f.y);

    float3 vec;

    vec.x = cos_theta * sin_phi;
    vec.y = sin_theta;
    vec.z = cos_theta * cos_phi;

    dst[p.y * size.x + p.x] = vec;
}

__kernel void flat_to_xyz(global float3 *dst,
                          float2 flat_range)
{
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(get_global_size(0), get_global_size(1));
    float2 f = flat_range * rescale(p, size);

    float3 vec;

    vec.xy = f.xy;
    vec.z = 1.0;

    vec = normalize(vec);

    dst[p.y * size.x + p.x] = vec;
}

__kernel void fisheye_to_xyz(global float3 *dst,
                             float2 flat_range)
{
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(get_global_size(0), get_global_size(1));
    float2 uv = flat_range * rescale(p, size);

    float phi   = atan2(uv.y, uv.x);
    float theta = M_PI_2 * (1.f - hypot(uv.x, uv.y));

    float sin_phi   = sin(phi);
    float cos_phi   = cos(phi);
    float sin_theta = sin(theta);
    float cos_theta = cos(theta);

    float3 vec;

    vec.x = cos_theta * cos_phi;
    vec.y = cos_theta * sin_phi;
    vec.z = sin_theta;

    vec = normalize(vec);

    dst[p.y * size.x + p.x] = vec;
}

__kernel void xyz_to_flat(global float2 *dst,
                          float2 iflat_range,
                          global float3 *m,
                          __read_only image2d_t src)
{
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(get_global_size(0), get_global_size(1));
    float3 vec = m[p.x + size.x * p.y];

    float theta = acos(vec.z);
    float r = tan(theta);
    float rr = fabs(r) < 1e+6f ? r : hypot((float)(size.x), (float)(size.y));
    float zf = vec.z;
    float h = hypot(vec.x, vec.y);
    float c = h <= 1e-6f ? 1.f : rr / h;

    float2 uv = vec.xy * c / iflat_range;

    uv = zf >= 0.f ? scale(uv, size) : 0.f;

    dst[p.x + p.y * size.x] = uv;
}

__kernel void xyz_to_equirect(global float2 *dst,
                              float2 iflat_range,
                              global float3 *m,
                              __read_only image2d_t src)
{
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(get_global_size(0), get_global_size(1));
    float3 vec = m[p.x + size.x * p.y];
    float2 uv = ((float2)(atan2(vec.x, vec.z), asin(vec.y))) / iflat_range;

    uv = scale(uv, size);

    dst[p.x + p.y * size.x] = uv;
}

__kernel void xyz_to_fisheye(global float2 *dst,
                             float2 iflat_range,
                             global float3 *m,
                             __read_only image2d_t src)
{
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(get_global_size(0), get_global_size(1));
    float3 vec = m[p.x + size.x * p.y];

    float h   = hypot(vec.x, vec.y);
    float lh  = h > 0.f ? h : 1.f;
    float phi = atan2(h, vec.z) / M_PI;

    float2 uv = vec.xy / lh * phi / iflat_range;

    uv = (uv + 0.5f) * (float2)(size.x, size.y);

    dst[p.x + p.y * size.x] = uv;
}

__kernel void stereographic_to_xyz(global float3 *dst,
                                   float2 flat_range)
{
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(get_global_size(0), get_global_size(1));

    float2 uv = rescale(p, size) * flat_range;

    const float r = hypot(uv.x, uv.y);
    const float theta = atan(r) * 2.f;
    const float sin_theta = sin(theta);

    float3 vec;

    vec.xy = uv / r * sin_theta;
    vec.z = cos(theta);

    vec = normalize(vec);

    dst[p.y * size.x + p.x] = vec;
}

__kernel void xyz_to_stereographic(global float2 *dst,
                                   float2 iflat_range,
                                   global float3 *m,
                                   __read_only image2d_t src)
{
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(get_global_size(0), get_global_size(1));
    float3 vec = m[p.x + size.x * p.y];

    const float theta = acos(vec.z);
    const float r = tan(theta * 0.5f);
    const float c = r / hypot(vec.x, vec.y);

    float2 uv = vec.xy * c / iflat_range;

    uv = scale(uv, size);

    dst[p.x + p.y * size.x] = uv;
}

__kernel void xyz_to_orthographic(global float2 *dst,
                                  float2 iflat_range,
                                  global float3 *m,
                                  __read_only image2d_t src)
{
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(get_global_size(0), get_global_size(1));
    float3 vec = m[p.x + size.x * p.y];

    const float theta = acos(vec.z);
    const float r = sin(theta);
    const float c = r / hypot(vec.x, vec.y);

    float2 uv = vec.xy * c / iflat_range;

    uv = scale(uv, size);

    dst[p.x + p.y * size.x] = uv;
}

__kernel void orthographic_to_xyz(global float3 *dst,
                                  float2 flat_range)
{
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(get_global_size(0), get_global_size(1));
    float2 uv = rescale(p, size) * flat_range;

    const float r = hypot(uv.x, uv.y);
    const float theta = asin(r);

    float3 vec;

    vec.z = cos(theta);
    if (vec.z > 0.f) {
        vec.xy = uv;

        vec = normalize(vec);
    } else {
        vec.x = 0.f;
        vec.y = 0.f;
        vec.z = 1.f;
    }

    dst[p.y * size.x + p.x] = vec;
}

__kernel void tetrahedron_to_xyz(global float3 *dst,
                                 float2 flat_range)
{
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(get_global_size(0), get_global_size(1));

    const float uf = (p.x + 0.5f) / size.x;
    const float vf = (p.y + 0.5f) / size.y;

    float3 vec;

    vec.x = uf < 0.5f ? uf * 4.f - 1.f : 3.f - uf * 4.f;
    vec.y = 1.f - vf * 2.f;
    vec.z = 2.f * fabs(1.f - fabs(1.f - uf * 2.f + vf)) - 1.f;

    vec = normalize(vec);

    dst[p.y * size.x + p.x] = vec;
}

__kernel void xyz_to_tetrahedron(global float2 *dst,
                                 float2 iflat_range,
                                 global float3 *m,
                                 __read_only image2d_t src)
{
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(get_global_size(0), get_global_size(1));
    float3 vec = m[p.x + size.x * p.y];

    float4 d;

    d.x = vec.x * 1.f + vec.y * 1.f + vec.z *-1.f;
    d.y = vec.x *-1.f + vec.y *-1.f + vec.z *-1.f;
    d.z = vec.x * 1.f + vec.y *-1.f + vec.z * 1.f;
    d.w = vec.x *-1.f + vec.y * 1.f + vec.z * 1.f;
    d.x = fmax(fmax(d.x, d.y), fmax(d.z, d.w));

    float x, y, z;

    x =  vec.x / d.x;
    y =  vec.y / d.x;
    z = -vec.z / d.x;

    float2 uv;

    uv.y = 0.5f - y * 0.5f;

    if ((x + y >= 0.f &&  y + z >= 0.f && -z - x <= 0.f) ||
        (x + y <= 0.f && -y + z >= 0.f &&  z - x >= 0.f)) {
        uv.x = 0.25f * x + 0.25f;
    } else {
        uv.x = 0.75f - 0.25f * x;
    }

    uv.x *= size.x;
    uv.y *= size.y;

    dst[p.x + p.y * size.x] = uv;
}

__kernel void remap(__write_only image2d_t dst,
                    __read_only image2d_t src,
                    global float2 *remap)
{
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 size = (int2)(get_global_size(0), get_global_size(1));

    float2 f = remap[p.y * size.x + p.x];
    float4 v = read_imagef(src, sampler, f.xy);

    write_imagef(dst, p, v);
}

static float4 multiply_quaternion(float4 a, float4 b)
{
    float4 c;

    c.s0 = a.s0 * b.s0 - a.s1 * b.s1 - a.s2 * b.s2 - a.s3 * b.s3;
    c.s1 = a.s1 * b.s0 + a.s0 * b.s1 + a.s2 * b.s3 - a.s3 * b.s2;
    c.s2 = a.s2 * b.s0 + a.s0 * b.s2 + a.s3 * b.s1 - a.s1 * b.s3;
    c.s3 = a.s3 * b.s0 + a.s0 * b.s3 + a.s1 * b.s2 - a.s2 * b.s1;

    return c;
}

__kernel void rotate(global float3 *dst,
                     float8 quaternion)
{
    int2 size = (int2)(get_global_size(0), get_global_size(1));
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    float3 vec = dst[p.y * size.x + p.x];
    float4 qv, temp, rqv;

    qv.x = 0.f;
    qv.s123 = vec.xyz;

    temp = multiply_quaternion(quaternion.s0123, qv);
    rqv  = multiply_quaternion(temp, quaternion.s4567);

    vec.xyz = rqv.s123;

    vec = normalize(vec);

    dst[p.y * size.x + p.x] = vec;
}

__kernel void mirror(global float3 *dst,
                     float3 mirror)
{
    int2 size = (int2)(get_global_size(0), get_global_size(1));
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    float3 vec = dst[p.y * size.x + p.x];

    vec.xyz *= mirror.xyz;

    dst[p.y * size.x + p.x] = vec;
}

__kernel void flip(global float2 *dst,
                   int2 flip)
{
    int2 size = (int2)(get_global_size(0), get_global_size(1));
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    float2 uv = dst[p.y * size.x + p.x];

    if (flip.x)
        uv.x = size.x - 1 - uv.x;

    if (flip.y)
        uv.y = size.y - 1 - uv.y;

    dst[p.y * size.x + p.x] = uv;
}
