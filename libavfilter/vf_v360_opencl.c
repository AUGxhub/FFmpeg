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
#include <float.h>

#include "libavutil/avassert.h"
#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "internal.h"
#include "opencl.h"
#include "opencl_source.h"
#include "v360.h"
#include "video.h"

static const char *input_kernels[NB_PROJECTIONS] =
{
    [DUAL_FISHEYE]    = "xyz_to_dfisheye",
    [EQUIRECTANGULAR] = "xyz_to_equirect",
    [FISHEYE]         = "xyz_to_fisheye",
    [FLAT]            = "xyz_to_flat",
    [ORTHOGRAPHIC]    = "xyz_to_orthographic",
    [STEREOGRAPHIC]   = "xyz_to_stereographic",
    [TETRAHEDRON]     = "xyz_to_tetrahedron",
};

static const char *output_kernels[NB_PROJECTIONS] =
{
    [DUAL_FISHEYE]    = "dfisheye_to_xyz",
    [EQUIRECTANGULAR] = "equirect_to_xyz",
    [FISHEYE]         = "fisheye_to_xyz",
    [FLAT]            = "flat_to_xyz",
    [ORTHOGRAPHIC]    = "orthographic_to_xyz",
    [STEREOGRAPHIC]   = "stereographic_to_xyz",
    [TETRAHEDRON]     = "tetrahedron_to_xyz",
};

typedef struct V360OpenCLContext {
    OpenCLFilterContext   ocf;
    int                   initialised;

    int                   in;
    int                   out;
    int                   prev_in[2];
    int                   prev_out[2];
    float                 h_fov;
    float                 v_fov;
    float                 d_fov;
    float                 ih_fov;
    float                 iv_fov;
    float                 id_fov;
    float                 yaw;
    float                 pitch;
    float                 roll;
    char                 *rorder;
    int                   iflip[2];
    int                   flip[3];
    int                   prev_iflip[2][2];
    int                   prev_flip[2][3];

    cl_kernel             in_kernel;
    cl_kernel             out_kernel;
    cl_kernel             rotate_kernel;
    cl_kernel             mirror_kernel;
    cl_kernel             flip_kernel;
    cl_kernel             remap_kernel;
    cl_mem                vectors[2];
    cl_mem                remap[2];
    cl_command_queue      command_queue;

    float                 prev_flat_range[2][2];
    float                 prev_iflat_range[2][2];
    float                 flat_range[2];
    float                 iflat_range[2];
    float                 output_mirror_modifier[3];
    float                 rot_quaternion[2][4];

    int                   need_rotate[2];
    int                   need_mirror[2];
    int                   need_flip[2];
    int                   rotation_order[3];
} V360OpenCLContext;

static int v360_opencl_init(AVFilterContext *avctx, int width, int height)
{
    V360OpenCLContext *ctx = avctx->priv;
    cl_int cle;
    int err;

    err = ff_opencl_filter_load_program(avctx, &ff_opencl_source_v360, 1);
    if (err < 0)
        goto fail;

    ctx->command_queue = clCreateCommandQueue(ctx->ocf.hwctx->context,
                                              ctx->ocf.hwctx->device_id,
                                              0, &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create OpenCL "
                     "command queue %d.\n", cle);

    ctx->in_kernel = clCreateKernel(ctx->ocf.program,
                                    input_kernels[ctx->in], &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create "
                     "input_format kernel %d.\n", cle);

    ctx->rotate_kernel = clCreateKernel(ctx->ocf.program,
                                        "rotate", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create "
                     "rotate kernel %d.\n", cle);

    ctx->mirror_kernel = clCreateKernel(ctx->ocf.program,
                                        "mirror", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create "
                     "mirror kernel %d.\n", cle);

    ctx->flip_kernel = clCreateKernel(ctx->ocf.program,
                                      "flip", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create "
                     "flip kernel %d.\n", cle);

    ctx->out_kernel = clCreateKernel(ctx->ocf.program,
                                     output_kernels[ctx->out], &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create "
                     "output_format kernel %d.\n", cle);

    ctx->remap_kernel = clCreateKernel(ctx->ocf.program,
                                       "remap", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create "
                     "remap kernel %d.\n", cle);

    ctx->vectors[0] = clCreateBuffer(ctx->ocf.hwctx->context, 0,
                                     width * height * sizeof(cl_float3),
                                     NULL, &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create "
                     "vectors image %d.\n", cle);

    ctx->vectors[1] = clCreateBuffer(ctx->ocf.hwctx->context, 0,
                                     width * height * sizeof(cl_float3),
                                     NULL, &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create "
                     "vectors image %d.\n", cle);

    ctx->remap[0] = clCreateBuffer(ctx->ocf.hwctx->context, 0,
                                   width * height * sizeof(cl_float2),
                                   NULL, &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create "
                     "remap image %d.\n", cle);

    ctx->remap[1] = clCreateBuffer(ctx->ocf.hwctx->context, 0,
                                   width * height * sizeof(cl_float2),
                                   NULL, &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create "
                     "remap image %d.\n", cle);

    ctx->initialised = 1;
    return 0;

fail:
    CL_RELEASE_KERNEL(ctx->in_kernel);
    CL_RELEASE_KERNEL(ctx->out_kernel);
    CL_RELEASE_KERNEL(ctx->rotate_kernel);
    CL_RELEASE_KERNEL(ctx->mirror_kernel);
    CL_RELEASE_KERNEL(ctx->flip_kernel);
    CL_RELEASE_KERNEL(ctx->remap_kernel);

    CL_RELEASE_MEMORY(ctx->vectors[0]);
    CL_RELEASE_MEMORY(ctx->vectors[1]);

    CL_RELEASE_MEMORY(ctx->remap[0]);
    CL_RELEASE_MEMORY(ctx->remap[1]);

    CL_RELEASE_QUEUE(ctx->command_queue);

    return err;
}

static void fov_from_dfov(int format, float d_fov, float w, float h, float *h_fov, float *v_fov)
{
    switch (format) {
    case EQUIRECTANGULAR:
        *h_fov = d_fov;
        *v_fov = d_fov * 0.5f;
        break;
    case ORTHOGRAPHIC:
        {
            const float d = 0.5f * hypotf(w, h);
            const float l = sinf(d_fov * M_PI / 360.f) / d;

            *h_fov = asinf(w * 0.5f * l) * 360.f / M_PI;
            *v_fov = asinf(h * 0.5f * l) * 360.f / M_PI;

            if (d_fov > 180.f) {
                *h_fov = 180.f - *h_fov;
                *v_fov = 180.f - *v_fov;
            }
        }
        break;
    case EQUISOLID:
        {
            const float d = 0.5f * hypotf(w, h);
            const float l = d / (sinf(d_fov * M_PI / 720.f));

            *h_fov = 2.f * asinf(w * 0.5f / l) * 360.f / M_PI;
            *v_fov = 2.f * asinf(h * 0.5f / l) * 360.f / M_PI;
        }
        break;
    case STEREOGRAPHIC:
        {
            const float d = 0.5f * hypotf(w, h);
            const float l = d / (tanf(d_fov * M_PI / 720.f));

            *h_fov = 2.f * atan2f(w * 0.5f, l) * 360.f / M_PI;
            *v_fov = 2.f * atan2f(h * 0.5f, l) * 360.f / M_PI;
        }
        break;
    case DUAL_FISHEYE:
        {
            const float d = hypotf(w * 0.5f, h);

            *h_fov = 0.5f * w / d * d_fov;
            *v_fov =        h / d * d_fov;
        }
        break;
    case FISHEYE:
        {
            const float d = hypotf(w, h);

            *h_fov = w / d * d_fov;
            *v_fov = h / d * d_fov;
        }
        break;
    case FLAT:
    default:
        {
            const float da = tanf(0.5f * FFMIN(d_fov, 359.f) * M_PI / 180.f);
            const float d = hypotf(w, h);

            *h_fov = atan2f(da * w, d) * 360.f / M_PI;
            *v_fov = atan2f(da * h, d) * 360.f / M_PI;

            if (*h_fov < 0.f)
                *h_fov += 360.f;
            if (*v_fov < 0.f)
                *v_fov += 360.f;
        }
        break;
    }
}

static void v360_set_fov(AVFilterContext *avctx)
{
    AVFilterLink *inlink = avctx->inputs[0];
    V360OpenCLContext *ctx = avctx->priv;
    float default_ih_fov, default_iv_fov;
    float default_h_fov, default_v_fov;

    switch (ctx->out) {
    case CYLINDRICAL:
    case FLAT:
        default_h_fov = 90.f;
        default_v_fov = 45.f;
        break;
    case EQUISOLID:
    case ORTHOGRAPHIC:
    case STEREOGRAPHIC:
    case DUAL_FISHEYE:
    case FISHEYE:
        default_h_fov = 180.f;
        default_v_fov = 180.f;
        break;
    default:
        default_h_fov = 360.f;
        default_v_fov = 180.f;
        break;
    }

    switch (ctx->in) {
    case CYLINDRICAL:
    case FLAT:
        default_ih_fov = 90.f;
        default_iv_fov = 45.f;
        break;
    case EQUISOLID:
    case ORTHOGRAPHIC:
    case STEREOGRAPHIC:
    case DUAL_FISHEYE:
    case FISHEYE:
        default_ih_fov = 180.f;
        default_iv_fov = 180.f;
        break;
    default:
        default_ih_fov = 360.f;
        default_iv_fov = 180.f;
        break;
    }

    if (ctx->h_fov == 0.f)
        ctx->h_fov = default_h_fov;

    if (ctx->v_fov == 0.f)
        ctx->v_fov = default_v_fov;

    if (ctx->ih_fov == 0.f)
        ctx->ih_fov = default_ih_fov;

    if (ctx->iv_fov == 0.f)
        ctx->iv_fov = default_iv_fov;

    if (ctx->id_fov > 0.f)
        fov_from_dfov(ctx->in, ctx->id_fov, inlink->w, inlink->h, &ctx->ih_fov, &ctx->iv_fov);

    if (ctx->d_fov > 0.f)
        fov_from_dfov(ctx->out, ctx->d_fov, inlink->w, inlink->h, &ctx->h_fov, &ctx->v_fov);
}

static int v360_opencl_config_input(AVFilterLink *inlink)
{
    AVFilterContext *avctx = inlink->dst;
    V360OpenCLContext *ctx = avctx->priv;

    v360_set_fov(avctx);

    ctx->prev_in[0] = ctx->prev_out[0] = -1;
    ctx->prev_in[1] = ctx->prev_out[1] = -1;

    ctx->rot_quaternion[0][0] = 1.f;
    ctx->rot_quaternion[0][1] = ctx->rot_quaternion[0][2] = ctx->rot_quaternion[0][3] = 0.f;

    return ff_opencl_filter_config_input(inlink);
}

static void multiply_quaternion(float c[4], const float a[4], const float b[4])
{
    c[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    c[1] = a[1] * b[0] + a[0] * b[1] + a[2] * b[3] - a[3] * b[2];
    c[2] = a[2] * b[0] + a[0] * b[2] + a[3] * b[1] - a[1] * b[3];
    c[3] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1];
}

static void conjugate_quaternion(float d[4], const float q[4])
{
    d[0] =  q[0];
    d[1] = -q[1];
    d[2] = -q[2];
    d[3] = -q[3];
}

static inline void set_mirror_modifier(int flip[3], float *modifier)
{
    modifier[0] = flip[0] ? -1.f : 1.f;
    modifier[1] = flip[1] ? -1.f : 1.f;
    modifier[2] = flip[2] ? -1.f : 1.f;
}

static inline void calculate_rotation(float yaw, float pitch, float roll,
                                      float rot_quaternion[2][4],
                                      const int rotation_order[3])
{
    const float yaw_rad   = yaw   * M_PI / 180.f;
    const float pitch_rad = pitch * M_PI / 180.f;
    const float roll_rad  = roll  * M_PI / 180.f;

    const float sin_yaw   = sinf(yaw_rad   * 0.5f);
    const float cos_yaw   = cosf(yaw_rad   * 0.5f);
    const float sin_pitch = sinf(pitch_rad * 0.5f);
    const float cos_pitch = cosf(pitch_rad * 0.5f);
    const float sin_roll  = sinf(roll_rad  * 0.5f);
    const float cos_roll  = cosf(roll_rad  * 0.5f);

    float m[3][4];
    float tmp[2][4];

    m[0][0] = cos_yaw;   m[0][1] = 0.f;       m[0][2] = sin_yaw; m[0][3] = 0.f;
    m[1][0] = cos_pitch; m[1][1] = sin_pitch; m[1][2] = 0.f;     m[1][3] = 0.f;
    m[2][0] = cos_roll;  m[2][1] = 0.f;       m[2][2] = 0.f;     m[2][3] = sin_roll;

    multiply_quaternion(tmp[0], rot_quaternion[0], m[rotation_order[0]]);
    multiply_quaternion(tmp[1], tmp[0], m[rotation_order[1]]);
    multiply_quaternion(rot_quaternion[0], tmp[1], m[rotation_order[2]]);

    conjugate_quaternion(rot_quaternion[1], rot_quaternion[0]);
}

static int get_rorder(char c)
{
    switch (c) {
    case 'Y':
    case 'y':
        return YAW;
    case 'P':
    case 'p':
        return PITCH;
    case 'R':
    case 'r':
        return ROLL;
    default:
        return -1;
    }
}

static int v360_opencl_filter_frame(AVFilterLink *inlink, AVFrame *input)
{
    AVFilterContext *avctx = inlink->dst;
    AVFilterLink *outlink = avctx->outputs[0];
    V360OpenCLContext *ctx = avctx->priv;
    AVFrame *output = NULL;
    AVHWFramesContext *input_frames_ctx;
    enum AVPixelFormat in_format;
    size_t global_work[2];
    cl_mem src, dst;
    int err, cle;

    if (!input->hw_frames_ctx)
        return AVERROR(EINVAL);
    input_frames_ctx = (AVHWFramesContext*)input->hw_frames_ctx->data;
    in_format = input_frames_ctx->sw_format;

    output = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!output) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    err = av_frame_copy_props(output, input);
    if (err < 0)
        goto fail;

    if (!ctx->initialised) {
        err = v360_opencl_init(avctx, inlink->w, inlink->h);
        if (err < 0)
            goto fail;
    }

    v360_set_fov(avctx);

    switch (ctx->out) {
    case EQUIRECTANGULAR:
        ctx->flat_range[0] = ctx->h_fov * M_PI / 360.f;
        ctx->flat_range[1] = ctx->v_fov * M_PI / 360.f;
        break;
    case FLAT:
        ctx->flat_range[0] = tanf(0.5f * ctx->h_fov * M_PI / 180.f);
        ctx->flat_range[1] = tanf(0.5f * ctx->v_fov * M_PI / 180.f);
        break;
    case DUAL_FISHEYE:
    case FISHEYE:
        ctx->flat_range[0] = ctx->h_fov / 180.f;
        ctx->flat_range[1] = ctx->v_fov / 180.f;
        break;
    case ORTHOGRAPHIC:
        ctx->flat_range[0] = sinf(FFMIN(ctx->h_fov, 180.f) * M_PI / 360.f);
        ctx->flat_range[1] = sinf(FFMIN(ctx->v_fov, 180.f) * M_PI / 360.f);
        break;
    case STEREOGRAPHIC:
        ctx->flat_range[0] = tanf(FFMIN(ctx->h_fov, 359.f) * M_PI / 720.f);
        ctx->flat_range[1] = tanf(FFMIN(ctx->v_fov, 359.f) * M_PI / 720.f);
        break;
    }

    switch (ctx->in) {
    case EQUIRECTANGULAR:
        ctx->iflat_range[0] = ctx->ih_fov * M_PI / 360.f;
        ctx->iflat_range[1] = ctx->iv_fov * M_PI / 360.f;
        break;
    case FLAT:
        ctx->iflat_range[0] = tanf(0.5f * ctx->ih_fov * M_PI / 180.f);
        ctx->iflat_range[1] = tanf(0.5f * ctx->iv_fov * M_PI / 180.f);
        break;
    case DUAL_FISHEYE:
        ctx->iflat_range[0] = ctx->ih_fov / 360.f;
        ctx->iflat_range[1] = ctx->iv_fov / 360.f;
        break;
    case FISHEYE:
        ctx->iflat_range[0] = ctx->ih_fov / 180.f;
        ctx->iflat_range[1] = ctx->iv_fov / 180.f;
        break;
    case ORTHOGRAPHIC:
        ctx->iflat_range[0] = sinf(FFMIN(ctx->ih_fov, 180.f) * M_PI / 360.f);
        ctx->iflat_range[1] = sinf(FFMIN(ctx->iv_fov, 180.f) * M_PI / 360.f);
        break;
    case STEREOGRAPHIC:
        ctx->iflat_range[0] = tanf(FFMIN(ctx->ih_fov, 359.f) * M_PI / 720.f);
        ctx->iflat_range[1] = tanf(FFMIN(ctx->iv_fov, 359.f) * M_PI / 720.f);
        break;
    }

    for (int order = 0; order < NB_RORDERS; order++) {
        const char c = ctx->rorder[order];
        int rorder;

        if (c == '\0') {
            av_log(ctx, AV_LOG_WARNING,
                   "Incomplete rorder option. Direction for all 3 rotation orders should be specified. Switching to default rorder.\n");
            ctx->rotation_order[0] = YAW;
            ctx->rotation_order[1] = PITCH;
            ctx->rotation_order[2] = ROLL;
            break;
        }

        rorder = get_rorder(c);
        if (rorder == -1) {
            av_log(ctx, AV_LOG_WARNING,
                   "Incorrect rotation order symbol '%c' in rorder option. Switching to default rorder.\n", c);
            ctx->rotation_order[0] = YAW;
            ctx->rotation_order[1] = PITCH;
            ctx->rotation_order[2] = ROLL;
            break;
        }

        ctx->rotation_order[order] = rorder;
    }

    calculate_rotation(ctx->yaw, ctx->pitch, ctx->roll,
                       ctx->rot_quaternion, ctx->rotation_order);

    set_mirror_modifier(ctx->flip, ctx->output_mirror_modifier);

    for (int p = 0; p < FF_ARRAY_ELEMS(output->data); p++) {
        const int pp = p > 0 && p < 3 ? 1 : 0;
        src = (cl_mem) input->data[p];
        dst = (cl_mem) output->data[p];

        if (!dst || !src)
            break;

        err = ff_opencl_filter_work_size_from_image(avctx, global_work,
                                                    output, p, 0);
        if (err < 0)
            goto fail;

        if ((pp == p) && ((ctx->prev_out[pp] != ctx->out) || (
                          ctx->prev_flat_range[pp][0] != ctx->flat_range[0] ||
                          ctx->prev_flat_range[pp][1] != ctx->flat_range[1]))) {
            CL_SET_KERNEL_ARG(ctx->out_kernel, 0, cl_mem, &ctx->vectors[pp]);
            CL_SET_KERNEL_ARG(ctx->out_kernel, 1, cl_float2, &ctx->flat_range);

            cle = clEnqueueNDRangeKernel(ctx->command_queue, ctx->out_kernel, 2, NULL,
                                         global_work, NULL, 0, NULL, NULL);

            CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue output_format kernel: %d.\n", cle);

            ctx->need_rotate[pp] = 1;
            ctx->need_mirror[pp] = 1;
            ctx->prev_flat_range[pp][0] = ctx->flat_range[0];
            ctx->prev_flat_range[pp][1] = ctx->flat_range[1];
            ctx->prev_out[pp] = ctx->out;
            ctx->prev_in[pp] = -1;
        }

        if ((pp == p) && (ctx->need_rotate[pp] ||
                          (ctx->yaw != 0.f || ctx->pitch != 0.f || ctx->roll != 0.f))) {
            CL_SET_KERNEL_ARG(ctx->rotate_kernel, 0, cl_mem, &ctx->vectors[pp]);
            CL_SET_KERNEL_ARG(ctx->rotate_kernel, 1, cl_float8, &ctx->rot_quaternion);

            cle = clEnqueueNDRangeKernel(ctx->command_queue, ctx->rotate_kernel, 2, NULL,
                                         global_work, NULL, 0, NULL, NULL);

            CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue rotate kernel: %d.\n", cle);

            ctx->need_mirror[pp] = 1;
            ctx->need_rotate[pp] = 0;
            ctx->prev_in[pp] = -1;
        }

        if ((pp == p) && (ctx->need_mirror[pp] ||
                          (ctx->flip[0] != ctx->prev_flip[pp][0] ||
                           ctx->flip[1] != ctx->prev_flip[pp][1] ||
                           ctx->flip[2] != ctx->prev_flip[pp][2]))) {
            CL_SET_KERNEL_ARG(ctx->mirror_kernel, 0, cl_mem, &ctx->vectors[pp]);
            CL_SET_KERNEL_ARG(ctx->mirror_kernel, 1, cl_float3, &ctx->output_mirror_modifier);

            cle = clEnqueueNDRangeKernel(ctx->command_queue, ctx->mirror_kernel, 2, NULL,
                                         global_work, NULL, 0, NULL, NULL);

            CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue mirror kernel: %d.\n", cle);

            ctx->prev_flip[pp][0] = ctx->flip[0];
            ctx->prev_flip[pp][1] = ctx->flip[1];
            ctx->prev_flip[pp][2] = ctx->flip[2];
            ctx->need_mirror[pp] = 0;
            ctx->prev_in[pp] = -1;
        }

        if ((pp == p) && ((ctx->prev_in[pp] != ctx->in) || (
                          ctx->prev_iflat_range[pp][0] != ctx->iflat_range[0] ||
                          ctx->prev_iflat_range[pp][1] != ctx->iflat_range[1]))) {
            CL_SET_KERNEL_ARG(ctx->in_kernel, 0, cl_mem, &ctx->remap[pp]);
            CL_SET_KERNEL_ARG(ctx->in_kernel, 1, cl_float2, &ctx->iflat_range);
            CL_SET_KERNEL_ARG(ctx->in_kernel, 2, cl_mem, &ctx->vectors[pp]);
            CL_SET_KERNEL_ARG(ctx->in_kernel, 3, cl_mem, &src);

            cle = clEnqueueNDRangeKernel(ctx->command_queue, ctx->in_kernel, 2, NULL,
                                         global_work, NULL, 0, NULL, NULL);

            CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue input_format kernel: %d.\n", cle);

            ctx->prev_iflat_range[pp][0] = ctx->iflat_range[0];
            ctx->prev_iflat_range[pp][1] = ctx->iflat_range[1];
            ctx->prev_in[pp] = ctx->in;
            ctx->need_flip[pp] = 1;
        }

        if ((pp == p) && (ctx->need_flip[pp] ||
                          (ctx->iflip[0] != ctx->prev_iflip[pp][0] ||
                           ctx->iflip[1] != ctx->prev_iflip[pp][1]))) {
            CL_SET_KERNEL_ARG(ctx->flip_kernel, 0, cl_mem, &ctx->remap[pp]);
            CL_SET_KERNEL_ARG(ctx->flip_kernel, 1, cl_int2, &ctx->iflip);

            cle = clEnqueueNDRangeKernel(ctx->command_queue, ctx->flip_kernel, 2, NULL,
                                         global_work, NULL, 0, NULL, NULL);

            CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue flip kernel: %d.\n", cle);

            ctx->need_flip[pp] = 0;
            ctx->prev_iflip[pp][0] = ctx->iflip[0];
            ctx->prev_iflip[pp][1] = ctx->iflip[1];
        }

        CL_SET_KERNEL_ARG(ctx->remap_kernel, 0, cl_mem, &dst);
        CL_SET_KERNEL_ARG(ctx->remap_kernel, 1, cl_mem, &src);
        CL_SET_KERNEL_ARG(ctx->remap_kernel, 2, cl_mem, &ctx->remap[pp]);

        cle = clEnqueueNDRangeKernel(ctx->command_queue, ctx->remap_kernel, 2, NULL,
                                     global_work, NULL, 0, NULL, NULL);

        CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue remap kernel: %d.\n", cle);

    }

    cle = clFlush(ctx->command_queue);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to flush command queue: %d.\n", cle);

    cle = clFinish(ctx->command_queue);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to finish kernel: %d.\n", cle);

    ctx->yaw = ctx->pitch = ctx->roll = 0.f;

    av_frame_free(&input);

    return ff_filter_frame(outlink, output);

fail:
    clFinish(ctx->command_queue);
    av_frame_free(&input);
    av_frame_free(&output);
    return err;
}

static av_cold void v360_opencl_uninit(AVFilterContext *avctx)
{
    V360OpenCLContext *ctx = avctx->priv;
    cl_int cle;

    CL_RELEASE_KERNEL(ctx->out_kernel);
    CL_RELEASE_KERNEL(ctx->rotate_kernel);
    CL_RELEASE_KERNEL(ctx->mirror_kernel);
    CL_RELEASE_KERNEL(ctx->flip_kernel);
    CL_RELEASE_KERNEL(ctx->in_kernel);
    CL_RELEASE_KERNEL(ctx->remap_kernel);

    CL_RELEASE_MEMORY(ctx->vectors[0]);
    CL_RELEASE_MEMORY(ctx->vectors[1]);

    CL_RELEASE_MEMORY(ctx->remap[0]);
    CL_RELEASE_MEMORY(ctx->remap[1]);

    CL_RELEASE_QUEUE(ctx->command_queue);

    ff_opencl_filter_uninit(avctx);
}

#define OFFSET(x) offsetof(V360OpenCLContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
#define TFLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_RUNTIME_PARAM)
static const AVOption v360_opencl_options[] = {
    { "input",  "set input projection",            OFFSET(in),     AV_OPT_TYPE_INT,    {.i64=EQUIRECTANGULAR}, 0,    NB_PROJECTIONS-1, FLAGS, "in" },
    {"equirect","equirectangular",                 0,              AV_OPT_TYPE_CONST,  {.i64=EQUIRECTANGULAR}, 0,                   0, FLAGS, "in" },
    {     "e",  "equirectangular",                 0,              AV_OPT_TYPE_CONST,  {.i64=EQUIRECTANGULAR}, 0,                   0, FLAGS, "in" },
    {   "flat", "regular video",                   0,              AV_OPT_TYPE_CONST,  {.i64=FLAT},            0,                   0, FLAGS, "in" },
    {  "fisheye","fisheye",                        0,              AV_OPT_TYPE_CONST,  {.i64=FISHEYE},         0,                   0, FLAGS, "in" },
    {"dfisheye","dual fisheye",                    0,              AV_OPT_TYPE_CONST,  {.i64=DUAL_FISHEYE},    0,                   0, FLAGS, "in" },
    {     "og", "orthographic",                    0,              AV_OPT_TYPE_CONST,  {.i64=ORTHOGRAPHIC},    0,                   0, FLAGS, "in" },
    {     "sg", "stereographic",                   0,              AV_OPT_TYPE_CONST,  {.i64=STEREOGRAPHIC},   0,                   0, FLAGS, "in" },
    {"tetrahedron", "tetrahedron",                 0,              AV_OPT_TYPE_CONST,  {.i64=TETRAHEDRON},     0,                   0, FLAGS, "in" },
    { "output", "set output projection",           OFFSET(out),    AV_OPT_TYPE_INT,    {.i64=FLAT},            0,    NB_PROJECTIONS-1, FLAGS, "out" },
    {"equirect","equirectangular",                 0,              AV_OPT_TYPE_CONST,  {.i64=EQUIRECTANGULAR}, 0,                   0, FLAGS, "out" },
    {     "e",  "equirectangular",                 0,              AV_OPT_TYPE_CONST,  {.i64=EQUIRECTANGULAR}, 0,                   0, FLAGS, "out" },
    {   "flat", "regular video",                   0,              AV_OPT_TYPE_CONST,  {.i64=FLAT},            0,                   0, FLAGS, "out" },
    {  "fisheye","fisheye",                        0,              AV_OPT_TYPE_CONST,  {.i64=FISHEYE},         0,                   0, FLAGS, "out" },
    {"dfisheye","dual fisheye",                    0,              AV_OPT_TYPE_CONST,  {.i64=DUAL_FISHEYE},    0,                   0, FLAGS, "out" },
    {     "og", "orthographic",                    0,              AV_OPT_TYPE_CONST,  {.i64=ORTHOGRAPHIC},    0,                   0, FLAGS, "out" },
    {     "sg", "stereographic",                   0,              AV_OPT_TYPE_CONST,  {.i64=STEREOGRAPHIC},   0,                   0, FLAGS, "out" },
    {"tetrahedron", "tetrahedron",                 0,              AV_OPT_TYPE_CONST,  {.i64=TETRAHEDRON},     0,                   0, FLAGS, "out" },
    {    "yaw", "yaw rotation",                    OFFSET(yaw),    AV_OPT_TYPE_FLOAT,  {.dbl=0.f},        -180.f,               180.f,TFLAGS, "yaw"},
    {  "pitch", "pitch rotation",                  OFFSET(pitch),  AV_OPT_TYPE_FLOAT,  {.dbl=0.f},        -180.f,               180.f,TFLAGS, "pitch"},
    {   "roll", "roll rotation",                   OFFSET(roll),   AV_OPT_TYPE_FLOAT,  {.dbl=0.f},        -180.f,               180.f,TFLAGS, "roll"},
    { "rorder", "rotation order",                  OFFSET(rorder), AV_OPT_TYPE_STRING, {.str="ypr"},           0,                   0,TFLAGS, "rorder"},
    { "h_fov",  "output horizontal field of view", OFFSET(h_fov),  AV_OPT_TYPE_FLOAT,  {.dbl=0.f},           0.f,               360.f,TFLAGS, "h_fov"},
    { "v_fov",  "output vertical field of view",   OFFSET(v_fov),  AV_OPT_TYPE_FLOAT,  {.dbl=0.f},           0.f,               360.f,TFLAGS, "v_fov"},
    { "d_fov",  "output diagonal field of view",   OFFSET(d_fov),  AV_OPT_TYPE_FLOAT,  {.dbl=0.f},           0.f,               360.f,TFLAGS, "d_fov"},
    { "h_flip", "flip out video horizontally",     OFFSET(flip[0]),AV_OPT_TYPE_BOOL,   {.i64=0},               0,                   1,TFLAGS, "h_flip"},
    { "v_flip", "flip out video vertically",       OFFSET(flip[1]),AV_OPT_TYPE_BOOL,   {.i64=0},               0,                   1,TFLAGS, "v_flip"},
    { "d_flip", "flip out video indepth",          OFFSET(flip[2]),AV_OPT_TYPE_BOOL,   {.i64=0},               0,                   1,TFLAGS, "d_flip"},
    {"ih_flip", "flip in video horizontally",      OFFSET(iflip[0]),AV_OPT_TYPE_BOOL,  {.i64=0},               0,                   1,TFLAGS, "ih_flip"},
    {"iv_flip", "flip in video vertically",        OFFSET(iflip[1]),AV_OPT_TYPE_BOOL,  {.i64=0},               0,                   1,TFLAGS, "iv_flip"},
    { "ih_fov", "input horizontal field of view",  OFFSET(ih_fov), AV_OPT_TYPE_FLOAT,  {.dbl=0.f},           0.f,               360.f,TFLAGS, "ih_fov"},
    { "iv_fov", "input vertical field of view",    OFFSET(iv_fov), AV_OPT_TYPE_FLOAT,  {.dbl=0.f},           0.f,               360.f,TFLAGS, "iv_fov"},
    { "id_fov", "input diagonal field of view",    OFFSET(id_fov), AV_OPT_TYPE_FLOAT,  {.dbl=0.f},           0.f,               360.f,TFLAGS, "id_fov"},
    { NULL }
};

AVFILTER_DEFINE_CLASS(v360_opencl);

static const AVFilterPad v360_opencl_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = &v360_opencl_filter_frame,
        .config_props = &v360_opencl_config_input,
    },
};

static const AVFilterPad v360_opencl_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &ff_opencl_filter_config_output,
    },
};

AVFilter ff_vf_v360_opencl = {
    .name            = "v360_opencl",
    .description     = NULL_IF_CONFIG_SMALL("Convert 360 projection of video via OpenCL."),
    .priv_size       = sizeof(V360OpenCLContext),
    .priv_class      = &v360_opencl_class,
    .init            = &ff_opencl_filter_init,
    .uninit          = &v360_opencl_uninit,
    FILTER_INPUTS(v360_opencl_inputs),
    FILTER_OUTPUTS(v360_opencl_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_OPENCL),
    .process_command = ff_filter_process_command,
    .flags_internal  = FF_FILTER_FLAG_HWFRAME_AWARE,
};
