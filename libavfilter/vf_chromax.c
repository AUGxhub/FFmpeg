/*
 * Copyright (c) 2020 Paul B Mahol
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

#include "libavutil/avstring.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "video.h"

typedef struct ChromaXContext {
    const AVClass *class;
    int mode;

    int depth;
    int chroma_w;
    int chroma_h;
    int nb_planes;
    int linesize[4];
    int planeheight[4];
    int planewidth[4];

    AVFrame *out;
    int (*filter_slice)(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs);
} ChromaXContext;

static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV440P, AV_PIX_FMT_YUV411P, AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_YUVA420P, AV_PIX_FMT_YUVA422P, AV_PIX_FMT_YUVA444P,
    AV_PIX_FMT_YUVJ444P, AV_PIX_FMT_YUVJ440P, AV_PIX_FMT_YUVJ422P, AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUVJ411P,
    AV_PIX_FMT_YUV420P9,   AV_PIX_FMT_YUV422P9,   AV_PIX_FMT_YUV444P9,
    AV_PIX_FMT_YUV420P10,  AV_PIX_FMT_YUV422P10,  AV_PIX_FMT_YUV440P10, AV_PIX_FMT_YUV444P10,
    AV_PIX_FMT_YUV444P12,  AV_PIX_FMT_YUV422P12,  AV_PIX_FMT_YUV440P12, AV_PIX_FMT_YUV420P12,
    AV_PIX_FMT_YUV444P14,  AV_PIX_FMT_YUV422P14,  AV_PIX_FMT_YUV420P14,
    AV_PIX_FMT_YUV420P16,  AV_PIX_FMT_YUV422P16,  AV_PIX_FMT_YUV444P16,
    AV_PIX_FMT_YUVA420P9,  AV_PIX_FMT_YUVA422P9,  AV_PIX_FMT_YUVA444P9,
    AV_PIX_FMT_YUVA420P10, AV_PIX_FMT_YUVA422P10, AV_PIX_FMT_YUVA444P10,
    AV_PIX_FMT_YUVA422P12, AV_PIX_FMT_YUVA444P12,
    AV_PIX_FMT_YUVA420P16, AV_PIX_FMT_YUVA422P16, AV_PIX_FMT_YUVA444P16,
    AV_PIX_FMT_NONE
};

static float magnitude_fun(float x, float y, float half, float scale)
{
    return M_SQRT2 * hypotf(x - half, y - half);
}

static float average_fun(float x, float y, float half, float scale)
{
    return (x + y) * 0.5f;
}

static float difference_fun(float x, float y, float half, float scale)
{
    return fabsf(x - y);
}

static float max_fun(float x, float y, float half, float scale)
{
    return fmaxf(x, y);
}

static float min_fun(float x, float y, float half, float scale)
{
    return fminf(x, y);
}

static float extractclip_fun(float x, float y, float half, float scale)
{
    return av_clipf(half + x - y, 0.f, scale);
}

static float extractscale_fun(float x, float y, float half, float scale)
{
    return av_clipf(half + (x - y) * 0.5f, 0.f, scale);
}

static float sumclip_fun(float x, float y, float half, float scale)
{
    return av_clipf(fabsf(x - half) + fabsf(y - half), 0.f, scale);
}

#define FILTER_FUNC(mode, name, type, fun)                                             \
static int mode ## _slice##name(AVFilterContext *ctx, void *arg,                       \
                                    int jobnr, int nb_jobs)                            \
{                                                                                      \
    ChromaXContext *s = ctx->priv;                                                     \
    AVFrame *in = arg;                                                                 \
    AVFrame *out = s->out;                                                             \
    const float half = ((1 << s->depth) - 1) * 0.5f;                                   \
    const float scale = ((1 << s->depth) - 1);                                         \
    const int in_ylinesize = in->linesize[0];                                          \
    const int in_ulinesize = in->linesize[1];                                          \
    const int in_vlinesize = in->linesize[2];                                          \
    const int out_ulinesize = out->linesize[1];                                        \
    const int out_vlinesize = out->linesize[2];                                        \
    const int chroma_w = s->chroma_w;                                                  \
    const int chroma_h = s->chroma_h;                                                  \
    const int h = s->planeheight[1];                                                   \
    const int w = s->planewidth[1];                                                    \
    const int slice_start = (h * jobnr) / nb_jobs;                                     \
    const int slice_end = (h * (jobnr+1)) / nb_jobs;                                   \
    type *out_uptr = (type *)(out->data[1] + slice_start * out_ulinesize);             \
    type *out_vptr = (type *)(out->data[2] + slice_start * out_vlinesize);             \
                                                                                       \
    if (in != out) {                                                                   \
        const int h = s->planeheight[1];                                               \
        const int slice_start = (h * jobnr) / nb_jobs;                                 \
        const int slice_end = (h * (jobnr+1)) / nb_jobs;                               \
                                                                                       \
        av_image_copy_plane(out->data[1] + slice_start * out->linesize[1],             \
                            out->linesize[1],                                          \
                            in->data[1] + slice_start * in->linesize[1],               \
                            in->linesize[1],                                           \
                            s->linesize[1], slice_end - slice_start);                  \
                                                                                       \
        av_image_copy_plane(out->data[2] + slice_start * out->linesize[2],             \
                            out->linesize[2],                                          \
                            in->data[2] + slice_start * in->linesize[2],               \
                            in->linesize[2],                                           \
                            s->linesize[2], slice_end - slice_start);                  \
                                                                                       \
        if (s->nb_planes == 4) {                                                       \
            const int h = s->planeheight[3];                                           \
            const int slice_start = (h * jobnr) / nb_jobs;                             \
            const int slice_end = (h * (jobnr+1)) / nb_jobs;                           \
            av_image_copy_plane(out->data[3] + slice_start * out->linesize[3],         \
                                out->linesize[3],                                      \
                                in->data[3] + slice_start * in->linesize[3],           \
                                in->linesize[3],                                       \
                                s->linesize[3], slice_end - slice_start);              \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    for (int y = slice_start; y < slice_end; y++) {                                    \
        const type *in_uptr = (const type *)(in->data[1] + y * in_ulinesize);          \
        const type *in_vptr = (const type *)(in->data[2] + y * in_vlinesize);          \
        type *out_yptr = (type *)(out->data[0] + y * chroma_h * in_ylinesize);         \
                                                                                       \
        for (int x = 0; x < w; x++) {                                                  \
            const float cu = in_uptr[x];                                               \
            const float cv = in_vptr[x];                                               \
                                                                                       \
            out_yptr[chroma_w * x] = lrintf(fun(cu, cv, half, scale));                 \
        }                                                                              \
                                                                                       \
        out_uptr += out_ulinesize / sizeof(type);                                      \
        out_vptr += out_vlinesize / sizeof(type);                                      \
    }                                                                                  \
                                                                                       \
    return 0;                                                                          \
}

FILTER_FUNC(magnitude, 8,  uint8_t, magnitude_fun)
FILTER_FUNC(magnitude, 16, uint16_t, magnitude_fun)

FILTER_FUNC(average, 8,  uint8_t, average_fun)
FILTER_FUNC(average, 16, uint16_t, average_fun)

FILTER_FUNC(difference, 8,  uint8_t, difference_fun)
FILTER_FUNC(difference, 16, uint16_t, difference_fun)

FILTER_FUNC(max, 8,  uint8_t, max_fun)
FILTER_FUNC(max, 16, uint16_t, max_fun)

FILTER_FUNC(min, 8,  uint8_t, min_fun)
FILTER_FUNC(min, 16, uint16_t, min_fun)

FILTER_FUNC(extractclip, 8,  uint8_t, extractclip_fun)
FILTER_FUNC(extractclip, 16, uint16_t, extractclip_fun)

FILTER_FUNC(extractscale, 8,  uint8_t, extractscale_fun)
FILTER_FUNC(extractscale, 16, uint16_t, extractscale_fun)

FILTER_FUNC(sumclip, 8,  uint8_t, sumclip_fun)
FILTER_FUNC(sumclip, 16, uint16_t, sumclip_fun)

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    ChromaXContext *s = ctx->priv;
    AVFrame *out;

    switch (s->mode) {
    case 0:
        s->filter_slice = s->depth <= 8 ? magnitude_slice8 : magnitude_slice16;
        break;
    case 1:
        s->filter_slice = s->depth <= 8 ? average_slice8 : average_slice16;
        break;
    case 2:
        s->filter_slice = s->depth <= 8 ? difference_slice8 : difference_slice16;
        break;
    case 3:
        s->filter_slice = s->depth <= 8 ? max_slice8 : max_slice16;
        break;
    case 4:
        s->filter_slice = s->depth <= 8 ? min_slice8 : min_slice16;
        break;
    case 5:
        s->filter_slice = s->depth <= 8 ? extractclip_slice8 : extractclip_slice16;
        break;
    case 6:
        s->filter_slice = s->depth <= 8 ? extractscale_slice8 : extractscale_slice16;
        break;
    case 7:
        s->filter_slice = s->depth <= 8 ? sumclip_slice8 : sumclip_slice16;
        break;
    }

    if (av_frame_is_writable(in)) {
        out = in;
    } else {
        out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
        if (!out) {
            av_frame_free(&in);
            return AVERROR(ENOMEM);
        }
        av_frame_copy_props(out, in);
    }

    s->out = out;
    ff_filter_execute(ctx, s->filter_slice, in, NULL,
                      FFMIN3(s->planeheight[1],
                             s->planeheight[2],
                             ff_filter_get_nb_threads(ctx)));
    if (out != in)
        av_frame_free(&in);

    return ff_filter_frame(outlink, out);
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    ChromaXContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    int ret;

    s->nb_planes = desc->nb_components;
    s->depth = desc->comp[0].depth;
    s->chroma_w = 1 << desc->log2_chroma_w;
    s->chroma_h = 1 << desc->log2_chroma_h;
    s->planeheight[1] = s->planeheight[2] = AV_CEIL_RSHIFT(inlink->h, desc->log2_chroma_h);
    s->planeheight[0] = s->planeheight[3] = inlink->h;
    s->planewidth[1] = s->planewidth[2] = AV_CEIL_RSHIFT(inlink->w, desc->log2_chroma_w);
    s->planewidth[0] = s->planewidth[3] = inlink->w;

    if ((ret = av_image_fill_linesizes(s->linesize, inlink->format, inlink->w)) < 0)
        return ret;

    return 0;
}

#define OFFSET(x) offsetof(ChromaXContext, x)
#define VF AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_RUNTIME_PARAM

static const AVOption chromax_options[] = {
    { "mode", "set mode", OFFSET(mode), AV_OPT_TYPE_INT, {.i64=0}, 0, 7, VF, "mode" },
    {   "magnitude", "", 0, AV_OPT_TYPE_CONST, {.i64=0}, 0, 0, VF, "mode" },
    {   "average",   "", 0, AV_OPT_TYPE_CONST, {.i64=1}, 0, 0, VF, "mode" },
    {   "difference","", 0, AV_OPT_TYPE_CONST, {.i64=2}, 0, 0, VF, "mode" },
    {   "max",       "", 0, AV_OPT_TYPE_CONST, {.i64=3}, 0, 0, VF, "mode" },
    {   "min",       "", 0, AV_OPT_TYPE_CONST, {.i64=4}, 0, 0, VF, "mode" },
    { "extractclip", "", 0, AV_OPT_TYPE_CONST, {.i64=5}, 0, 0, VF, "mode" },
    { "extractscale","", 0, AV_OPT_TYPE_CONST, {.i64=6}, 0, 0, VF, "mode" },
    {   "sumclip",   "", 0, AV_OPT_TYPE_CONST, {.i64=7}, 0, 0, VF, "mode" },
    { NULL }
};

static const AVFilterPad inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
        .config_props = config_input,
    },
};

static const AVFilterPad outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
};

AVFILTER_DEFINE_CLASS(chromax);

const AVFilter ff_vf_chromax = {
    .name          = "chromax",
    .description   = NULL_IF_CONFIG_SMALL("Visualise chrominance."),
    .priv_size     = sizeof(ChromaXContext),
    .priv_class    = &chromax_class,
    FILTER_OUTPUTS(outputs),
    FILTER_INPUTS(inputs),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC | AVFILTER_FLAG_SLICE_THREADS,
    .process_command = ff_filter_process_command,
};
