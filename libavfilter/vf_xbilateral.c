/*
 * Copyright (c) 2019 Paul B Mahol
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "video.h"

enum KernelTypes {
    Flat,
    Gaussian,
    AndrewsWave,
    ElFallahFord,
    HubersMiniMax,
    Lorentzian,
    TukeyBiWeight,
    LinearDescent,
    Cosine,
    Inverse
};

typedef struct BilateralContext {
    const AVClass *class;

    int radius;
    int radiusV;
    float sigmaS;
    float sigmaR;
    float centerw_factor;
    int kernS;
    int kernR;
    int planes;

    int diameter;
    int diameterV;
    int nb_threads;
    int nb_planes;
    int depth;
    int max;
    int planewidth[4];
    int planeheight[4];

    AVFrame *weights;
    AVFrame *sums;

    float *spatialw;
    float *diffw;
} BilateralContext;

#define OFFSET(x) offsetof(BilateralContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_RUNTIME_PARAM

static const AVOption xbilateral_options[] = {
    { "radius", "set radius",           OFFSET(radius), AV_OPT_TYPE_INT,   {.i64=1},     1, 255, FLAGS },
    { "radiusV","set vertical radius",  OFFSET(radiusV),AV_OPT_TYPE_INT,   {.i64=0},     0, 255, FLAGS },
    { "sigmaS", "set spatial sigma",    OFFSET(sigmaS), AV_OPT_TYPE_FLOAT, {.dbl=0.1}, 0.0, 512, FLAGS },
    { "sigmaR", "set range sigma",      OFFSET(sigmaR), AV_OPT_TYPE_FLOAT, {.dbl=0.1}, 0.0,   1, FLAGS },
    { "centerwf", "set center pixel weight factor", OFFSET(centerw_factor), AV_OPT_TYPE_FLOAT, {.dbl=1}, 0.0, 10, FLAGS },
    { "planes", "set planes to filter", OFFSET(planes), AV_OPT_TYPE_INT,   {.i64=1},     0, 0xF, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(xbilateral);

static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_YUVA444P, AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV440P,
    AV_PIX_FMT_YUVJ444P, AV_PIX_FMT_YUVJ440P,
    AV_PIX_FMT_YUVA422P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUVA420P, AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUVJ422P, AV_PIX_FMT_YUVJ420P,
    AV_PIX_FMT_YUVJ411P, AV_PIX_FMT_YUV411P, AV_PIX_FMT_YUV410P,
    AV_PIX_FMT_YUV420P9, AV_PIX_FMT_YUV422P9, AV_PIX_FMT_YUV444P9,
    AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV444P10,
    AV_PIX_FMT_YUV420P12, AV_PIX_FMT_YUV422P12, AV_PIX_FMT_YUV444P12, AV_PIX_FMT_YUV440P12,
    AV_PIX_FMT_YUV420P14, AV_PIX_FMT_YUV422P14, AV_PIX_FMT_YUV444P14,
    AV_PIX_FMT_YUV420P16, AV_PIX_FMT_YUV422P16, AV_PIX_FMT_YUV444P16,
    AV_PIX_FMT_YUVA420P9, AV_PIX_FMT_YUVA422P9, AV_PIX_FMT_YUVA444P9,
    AV_PIX_FMT_YUVA420P10, AV_PIX_FMT_YUVA422P10, AV_PIX_FMT_YUVA444P10,
    AV_PIX_FMT_YUVA420P16, AV_PIX_FMT_YUVA422P16, AV_PIX_FMT_YUVA444P16,
    AV_PIX_FMT_GBRP, AV_PIX_FMT_GBRP9, AV_PIX_FMT_GBRP10,
    AV_PIX_FMT_GBRP12, AV_PIX_FMT_GBRP14, AV_PIX_FMT_GBRP16,
    AV_PIX_FMT_GBRAP, AV_PIX_FMT_GBRAP10, AV_PIX_FMT_GBRAP12, AV_PIX_FMT_GBRAP16,
    AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAY9, AV_PIX_FMT_GRAY10, AV_PIX_FMT_GRAY12, AV_PIX_FMT_GRAY14, AV_PIX_FMT_GRAY16,
    AV_PIX_FMT_NONE
};

static double kernel_value(double x, double sigma, int kernel)
{
    switch (kernel) {
    case AndrewsWave:
        if (x <= sigma)
            return ((sin((M_PI * x) / sigma) * sigma) / M_PI);
        return 0.0;
    case ElFallahFord:
        return (1.0 / sqrt(1.0 + ((x * x) / (sigma * sigma))));
    case Gaussian:
        return (exp(-((x * x) / (2.0 * sigma * sigma))));
    case HubersMiniMax:
        if (x <= sigma)
            return (1.0 / sigma);
        return (1.0 / x);
    case Lorentzian:
        return (2.0 / (2.0 * sigma * sigma + x * x));
    case TukeyBiWeight:
        if (x <= sigma)
            return (0.5 * pow((1.0 - ((x * x) / (sigma * sigma))), 2));
        return 0.0;
    case LinearDescent:
        if (x <= sigma)
            return (1.0 - (x / sigma));
        return 0.0;
    case Cosine:
        if (x <= sigma)
            return (cos((M_PI * x) / (2.0 * sigma)));
        return 0.0;
    case Flat:
        if (x <= sigma)
            return (1.0 / sigma);
        return 0.0;
    case Inverse:
        if (x <= sigma) {
            if (x != 0.0)
                return (1.0 / x);
            return 1.0;
        }
        return 0.0;
    }
    return 0.0;
}

static int build_tables(BilateralContext *s)
{
    int diameter = s->diameter;
    int diameterV = s->diameterV;
    int window = diameter * diameterV;
    int radius = s->radius;
    int radiusV = s->radiusV;

    s->spatialw = av_calloc(window, sizeof(*s->spatialw));
    if (!s->spatialw)
        return AVERROR(ENOMEM);
    double *disTable = av_calloc(window, sizeof(*disTable));

    for (int b = 0, y = -radiusV; y <= radiusV; y++) {
        int temp = y * y;
        for (int x = -radius; x <= radius; x++)
            disTable[b++] = sqrt((double)(temp + x * x));
    }

    for (int x = 0; x < window; x++)
        s->spatialw[x] = kernel_value(disTable[x], s->sigmaS, s->kernS);
    s->spatialw[radiusV * diameter + radius] *= s->centerw_factor;

    s->diffw = av_calloc(s->max + 1, sizeof(*s->diffw));
    if (!s->diffw)
        return AVERROR(ENOMEM);

    for (int x = 0; x <= s->max; x++)
        s->diffw[x] = kernel_value(x, s->sigmaR * s->max, s->kernR);

    av_free(disTable);

    return 0;
}

static int config_params(AVFilterContext *ctx)
{
    return 0;
}

typedef struct ThreadData {
    AVFrame *in, *out;
} ThreadData;

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    BilateralContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);

    if (!s->radiusV)
        s->radiusV = s->radius;

    s->diameter = s->radius + s->radius + 1;
    s->diameterV = s->radiusV + s->radiusV + 1;
    s->depth = desc->comp[0].depth;
    s->max = (1 << s->depth) - 1;
    config_params(ctx);

    s->planewidth[1] = s->planewidth[2] = AV_CEIL_RSHIFT(inlink->w, desc->log2_chroma_w);
    s->planewidth[0] = s->planewidth[3] = inlink->w;
    s->planeheight[1] = s->planeheight[2] = AV_CEIL_RSHIFT(inlink->h, desc->log2_chroma_h);
    s->planeheight[0] = s->planeheight[3] = inlink->h;

    s->nb_planes = av_pix_fmt_count_planes(inlink->format);
    s->nb_threads = ff_filter_get_nb_threads(ctx);

    s->weights = ff_get_video_buffer(inlink, inlink->w * 4, inlink->h);
    s->sums = ff_get_video_buffer(inlink, inlink->w * 4, inlink->h);
    if (!s->sums || !s->weights)
        return AVERROR(ENOMEM);

    return build_tables(s);
}

typedef uint8_t PixelType;

static int bilateral_planes(AVFilterContext *ctx, void *arg,
                           int jobnr, int nb_jobs)
{
    BilateralContext *s = ctx->priv;
    ThreadData *td = arg;
    AVFrame *out = td->out;
    AVFrame *in = td->in;

    for (int plane = 0; plane < s->nb_planes; plane++) {
        const ptrdiff_t weights_linesize = s->weights->linesize[plane] / 4;
        const ptrdiff_t sums_linesize = s->sums->linesize[plane] / 4;
        const ptrdiff_t src_linesize = in->linesize[plane];
        const ptrdiff_t dst_linesize = out->linesize[plane];
        const int width = s->planewidth[plane];
        const int height = s->planeheight[plane];
        const int diameter = s->diameter;
        const int radius = s->radius;
        const int radiusV = s->radiusV;
        const float *const spatialw = s->spatialw;
        const float *const diffw = s->diffw;
        const int slice_start = (height * jobnr) / nb_jobs;
        const int slice_end = (height * (jobnr+1)) / nb_jobs;

        if (!(s->planes & (1 << plane))) {
            if (out != in) {
                const uint8_t *src = in->data[plane];
                uint8_t *dst = out->data[plane];

                av_image_copy_plane(dst + slice_start * dst_linesize,
                                    dst_linesize,
                                    src + slice_start * src_linesize,
                                    src_linesize,
                                    width * ((s->depth + 7) / 8),
                                    slice_end - slice_start);
            }
            continue;
        }

        {
            for (int y = slice_start; y < slice_end; y++) {
                float *weights = (float *)s->weights->data[plane] + y * weights_linesize;
                float *sums = (float *)s->sums->data[plane] + y * sums_linesize;

                for (int x = 0; x < width; x++)
                    weights[x] = sums[x] = 0.f;
            }
        }

        for (int offset_y = -radiusV; offset_y <= radiusV; offset_y++) {
            for (int offset_x = -radius; offset_x <= radius; offset_x++) {
                const float spatial_weight = spatialw[(offset_y + radiusV) * diameter + offset_x + radius];
                const PixelType *srcp = (const PixelType *)in->data[plane] + slice_start * src_linesize;
                const PixelType *const src = (const PixelType *)in->data[plane];
                float *weights = (float *)s->weights->data[plane] + slice_start * weights_linesize;
                float *sums = (float *)s->sums->data[plane] + slice_start * sums_linesize;
                const PixelType *tp = srcp;

                for (int y = slice_start; y < slice_end; y++) {
                    const PixelType *const srcpT = src + av_clip(y + offset_y, 0, height - 1) * src_linesize;

                    for (int x = 0; x < radius; x++) {
                        const int cP = tp[x];
                        const int value = srcpT[FFMAX(x + offset_x, 0)];
                        const float weight = spatial_weight * diffw[abs(cP - value)];

                        weights[x] += value * weight;
                        sums[x] += weight;
                    }

                    for (int x = radius; x < width - radius; x++) {
                        const int cP = tp[x];
                        const int value = srcpT[x + offset_x];
                        const float weight = spatial_weight * diffw[abs(cP - value)];

                        weights[x] += value * weight;
                        sums[x] += weight;
                    }

                    for (int x = width - radius; x < width; x++) {
                        const int cP = tp[x];
                        const int value = srcpT[FFMIN(x + offset_x, width - 1)];
                        const float weight = spatial_weight * diffw[abs(cP - value)];

                        weights[x] += value * weight;
                        sums[x] += weight;
                    }

                    srcp += src_linesize;
                    tp   += src_linesize;
                    weights += weights_linesize;
                    sums += sums_linesize;
                }
            }
        }

        {
            for (int y = slice_start; y < slice_end; y++) {
                PixelType *dstp = (PixelType *)out->data[plane] + y * dst_linesize;
                float *weights = (float *)s->weights->data[plane] + y * weights_linesize;
                float *sums = (float *)s->sums->data[plane] + y * sums_linesize;

                for (int x = 0; x < width; x++)
                    dstp[x] = lrintf(weights[x] / sums[x]);
            }
        }
    }

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    BilateralContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    ThreadData td;
    AVFrame *out;

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

    td.in = in;
    td.out = out;
    ff_filter_execute(ctx, bilateral_planes, &td, NULL, s->nb_threads);

    if (out != in)
        av_frame_free(&in);
    return ff_filter_frame(outlink, out);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    BilateralContext *s = ctx->priv;

    av_freep(&s->spatialw);
    av_freep(&s->diffw);
    av_frame_free(&s->weights);
    av_frame_free(&s->sums);
}

static int process_command(AVFilterContext *ctx,
                           const char *cmd,
                           const char *arg,
                           char *res,
                           int res_len,
                           int flags)
{
    int ret = ff_filter_process_command(ctx, cmd, arg, res, res_len, flags);

    if (ret < 0)
        return ret;

    return config_params(ctx);
}

static const AVFilterPad bilateral_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
        .filter_frame = filter_frame,
    },
};

static const AVFilterPad bilateral_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
};

const AVFilter ff_vf_xbilateral = {
    .name          = "xbilateral",
    .description   = NULL_IF_CONFIG_SMALL("Apply Bilateral filter."),
    .priv_size     = sizeof(BilateralContext),
    .priv_class    = &xbilateral_class,
    .uninit        = uninit,
    FILTER_INPUTS(bilateral_inputs),
    FILTER_OUTPUTS(bilateral_outputs),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC |
                     AVFILTER_FLAG_SLICE_THREADS,
    .process_command = process_command,
};
