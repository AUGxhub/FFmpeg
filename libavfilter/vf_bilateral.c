/*
 * Copyright (c) 2017 Ming Yang
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

typedef struct BilateralContext {
    const AVClass *class;

    float sigmaS;
    float sigmaR;
    int planes;

    int nb_planes;
    int depth;
    int planewidth[4];
    int planeheight[4];

    float alpha;
    float range_table[65536];

    int (*input_fun)(AVFilterContext *ctx, void *arg,
                     int jobnr, int nb_jobs);

    int (*output_fun)(AVFilterContext *ctx, void *arg,
                      int jobnr, int nb_jobs);

    int stride[4];

    float *input[4];
    float *output[4];

    float *left_pass[4];
    float *left_pass_factor[4];

    float *right_pass[4];
    float *right_pass_factor[4];

    float *up_pass[4];
    float *up_pass_factor[4];

    float *down_pass[4];
    float *down_pass_factor[4];
} BilateralContext;

#define OFFSET(x) offsetof(BilateralContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_RUNTIME_PARAM

static const AVOption bilateral_options[] = {
    { "sigmaS", "set spatial sigma",    OFFSET(sigmaS), AV_OPT_TYPE_FLOAT, {.dbl=0.1}, 0.0, 512, FLAGS },
    { "sigmaR", "set range sigma",      OFFSET(sigmaR), AV_OPT_TYPE_FLOAT, {.dbl=0.1}, 0.0,   1, FLAGS },
    { "planes", "set planes to filter", OFFSET(planes), AV_OPT_TYPE_INT,   {.i64=1},     0, 0xF, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(bilateral);

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

static int config_params(AVFilterContext *ctx)
{
    BilateralContext *s = ctx->priv;
    float inv_sigma_range;

    inv_sigma_range = 1.0f / (s->sigmaR * ((1 << s->depth) - 1));
    s->alpha = expf(-sqrtf(2.f) / s->sigmaS);

    //compute a lookup table
    for (int i = 0; i < (1 << s->depth); i++)
        s->range_table[i] = s->alpha * expf(-i * inv_sigma_range);

    return 0;
}

typedef struct ThreadData {
    AVFrame *in, *out;
} ThreadData;

static int left_pass(AVFilterContext *ctx, void *arg,
                     int jobnr, int nb_jobs)
{
    BilateralContext *s = ctx->priv;
    const float *range_table = s->range_table;
    const float inv_alpha_f = 1.f - s->alpha;

    for (int plane = 0; plane < s->nb_planes; plane++) {
        const int height = s->planeheight[plane];
        const int width = s->planewidth[plane];
        const int slice_start = (height * jobnr) / nb_jobs;
        const int slice_end = (height * (jobnr+1)) / nb_jobs;
        const int src_linesize = s->stride[plane];
        const float *src = s->input[plane] + slice_start * src_linesize;
        const int left_pass_linesize = s->stride[plane];
        const int left_pass_factor_linesize = s->stride[plane];
        float *left_pass = s->left_pass[plane] + slice_start * left_pass_linesize;
        float *left_pass_factor = s->left_pass_factor[plane] + slice_start * left_pass_factor_linesize;

        if (!(s->planes & (1 << plane)))
            continue;

        for (int y = slice_start; y < slice_end; y++) {
            left_pass_factor[0] = 1.f;
            left_pass[0] = src[0];

            for (int x = 1; x < width; x++) {
                const int diff = fabsf(src[x] - src[x - 1]);
                const float alpha_f = range_table[diff];

                left_pass_factor[x] = inv_alpha_f + alpha_f * left_pass_factor[x - 1];
                left_pass[x] = inv_alpha_f * src[x] + alpha_f * left_pass[x - 1];
            }

            src += src_linesize;
            left_pass += left_pass_linesize;
            left_pass_factor += left_pass_factor_linesize;
        }
    }

    return 0;
}

static int right_pass(AVFilterContext *ctx, void *arg,
                      int jobnr, int nb_jobs)
{
    BilateralContext *s = ctx->priv;
    const float *range_table = s->range_table;
    const float inv_alpha_f = 1.f - s->alpha;

    for (int plane = 0; plane < s->nb_planes; plane++) {
        const int height = s->planeheight[plane];
        const int width = s->planewidth[plane];
        const int slice_start = height - (height * (jobnr + 1)) / nb_jobs;
        const int slice_end = height - 1 - (height * jobnr) / nb_jobs;
        const int src_linesize = s->stride[plane];
        const int right_pass_linesize = s->stride[plane];
        const int right_pass_factor_linesize = s->stride[plane];
        const float *src = s->input[plane] + slice_end * src_linesize;
        float *right_pass = s->right_pass[plane] + slice_end * right_pass_linesize;
        float *right_pass_factor = s->right_pass_factor[plane] + slice_end * right_pass_factor_linesize;

        if (!(s->planes & (1 << plane)))
            continue;

        for (int y = slice_end; y >= slice_start; y--) {
            right_pass_factor[width - 1] = 1.f;
            right_pass[width - 1] = src[width - 1];

            for (int x = width - 2; x >= 0; x--) {
                const int diff = fabsf(src[x] - src[x + 1]);
                const float alpha_f = range_table[diff];

                right_pass_factor[x] = inv_alpha_f + alpha_f * right_pass_factor[x + 1];
                right_pass[x] = inv_alpha_f * src[x] + alpha_f * right_pass[x + 1];
            }

            src -= src_linesize;
            right_pass -= right_pass_linesize;
            right_pass_factor -= right_pass_factor_linesize;
        }
    }

    return 0;
}

static int vertical_pass(AVFilterContext *ctx, void *arg,
                        int jobnr, int nb_jobs)
{
    BilateralContext *s = ctx->priv;

    for (int plane = 0; plane < s->nb_planes; plane++) {
        const int height = s->planeheight[plane];
        const int width = s->planewidth[plane];
        const int slice_start = (height * jobnr) / nb_jobs;
        const int slice_end = (height * (jobnr+1)) / nb_jobs;
        const int left_pass_linesize = s->stride[plane];
        const int left_pass_factor_linesize = s->stride[plane];
        const float *left_pass = ((const float *)s->left_pass[plane]) + slice_start * left_pass_linesize;
        const float *left_pass_factor = ((const float *)s->left_pass_factor[plane]) + slice_start * left_pass_factor_linesize;
        const int right_pass_linesize = s->stride[plane];
        const int right_pass_factor_linesize = s->stride[plane];
        const float *right_pass = ((const float *)s->right_pass[plane]) + slice_start * right_pass_linesize;
        const float *right_pass_factor = ((const float *)s->right_pass_factor[plane]) + slice_start * right_pass_factor_linesize;
        float *dst = (float *)left_pass;

        if (!(s->planes & (1 << plane)))
            continue;

        for (int y = slice_start; y < slice_end; y++) {
            for (int x = 0; x < width; x++) {
                const float factor = 1.f / (left_pass_factor[x] + right_pass_factor[x]);
                dst[x] = factor * (left_pass[x] + right_pass[x]);
            }

            dst += left_pass_linesize;
            left_pass += left_pass_linesize;
            left_pass_factor += left_pass_factor_linesize;
            right_pass += right_pass_linesize;
            right_pass_factor += right_pass_factor_linesize;
        }
    }

    return 0;
}

static int down_pass(AVFilterContext *ctx, void *arg,
                     int jobnr, int nb_jobs)
{
    BilateralContext *s = ctx->priv;
    const float *range_table = s->range_table;
    const float inv_alpha_f = 1.f - s->alpha;

    for (int plane = 0; plane < s->nb_planes; plane++) {
        const int height = s->planeheight[plane];
        const int width = s->planewidth[plane];
        const int slice_start = (width * jobnr) / nb_jobs;
        const int slice_end = (width * (jobnr+1)) / nb_jobs;
        const int src_linesize = s->stride[plane];
        const int src_hor_linesize = s->stride[plane];
        const float *src = (const float *)s->input[plane];
        const float *src_hor = (const float *)s->left_pass[plane];
        const int down_pass_linesize = s->stride[plane];
        const int down_pass_factor_linesize = s->stride[plane];
        float *down_pass = (float *)s->down_pass[plane];
        float *down_pass_factor = (float *)s->down_pass_factor[plane];

        if (!(s->planes & (1 << plane)))
            continue;

        for (int x = slice_start; x < slice_end; x++) {
            down_pass_factor[x] = 1.f;
            down_pass[x] = src_hor[x];
        }

        src += src_linesize;
        src_hor += src_hor_linesize;
        down_pass += down_pass_linesize;
        down_pass_factor += down_pass_factor_linesize;

        for (int y = 1; y < height; y++) {
            for (int x = slice_start; x < slice_end; x++) {
                const int diff = fabsf(src[x] - src[x - src_linesize]);
                const float alpha_f = range_table[diff];

                down_pass_factor[x] = inv_alpha_f + alpha_f * down_pass_factor[x - down_pass_factor_linesize];
                down_pass[x] = inv_alpha_f * src_hor[x] + alpha_f * down_pass[x - down_pass_linesize];
            }

            src += src_linesize;
            src_hor += src_hor_linesize;
            down_pass += down_pass_linesize;
            down_pass_factor += down_pass_factor_linesize;
        }
    }

    return 0;
}

static int up_pass(AVFilterContext *ctx, void *arg,
                   int jobnr, int nb_jobs)
{
    BilateralContext *s = ctx->priv;
    const float *range_table = s->range_table;
    const float inv_alpha_f = 1.f - s->alpha;

    for (int plane = 0; plane < s->nb_planes; plane++) {
        const int height = s->planeheight[plane];
        const int width = s->planewidth[plane];
        const int slice_start = width - (width * (jobnr + 1)) / nb_jobs;
        const int slice_end = width - 1 - (width * jobnr) / nb_jobs;
        const int src_linesize = s->stride[plane];
        const int src_hor_linesize = s->stride[plane];
        const float *src = (const float *)s->input[plane] + (height - 1) * src_linesize;
        const float *src_hor = (const float *)s->left_pass[plane] + (height - 1) * src_hor_linesize;
        const int up_pass_linesize = s->stride[plane];
        const int up_pass_factor_linesize = s->stride[plane];
        float *up_pass = ((float *)s->up_pass[plane]) + (height - 1) * up_pass_linesize;
        float *up_pass_factor = ((float *)s->up_pass_factor[plane]) + (height - 1) * up_pass_factor_linesize;

        if (!(s->planes & (1 << plane)))
            continue;

        for (int x = slice_end; x >= slice_start; x--) {
            up_pass_factor[x] = 1.f;
            up_pass[x] = src_hor[x];
        }

        src -= src_linesize;
        src_hor -= src_hor_linesize;
        up_pass -= up_pass_linesize;
        up_pass_factor -= up_pass_factor_linesize;

        for (int y = 1; y < height; y++) {
            for (int x = slice_end; x >= slice_start; x--) {
                const int diff = fabsf(src[x] - src[x + src_linesize]);
                const float alpha_f = range_table[diff];

                up_pass_factor[x] = inv_alpha_f + alpha_f * up_pass_factor[x + up_pass_factor_linesize];
                up_pass[x] = inv_alpha_f * src_hor[x] + alpha_f * up_pass[x + up_pass_linesize];
            }

            src -= src_linesize;
            src_hor -= src_hor_linesize;
            up_pass -= up_pass_linesize;
            up_pass_factor -= up_pass_factor_linesize;
        }
    }

    return 0;
}

static int average_pass(AVFilterContext *ctx, void *arg,
                        int jobnr, int nb_jobs)
{
    BilateralContext *s = ctx->priv;

    for (int plane = 0; plane < s->nb_planes; plane++) {
        const int height = s->planeheight[plane];
        const int width = s->planewidth[plane];
        const int slice_start = (height * jobnr) / nb_jobs;
        const int slice_end = (height * (jobnr+1)) / nb_jobs;
        const int dst_linesize = s->stride[plane];
        float *dst = (float *)s->output[plane] + slice_start * dst_linesize;
        const int down_pass_linesize = s->stride[plane];
        const int down_pass_factor_linesize = s->stride[plane];
        const int up_pass_linesize = s->stride[plane];
        const int up_pass_factor_linesize = s->stride[plane];
        const float *down_pass = ((const float *)s->down_pass[plane]) + slice_start * down_pass_linesize;
        const float *down_pass_factor = ((const float *)s->down_pass_factor[plane]) + slice_start * down_pass_factor_linesize;
        const float *up_pass = ((const float *)s->up_pass[plane]) + slice_start * up_pass_linesize;
        const float *up_pass_factor = ((const float *)s->up_pass_factor[plane]) + slice_start * up_pass_factor_linesize;

        for (int y = slice_start; y < slice_end; y++) {
            for (int x = 0; x < width; x++)
                dst[x] = (up_pass[x] + down_pass[x]) / (up_pass_factor[x] + down_pass_factor[x]);

            dst += dst_linesize;
            up_pass += up_pass_linesize;
            up_pass_factor += up_pass_factor_linesize;
            down_pass += down_pass_linesize;
            down_pass_factor += down_pass_factor_linesize;
        }
    }

    return 0;
}

static int input_pass(AVFilterContext *ctx, void *arg,
                      int jobnr, int nb_jobs)
{
    BilateralContext *s = ctx->priv;
    ThreadData *td = arg;
    AVFrame *in = td->in;

    for (int plane = 0; plane < s->nb_planes; plane++) {
        const int height = s->planeheight[plane];
        const int width = s->planewidth[plane];
        const int slice_start = (height * jobnr) / nb_jobs;
        const int slice_end = (height * (jobnr+1)) / nb_jobs;
        const int src_linesize = in->linesize[plane];
        const int dst_linesize = s->stride[plane];
        const uint8_t *src = in->data[plane] + slice_start * src_linesize;
        float *dst = (float *)s->input[plane] + slice_start * dst_linesize;

        for (int y = slice_start; y < slice_end; y++) {
            for (int x = 0; x < width; x++)
                dst[x] = src[x];

            dst += dst_linesize;
            src += src_linesize;
        }
    }

    return 0;
}

static int input_pass16(AVFilterContext *ctx, void *arg,
                        int jobnr, int nb_jobs)
{
    BilateralContext *s = ctx->priv;
    ThreadData *td = arg;
    AVFrame *in = td->in;

    for (int plane = 0; plane < s->nb_planes; plane++) {
        const int height = s->planeheight[plane];
        const int width = s->planewidth[plane];
        const int slice_start = (height * jobnr) / nb_jobs;
        const int slice_end = (height * (jobnr+1)) / nb_jobs;
        const int src_linesize = in->linesize[plane] / 2;
        const int dst_linesize = s->stride[plane];
        const uint16_t *src = (const uint16_t *)in->data[plane] + slice_start * src_linesize;
        float *dst = (float *)s->input[plane] + slice_start * dst_linesize;

        for (int y = slice_start; y < slice_end; y++) {
            for (int x = 0; x < width; x++)
                dst[x] = src[x];

            dst += dst_linesize;
            src += src_linesize;
        }
    }

    return 0;
}


static int output_pass(AVFilterContext *ctx, void *arg,
                       int jobnr, int nb_jobs)
{
    BilateralContext *s = ctx->priv;
    ThreadData *td = arg;
    AVFrame *out = td->out;
    AVFrame *in = td->in;

    for (int plane = 0; plane < s->nb_planes; plane++) {
        const int height = s->planeheight[plane];
        const int width = s->planewidth[plane];
        const int slice_start = (height * jobnr) / nb_jobs;
        const int slice_end = (height * (jobnr+1)) / nb_jobs;
        const int dst_linesize = out->linesize[plane];
        const int orig_linesize = in->linesize[plane];
        const int src_linesize = s->stride[plane];
        const float *src = (const float *)s->output[plane] + slice_start * dst_linesize;
        const uint8_t *orig = in->data[plane] + slice_start * orig_linesize;
        uint8_t *dst = out->data[plane] + slice_start * src_linesize;

         if (!(s->planes & (1 << plane))) {
             if (in != out)
                 av_image_copy_plane(dst, dst_linesize, orig, orig_linesize,
                                     width * ((s->depth + 7) / 8), slice_end - slice_start);
             continue;
        }

        for (int y = slice_start; y < slice_end; y++) {
            for (int x = 0; x < width; x++)
                dst[x] = av_clip_uint8(lrintf(src[x]));

            dst += dst_linesize;
            src += src_linesize;
        }
    }

    return 0;
}

static int output_pass16(AVFilterContext *ctx, void *arg,
                         int jobnr, int nb_jobs)
{
    BilateralContext *s = ctx->priv;
    const int depth = s->depth;
    ThreadData *td = arg;
    AVFrame *out = td->out;
    AVFrame *in = td->in;

    for (int plane = 0; plane < s->nb_planes; plane++) {
        const int height = s->planeheight[plane];
        const int width = s->planewidth[plane];
        const int slice_start = (height * jobnr) / nb_jobs;
        const int slice_end = (height * (jobnr+1)) / nb_jobs;
        const int dst_linesize = out->linesize[plane] / 2;
        const int orig_linesize = in->linesize[plane];
        const int src_linesize = s->stride[plane];
        const float *src = (const float *)s->output[plane] + slice_start * dst_linesize;
        const uint8_t *orig = in->data[plane] + slice_start * orig_linesize;
        uint16_t *dst = (uint16_t *)out->data[plane] + slice_start * src_linesize;

         if (!(s->planes & (1 << plane))) {
             if (in != out)
                 av_image_copy_plane((uint8_t *)dst, dst_linesize, orig, orig_linesize,
                                     width * ((s->depth + 7) / 8), slice_end - slice_start);
             continue;
        }

        for (int y = slice_start; y < slice_end; y++) {
            for (int x = 0; x < width; x++)
                dst[x] = av_clip_uintp2_c(lrintf(src[x]), depth);

            dst += dst_linesize;
            src += src_linesize;
        }
    }

    return 0;
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    BilateralContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);

    s->depth = desc->comp[0].depth;

    config_params(ctx);

    s->planewidth[1] = s->planewidth[2] = AV_CEIL_RSHIFT(inlink->w, desc->log2_chroma_w);
    s->planewidth[0] = s->planewidth[3] = inlink->w;
    s->planeheight[1] = s->planeheight[2] = AV_CEIL_RSHIFT(inlink->h, desc->log2_chroma_h);
    s->planeheight[0] = s->planeheight[3] = inlink->h;

    s->nb_planes = av_pix_fmt_count_planes(inlink->format);

    s->input_fun  = s->depth <= 8 ? input_pass  : input_pass16;
    s->output_fun = s->depth <= 8 ? output_pass : output_pass16;

    for (int p = 0; p < s->nb_planes; p++) {
        int stride;

        s->stride[p] = stride   = FFALIGN(s->planewidth[p], 16);
        s->input[p]             = av_calloc(stride * s->planeheight[p], sizeof(float));
        s->output[p]            = av_calloc(stride * s->planeheight[p], sizeof(float));
        s->left_pass[p]         = av_calloc(stride * s->planeheight[p], sizeof(float));
        s->left_pass_factor[p]  = av_calloc(stride * s->planeheight[p], sizeof(float));
        s->right_pass[p]        = av_calloc(stride * s->planeheight[p], sizeof(float));
        s->right_pass_factor[p] = av_calloc(stride * s->planeheight[p], sizeof(float));
        s->up_pass[p]           = av_calloc(stride * s->planeheight[p], sizeof(float));
        s->up_pass_factor[p]    = av_calloc(stride * s->planeheight[p], sizeof(float));
        s->down_pass[p]         = av_calloc(stride * s->planeheight[p], sizeof(float));
        s->down_pass_factor[p]  = av_calloc(stride * s->planeheight[p], sizeof(float));
        if (!s->input[p]             ||
            !s->output[p]            ||
            !s->left_pass[p]         ||
            !s->left_pass_factor[p]  ||
            !s->right_pass[p]        ||
            !s->right_pass_factor[p] ||
            !s->up_pass[p]           ||
            !s->up_pass_factor[p]    ||
            !s->down_pass[p]         ||
            !s->down_pass_factor[p])
            return AVERROR(ENOMEM);
    }

    return 0;
}

static int bilateral_planes(AVFilterContext *ctx,
                            AVFrame *out,
                            AVFrame *in)
{
    BilateralContext *s = ctx->priv;
    const int nb_threads = ff_filter_get_nb_threads(ctx);
    ThreadData td;

    td.in = in;
    td.out = out;
    ff_filter_execute(ctx, s->input_fun, &td, NULL,
                      FFMIN(s->planeheight[1], nb_threads));

    ff_filter_execute(ctx, left_pass, NULL, NULL,
                      FFMIN(s->planeheight[1], nb_threads));

    ff_filter_execute(ctx, right_pass, NULL, NULL,
                      FFMIN(s->planeheight[1], nb_threads));

    ff_filter_execute(ctx, vertical_pass, NULL, NULL,
                      FFMIN(s->planeheight[1], nb_threads));

    ff_filter_execute(ctx, down_pass, NULL, NULL,
                      FFMIN(s->planewidth[1], nb_threads));

    ff_filter_execute(ctx, up_pass, NULL, NULL,
                      FFMIN(s->planewidth[1], nb_threads));

    ff_filter_execute(ctx, average_pass, NULL, NULL,
                      FFMIN(s->planeheight[1], nb_threads));

    ff_filter_execute(ctx, s->output_fun, &td, NULL,
                      FFMIN(s->planeheight[1], nb_threads));

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
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

    bilateral_planes(ctx, out, in);

    if (out != in)
        av_frame_free(&in);
    return ff_filter_frame(outlink, out);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    BilateralContext *s = ctx->priv;

    for (int p = 0; p < s->nb_planes; p++) {
        av_freep(&s->input[p]);
        av_freep(&s->output[p]);

        av_freep(&s->left_pass[p]);
        av_freep(&s->left_pass_factor[p]);

        av_freep(&s->right_pass[p]);
        av_freep(&s->right_pass_factor[p]);

        av_freep(&s->up_pass[p]);
        av_freep(&s->up_pass_factor[p]);

        av_freep(&s->down_pass[p]);
        av_freep(&s->down_pass_factor[p]);
    }
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

const AVFilter ff_vf_bilateral = {
    .name          = "bilateral",
    .description   = NULL_IF_CONFIG_SMALL("Apply Bilateral filter."),
    .priv_size     = sizeof(BilateralContext),
    .priv_class    = &bilateral_class,
    .uninit        = uninit,
    FILTER_INPUTS(bilateral_inputs),
    FILTER_OUTPUTS(bilateral_outputs),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC |
                     AVFILTER_FLAG_SLICE_THREADS,
    .process_command = process_command,
};
