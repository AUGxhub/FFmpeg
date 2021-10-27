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

/**
 * @file
 * Calculate the WPSNR between two input videos.
 */

#include "libavutil/avstring.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "drawutils.h"
#include "formats.h"
#include "framesync.h"
#include "internal.h"
#include "video.h"

#define WIDTH 3840.
#define HEIGHT 2160.

typedef struct WPSNRContext {
    const AVClass *class;
    FFFrameSync fs;
    double wmse, min_wmse, max_wmse, wmse_comp[4];
    uint64_t nb_frames;
    int mode;
    int sizeN;
    int sizeM;
    double apic;
    double amin;
    int max[4], average_max;
    char comps[4];
    int nb_components;
    int nb_threads;
    int planewidth[4];
    int planeheight[4];
    double planeweight[4];
    double **score;

    uint16_t *tmp;
    uint64_t *sat;
    double   *weights;

    void (*compute_hx)(const uint8_t *ssrc,
                       int linesize,
                       int w, int h,
                       uint16_t *dstp,
                       int dst_linesize);

    void (*compute_sat)(const uint16_t *ssrc,
                        int linesize,
                        int w, int h,
                        uint64_t *dstp,
                        int dst_linesize);

    int (*compute_weights)(AVFilterContext *ctx, void *arg,
                           int jobnr, int nb_jobs);

    int (*compute_wmse)(AVFilterContext *ctx, void *arg,
                        int jobnr, int nb_jobs);
} WPSNRContext;

#define OFFSET(x) offsetof(WPSNRContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption wpsnr_options[] = {
    { "mode", "set the mode", OFFSET(mode), AV_OPT_TYPE_INT, {.i64=0}, 0, 1, FLAGS, "mode" },
    { "block",  "block based",  0, AV_OPT_TYPE_CONST, {.i64=0}, 0, 1, FLAGS, "mode" },
    { "sample", "sample based", 0, AV_OPT_TYPE_CONST, {.i64=1}, 0, 1, FLAGS, "mode" },
    { NULL }
};

FRAMESYNC_DEFINE_CLASS(wpsnr, WPSNRContext, fs);

#define COMPUTE_HX(type, stype, depth)               \
static void compute_hx##depth(const uint8_t *ssrc,   \
                              int linesize,          \
                              int w, int h,          \
                              uint16_t *dstp,        \
                              int dst_linesize)      \
{                                                    \
    const type *src = (const type *)ssrc;            \
    stype *dst = (stype *)dstp;                      \
                                                     \
    linesize /= (depth / 8);                         \
                                                     \
    src += linesize;                                 \
    dst += dst_linesize;                             \
    for (int y = 1; y < h - 1; y++) {                \
        for (int x = 1; x < w - 1; x++) {            \
            int v = 12 * src[x] -                    \
                    2 * (src[x-1] + src[x+1] +       \
                         src[x + linesize] +         \
                         src[x - linesize]) -        \
                    1 * (src[x - 1 - linesize] +     \
                         src[x + 1 - linesize] +     \
                         src[x - 1 + linesize] +     \
                         src[x + 1 + linesize]);     \
            dst[x] = FFABS(v);                       \
        }                                            \
                                                     \
        src += linesize;                             \
        dst += dst_linesize;                         \
    }                                                \
}

COMPUTE_HX(uint8_t,  uint16_t, 8)
COMPUTE_HX(uint16_t, uint16_t, 16)

#define COMPUTE_SAT(type, stype, depth)              \
static void compute_sat##depth(const uint16_t *ssrc, \
                               int linesize,         \
                               int w, int h,         \
                               uint64_t *dstp,       \
                               int dst_linesize)     \
{                                                    \
    const type *src = (const type *)ssrc;            \
    stype *dst = (stype *)dstp;                      \
                                                     \
    dst += dst_linesize;                             \
                                                     \
    for (int y = 0; y < h; y++) {                    \
        stype sum = 0;                               \
                                                     \
        for (int x = 1; x < w; x++) {                \
            sum += src[x - 1];                       \
            dst[x] = sum + dst[x - dst_linesize];    \
        }                                            \
                                                     \
        src += linesize;                             \
        dst += dst_linesize;                         \
    }                                                \
}

COMPUTE_SAT(uint16_t, uint64_t, 16)

static inline unsigned pow_2(int base)
{
    return base*base;
}

static inline double get_wpsnr(double wmse, uint64_t nb_frames, int max)
{
    return 10.0 * log10(pow_2(max) / (wmse / nb_frames));
}

typedef struct ThreadData {
    const uint8_t *main_data[4];
    const uint8_t *ref_data[4];
    int main_linesize[4];
    int ref_linesize[4];
    int planewidth[4];
    int planeheight[4];
    double **score;
    int nb_components;
} ThreadData;

static double get_hxs(const uint64_t *src,
                      int x, int y,
                      int linesize,
                      int t, int b,
                      int l, int r)
{
    const uint64_t tl = src[(y - t) * linesize + x - l];
    const uint64_t tr = src[(y - t) * linesize + x + r];
    const uint64_t bl = src[(y + b) * linesize + x - l];
    const uint64_t br = src[(y + b) * linesize + x + r];
    const uint64_t sum = br + tl - bl - tr;

    return fabs(sum * 0.25);
}

static double get_hx(const uint8_t *src, int linesize, int w, int h)
{
    int64_t sum = 0;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            sum += 12 * src[x] -
                    2 * (src[x-1] + src[x+1] +
                         src[x + linesize] +
                         src[x - linesize]) -
                    1 * (src[x - 1 - linesize] +
                         src[x + 1 - linesize] +
                         src[x - 1 + linesize] +
                         src[x + 1 + linesize]);
        }

        src += linesize;
    }

    return fabs(sum * 0.25);
}

static double get_hx16(const uint8_t *ssrc, int linesize, int w, int h)
{
    const uint16_t *src = (const uint16_t *)ssrc;
    int64_t sum = 0;

    linesize /= 2;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            sum += 12 * src[x] -
                    2 * (src[x-1] + src[x+1] +
                         src[x + linesize] +
                         src[x - linesize]) -
                    1 * (src[x - 1 - linesize] +
                         src[x + 1 - linesize] +
                         src[x - 1 + linesize] +
                         src[x + 1 + linesize]);
        }

        src += linesize;
    }

    return fabs(sum * 0.25);
}

static double get_sd(const uint8_t *ref, int ref_linesize,
                     const uint8_t *main, int main_linesize,
                     int w, int h)
{
    int64_t sum = 0;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++)
            sum += pow_2(ref[x] - main[x]);
        ref += ref_linesize;
        main += main_linesize;
    }

    return sum;
}

static double get_sd16(const uint8_t *rref, int ref_linesize,
                       const uint8_t *mmain, int main_linesize,
                       int w, int h)
{
    const uint16_t *ref = (const uint16_t *)rref;
    const uint16_t *main = (const uint16_t *)mmain;
    int64_t sum = 0;

    ref_linesize /= 2;
    main_linesize /= 2;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++)
            sum += pow_2(ref[x] - main[x]);
        ref += ref_linesize;
        main += main_linesize;
    }

    return sum;
}

static
int compute_block_wmse16(AVFilterContext *ctx, void *arg,
                         int jobnr, int nb_jobs)
{
    WPSNRContext *s = ctx->priv;
    ThreadData *td = arg;
    double *score = td->score[jobnr];
    const int sizeN = s->sizeN;

    for (int c = 0; c < s->nb_components; c++) {
        const int planew = (td->planewidth[0] - 2) / sizeN;
        const int blockw = (sizeN * td->planewidth[c]) / td->planewidth[0];
        const int blockh = (sizeN * td->planeheight[c]) / td->planeheight[0];
        const int slice_start = ((td->planeheight[c] / blockh) * jobnr) / nb_jobs;
        const int slice_end = ((td->planeheight[c] / blockh) * (jobnr+1)) / nb_jobs;
        const int ref_linesize = td->ref_linesize[c];
        const int main_linesize = td->main_linesize[c];
        const uint8_t *main_line = td->main_data[c] + main_linesize * slice_start * blockw;
        const uint8_t *ref_line = td->ref_data[c] + ref_linesize * slice_start * blockw;
        const double *weights = s->weights + slice_start * planew;
        double m = 0;

        for (int i = slice_start; i < slice_end; i++) {
            for (int j = 0; j < planew; j++) {
                m += weights[j] * get_sd16(ref_line + 2 * j * blockw, ref_linesize,
                                           main_line + 2 * j * blockw, main_linesize,
                                           blockw, blockh);
            }
            weights += planew;
            ref_line += ref_linesize * blockw;
            main_line += main_linesize * blockw;
        }
        score[c] = m;
    }

    return 0;
}

static
int compute_sample_wmse16(AVFilterContext *ctx, void *arg,
                          int jobnr, int nb_jobs)
{
    WPSNRContext *s = ctx->priv;
    ThreadData *td = arg;
    double *score = td->score[jobnr];
    for (int c = 0; c < s->nb_components; c++) {
        const int wstepw = td->planewidth[0] / td->planewidth[c];
        const int wsteph = td->planeheight[0] / td->planeheight[c];
        const int blockw = td->planewidth[c];
        const int blockh = td->planeheight[c];
        const int slice_start = (blockh * jobnr) / nb_jobs;
        const int slice_end = (blockh * (jobnr+1)) / nb_jobs;
        const int ref_linesize = td->ref_linesize[c];
        const int main_linesize = td->main_linesize[c];
        const uint8_t *main_line = td->main_data[c] + main_linesize * slice_start;
        const uint8_t *ref_line = td->ref_data[c] + ref_linesize * slice_start;
        const double *weights = s->weights + slice_start * blockw;
        double m = 0.;

        for (int i = slice_start; i < slice_end; i++) {
            for (int j = 0; j < blockw; j++)
                m += weights[j * wstepw] * pow_2(AV_RN16(ref_line + 2*j) - AV_RN16(main_line + 2*j));
            weights += blockw * wsteph;
            ref_line += ref_linesize;
            main_line += main_linesize;
        }
        score[c] = m;
    }

    return 0;
}

static
int compute_block_weights8(AVFilterContext *ctx, void *arg,
                           int jobnr, int nb_jobs)
{
    WPSNRContext *s = ctx->priv;
    const int sizeN = s->sizeN;
    const double amin = s->amin;
    const double sizeN2 = sizeN * sizeN;
    const double wnum = sqrt(s->apic) * sizeN2;
    ThreadData *td = arg;
    const int blockw = (td->planewidth[0] - 2) / sizeN;
    const int blockh = (td->planeheight[0] - 2) / sizeN;
    const int slice_start = (blockh * jobnr) / nb_jobs;
    const int slice_end = (blockh * (jobnr+1)) / nb_jobs;
    const int ref_linesize = td->ref_linesize[0];
    const int main_linesize = td->main_linesize[0];
    const uint8_t *main_line = td->main_data[0] + main_linesize * slice_start * sizeN + 1 + main_linesize;
    const uint8_t *ref_line = td->ref_data[0] + ref_linesize * slice_start * sizeN + 1 + ref_linesize;
    double *weights = s->weights + slice_start * blockw;

    for (int i = slice_start; i < slice_end; i++) {
        for (int j = 0; j < blockw; j++) {
            const double ak = fmax(sizeN2 * amin, get_hx(ref_line + j * sizeN, ref_linesize, sizeN, sizeN));
            const double wk = wnum / ak;

            weights[j] = wk;
        }
        weights += blockw;
        ref_line += ref_linesize * sizeN;
        main_line += main_linesize * sizeN;
    }

    return 0;
}

static
int compute_sample_weights8(AVFilterContext *ctx, void *arg,
                            int jobnr, int nb_jobs)
{
    WPSNRContext *s = ctx->priv;
    const int sizeM = s->sizeM;
    const double apic = s->apic;
    const double sizeM2 = sizeM * sizeM;
    const double wnum = sqrt(apic) * sizeM2;
    ThreadData *td = arg;
    const uint64_t *sat = s->sat;
    const int blockw = td->planewidth[0];
    const int blockh = td->planeheight[0];
    const int slice_start = (blockh * jobnr) / nb_jobs;
    const int slice_end = (blockh * (jobnr+1)) / nb_jobs;
    const int ref_linesize = td->ref_linesize[0];
    const int main_linesize = td->main_linesize[0];
    const uint8_t *main_line = td->main_data[0] + main_linesize * slice_start;
    const uint8_t *ref_line = td->ref_data[0] + ref_linesize * slice_start;
    double *weights = s->weights + slice_start * blockw;

    for (int i = slice_start; i < slice_end; i++) {
        const int top = FFMAX(i - sizeM, 0);
        const int bottom = FFMIN(i + sizeM + 1, blockh - i);
        for (int j = 0; j < blockw; j++) {
            const int left = FFMAX(j - sizeM, 0);
            const int right = FFMIN(j + sizeM + 1, blockw - j);
            const double ak = fmax(0.25, get_hxs(sat, j, i,
                                                 blockw + 1,
                                                 top, bottom,
                                                 left, right));
            weights[j] = wnum / ak;
        }
        weights += blockw;
        ref_line += ref_linesize;
        main_line += main_linesize;
    }

    return 0;
}

static
int compute_sample_weights16(AVFilterContext *ctx, void *arg,
                             int jobnr, int nb_jobs)
{
    WPSNRContext *s = ctx->priv;
    const int sizeM = s->sizeM;
    const double sizeM2 = sizeM * sizeM;
    const double wnum = sqrt(s->apic) * sizeM2;
    ThreadData *td = arg;
    const uint64_t *sat = s->sat;
    const int blockw = td->planewidth[0];
    const int blockh = td->planeheight[0];
    const int slice_start = (blockh * jobnr) / nb_jobs;
    const int slice_end = (blockh * (jobnr+1)) / nb_jobs;
    const int ref_linesize = td->ref_linesize[0];
    const int main_linesize = td->main_linesize[0];
    const uint8_t *main_line = td->main_data[0] + main_linesize * slice_start;
    const uint8_t *ref_line = td->ref_data[0] + ref_linesize * slice_start;
    double *weights = s->weights + slice_start * blockw;

    for (int i = slice_start; i < slice_end; i++) {
        const int top = FFMAX(i - sizeM, 0);
        const int bottom = FFMIN(i + sizeM + 1, blockh - i);
        for (int j = 0; j < blockw; j++) {
            const int left = FFMAX(j - sizeM, 0);
            const int right = FFMIN(j + sizeM + 1, blockw - j);
            const double ak = fmax(0.25, get_hxs(sat, j, i,
                                                 blockw + 1,
                                                 top, bottom,
                                                 left, right));
            weights[j] = wnum / ak;
        }
        weights += blockw;
        ref_line += ref_linesize;
        main_line += main_linesize;
    }

    return 0;
}

static
int compute_block_weights16(AVFilterContext *ctx, void *arg,
                            int jobnr, int nb_jobs)
{
    WPSNRContext *s = ctx->priv;
    const int sizeN = s->sizeN;
    const double amin = s->amin;
    const double sizeN2 = sizeN * sizeN;
    const double wnum = sqrt(s->apic) * sizeN2;
    ThreadData *td = arg;
    const int blockw = (td->planewidth[0] - 2) / sizeN;
    const int blockh = (td->planeheight[0] - 2) / sizeN;
    const int slice_start = (blockh * jobnr) / nb_jobs;
    const int slice_end = (blockh * (jobnr+1)) / nb_jobs;
    const int ref_linesize = td->ref_linesize[0];
    const int main_linesize = td->main_linesize[0];
    const uint8_t *main_line = td->main_data[0] + main_linesize * slice_start * sizeN + 1 + main_linesize;
    const uint8_t *ref_line = td->ref_data[0] + ref_linesize * slice_start * sizeN + 1 + ref_linesize;
    double *weights = s->weights + slice_start * blockw;

    for (int i = slice_start; i < slice_end; i++) {
        for (int j = 0; j < blockw; j++) {
            const double ak = fmax(sizeN2 * amin, get_hx16(ref_line + 2 * j * sizeN, ref_linesize, sizeN, sizeN));
            const double wk = wnum / ak;

            weights[j] = wk;
        }
        weights += blockw;
        ref_line += ref_linesize * sizeN;
        main_line += main_linesize * sizeN;
    }

    return 0;
}

static
int compute_block_wmse8(AVFilterContext *ctx, void *arg,
                        int jobnr, int nb_jobs)
{
    WPSNRContext *s = ctx->priv;
    ThreadData *td = arg;
    double *score = td->score[jobnr];
    const int sizeN = s->sizeN;

    for (int c = 0; c < s->nb_components; c++) {
        const int planew = (td->planewidth[0] - 2) / sizeN;
        const int blockw = (sizeN * td->planewidth[c]) / td->planewidth[0];
        const int blockh = (sizeN * td->planeheight[c]) / td->planeheight[0];
        const int slice_start = ((td->planeheight[c] / blockh) * jobnr) / nb_jobs;
        const int slice_end = ((td->planeheight[c] / blockh) * (jobnr+1)) / nb_jobs;
        const int ref_linesize = td->ref_linesize[c];
        const int main_linesize = td->main_linesize[c];
        const uint8_t *main_line = td->main_data[c] + main_linesize * slice_start * blockw;
        const uint8_t *ref_line = td->ref_data[c] + ref_linesize * slice_start * blockw;
        const double *weights = s->weights + slice_start * planew;
        double m = 0;

        for (int i = slice_start; i < slice_end; i++) {
            for (int j = 0; j < planew; j++) {
                m += weights[j] * get_sd(ref_line + j * blockw, ref_linesize,
                                         main_line + j * blockw, main_linesize,
                                         blockw, blockh);
            }
            weights += planew;
            ref_line += ref_linesize * blockw;
            main_line += main_linesize * blockw;
        }
        score[c] = m;
    }

    return 0;
}

static
int compute_sample_wmse8(AVFilterContext *ctx, void *arg,
                         int jobnr, int nb_jobs)
{
    WPSNRContext *s = ctx->priv;
    ThreadData *td = arg;
    double *score = td->score[jobnr];

    for (int c = 0; c < s->nb_components; c++) {
        const int wstepw = td->planewidth[0] / td->planewidth[c];
        const int wsteph = td->planeheight[0] / td->planeheight[c];
        const int blockw = td->planewidth[c];
        const int blockh = td->planeheight[c];
        const int slice_start = (blockh * jobnr) / nb_jobs;
        const int slice_end = (blockh * (jobnr+1)) / nb_jobs;
        const int ref_linesize = td->ref_linesize[c];
        const int main_linesize = td->main_linesize[c];
        const uint8_t *main_line = td->main_data[c] + main_linesize * slice_start;
        const uint8_t *ref_line = td->ref_data[c] + ref_linesize * slice_start;
        const double *weights = s->weights + slice_start * blockw;
        double m = 0.;

        for (int i = slice_start; i < slice_end; i++) {
            for (int j = 0; j < blockw; j++)
                m += weights[j * wstepw] * pow_2(ref_line[j] - main_line[j]);
            weights += blockw * wsteph;
            ref_line += ref_linesize;
            main_line += main_linesize;
        }
        score[c] = m;
    }

    return 0;
}

static void set_meta(AVDictionary **metadata, const char *key, char comp, float d)
{
    char value[128];
    snprintf(value, sizeof(value), "%f", d);
    if (comp) {
        char key2[128];
        snprintf(key2, sizeof(key2), "%s%c", key, comp);
        av_dict_set(metadata, key2, value, 0);
    } else {
        av_dict_set(metadata, key, value, 0);
    }
}

static const char *const wpsnr_keys[][2] =
{
    {
        "lavfi.wpsnr.block.wmse.",
        "lavfi.wpsnr.sample.wmse.",
    },
    {
        "lavfi.wpsnr.block.wpsnr.",
        "lavfi.wpsnr.sample.wpsnr.",
    },
    {
        "lavfi.wpsnr.block.wmse_avg",
        "lavfi.wpsnr.sample.wmse_avg",
    },
    {
        "lavfi.wpsnr.block.wpsnr_avg",
        "lavfi.wpsnr.sample.wpsnr_avg",
    },
};

static int do_wpsnr(FFFrameSync *fs)
{
    AVFilterContext *ctx = fs->parent;
    WPSNRContext *s = ctx->priv;
    AVFrame *master, *ref;
    double comp_wmse[4] = { 0 }, wmse = 0.;
    uint64_t comp_sum[4] = { 0 };
    AVDictionary **metadata;
    ThreadData td;
    int ret;

    ret = ff_framesync_dualinput_get(fs, &master, &ref);
    if (ret < 0)
        return ret;
    if (ctx->is_disabled || !ref)
        return ff_filter_frame(ctx->outputs[0], master);
    metadata = &master->metadata;

    td.nb_components = s->nb_components;
    td.score = s->score;
    for (int c = 0; c < s->nb_components; c++) {
        td.main_data[c] = master->data[c];
        td.ref_data[c] = ref->data[c];
        td.main_linesize[c] = master->linesize[c];
        td.ref_linesize[c] = ref->linesize[c];
        td.planewidth[c] = s->planewidth[c];
        td.planeheight[c] = s->planeheight[c];
    }

    if (s->mode) {
        s->compute_hx(ref->data[0], ref->linesize[0],
                      s->planewidth[0], s->planeheight[0],
                      s->tmp, s->planewidth[0]);
        s->compute_sat(s->tmp, s->planewidth[0], s->planewidth[0],
                       s->planeheight[0], s->sat, s->planewidth[0]);
        ff_filter_execute(ctx, s->compute_weights, &td, NULL,
                          FFMIN(s->planeheight[0], s->nb_threads));
    } else {
        ff_filter_execute(ctx, s->compute_weights, &td, NULL,
                          FFMIN(s->planeheight[0] / s->sizeN, s->nb_threads));
    }

    ff_filter_execute(ctx, s->compute_wmse, &td, NULL,
                      FFMIN(s->planeheight[1], s->nb_threads));

    for (int j = 0; j < s->nb_threads; j++) {
        for (int c = 0; c < s->nb_components; c++)
            comp_sum[c] += s->score[j][c];
    }

    for (int c = 0; c < s->nb_components; c++)
        comp_wmse[c] = comp_sum[c] / ((double)(s->planewidth[c] - 2) * (s->planeheight[c] - 2));

    for (int c = 0; c < s->nb_components; c++)
        wmse += comp_wmse[c] * s->planeweight[c];

    s->min_wmse = FFMIN(s->min_wmse, wmse);
    s->max_wmse = FFMAX(s->max_wmse, wmse);

    s->wmse += wmse;

    for (int j = 0; j < s->nb_components; j++)
        s->wmse_comp[j] += comp_wmse[j];
    s->nb_frames++;

    for (int j = 0; j < s->nb_components; j++) {
        set_meta(metadata, wpsnr_keys[0][s->mode], s->comps[j], comp_wmse[j]);
        set_meta(metadata, wpsnr_keys[1][s->mode], s->comps[j], get_wpsnr(comp_wmse[j], 1, s->max[j]));
    }
    set_meta(metadata, wpsnr_keys[2][s->mode], 0, wmse);
    set_meta(metadata, wpsnr_keys[3][s->mode], 0, get_wpsnr(wmse, 1, s->average_max));

    return ff_filter_frame(ctx->outputs[0], master);
}

static av_cold int init(AVFilterContext *ctx)
{
    WPSNRContext *s = ctx->priv;

    s->min_wmse = +INFINITY;
    s->max_wmse = -INFINITY;

    s->fs.on_event = do_wpsnr;
    return 0;
}

static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAY9, AV_PIX_FMT_GRAY10, AV_PIX_FMT_GRAY12, AV_PIX_FMT_GRAY14, AV_PIX_FMT_GRAY16,
#define PF_NOALPHA(suf) AV_PIX_FMT_YUV420##suf,  AV_PIX_FMT_YUV422##suf,  AV_PIX_FMT_YUV444##suf
#define PF_ALPHA(suf)   AV_PIX_FMT_YUVA420##suf, AV_PIX_FMT_YUVA422##suf, AV_PIX_FMT_YUVA444##suf
#define PF(suf)         PF_NOALPHA(suf), PF_ALPHA(suf)
    PF(P), PF(P9), PF(P10), PF_NOALPHA(P12), PF_NOALPHA(P14), PF(P16),
    AV_PIX_FMT_YUV440P, AV_PIX_FMT_YUV411P, AV_PIX_FMT_YUV410P,
    AV_PIX_FMT_YUVJ411P, AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUVJ422P,
    AV_PIX_FMT_YUVJ440P, AV_PIX_FMT_YUVJ444P,
    AV_PIX_FMT_NONE
};

static int get_sizeM(int w, int h)
{
    return round(14 * sqrt(w * h / (WIDTH * HEIGHT)));
}

static int get_sizeN(int w, int h)
{
    return ceil(128 * sqrt(w * h / (WIDTH * HEIGHT)));
}

static double get_apic(int w, int h, int depth)
{
    return (1 << depth) * sqrt((WIDTH * HEIGHT) / (w * h));
}

static double get_amin(int depth)
{
    return 1 << (depth - 6);
}

static int config_input_ref(AVFilterLink *inlink)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    AVFilterContext *ctx  = inlink->dst;
    WPSNRContext *s = ctx->priv;
    double average_max;
    unsigned sum;

    s->sizeN = get_sizeN(inlink->w, inlink->h);
    s->sizeM = get_sizeM(inlink->w, inlink->h);
    s->nb_threads = ff_filter_get_nb_threads(ctx);
    s->nb_components = desc->nb_components;
    if (ctx->inputs[0]->w != ctx->inputs[1]->w ||
        ctx->inputs[0]->h != ctx->inputs[1]->h) {
        av_log(ctx, AV_LOG_ERROR, "Width and height of input videos must be same.\n");
        return AVERROR(EINVAL);
    }

    s->max[0] = (1 << desc->comp[0].depth) - 1;
    s->max[1] = (1 << desc->comp[1].depth) - 1;
    s->max[2] = (1 << desc->comp[2].depth) - 1;
    s->max[3] = (1 << desc->comp[3].depth) - 1;

    s->apic = get_apic(inlink->w, inlink->h, desc->comp[0].depth);
    s->amin = get_amin(desc->comp[0].depth);

    s->compute_hx  = desc->comp[0].depth <= 8 ? compute_hx8 : compute_hx16;
    s->compute_sat = compute_sat16;
    s->compute_weights = desc->comp[0].depth <= 8 ? compute_block_weights8 : compute_block_weights16;
    s->compute_wmse = desc->comp[0].depth <= 8 ? compute_block_wmse8 : compute_block_wmse16;
    if (s->mode) {
        s->compute_weights = desc->comp[0].depth <= 8 ? compute_sample_weights8 : compute_sample_weights16;
        s->compute_wmse = desc->comp[0].depth <= 8 ? compute_sample_wmse8 : compute_sample_wmse16;
    }

    s->comps[0] = 'y' ;
    s->comps[1] = 'u' ;
    s->comps[2] = 'v' ;
    s->comps[3] = 'a';

    s->planeheight[1] = s->planeheight[2] = AV_CEIL_RSHIFT(inlink->h, desc->log2_chroma_h);
    s->planeheight[0] = s->planeheight[3] = inlink->h;
    s->planewidth[1]  = s->planewidth[2]  = AV_CEIL_RSHIFT(inlink->w, desc->log2_chroma_w);
    s->planewidth[0]  = s->planewidth[3]  = inlink->w;
    sum = 0;
    for (int j = 0; j < s->nb_components; j++)
        sum += s->planeheight[j] * s->planewidth[j];
    average_max = 0;
    for (int j = 0; j < s->nb_components; j++) {
        s->planeweight[j] = (double) s->planeheight[j] * s->planewidth[j] / sum;
        average_max += s->max[j] * s->planeweight[j];
    }
    s->average_max = lrint(average_max);

    s->score = av_calloc(s->nb_threads, sizeof(*s->score));
    if (!s->score)
        return AVERROR(ENOMEM);

    for (int t = 0; t < s->nb_threads; t++) {
        s->score[t] = av_calloc(s->nb_components, sizeof(*s->score[0]));
        if (!s->score[t])
            return AVERROR(ENOMEM);
    }

    s->tmp = av_calloc(inlink->w * inlink->h, sizeof(s->tmp));
    s->sat = av_calloc((inlink->w + 1) * (inlink->h + 1), sizeof(s->sat));
    s->weights = av_calloc(inlink->w * inlink->h, sizeof(s->weights));
    if (!s->sat || !s->tmp || !s->weights)
        return AVERROR(ENOMEM);

    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    WPSNRContext *s = ctx->priv;
    AVFilterLink *mainlink = ctx->inputs[0];
    int ret;

    ret = ff_framesync_init_dualinput(&s->fs, ctx);
    if (ret < 0)
        return ret;
    outlink->w = mainlink->w;
    outlink->h = mainlink->h;
    outlink->time_base = mainlink->time_base;
    outlink->sample_aspect_ratio = mainlink->sample_aspect_ratio;
    outlink->frame_rate = mainlink->frame_rate;
    if ((ret = ff_framesync_configure(&s->fs)) < 0)
        return ret;

    outlink->time_base = s->fs.time_base;

    if (av_cmp_q(mainlink->time_base, outlink->time_base) ||
        av_cmp_q(ctx->inputs[1]->time_base, outlink->time_base))
        av_log(ctx, AV_LOG_WARNING, "not matching timebases found between first input: %d/%d and second input %d/%d, results may be incorrect!\n",
               mainlink->time_base.num, mainlink->time_base.den,
               ctx->inputs[1]->time_base.num, ctx->inputs[1]->time_base.den);

    return 0;
}

static int activate(AVFilterContext *ctx)
{
    WPSNRContext *s = ctx->priv;
    return ff_framesync_activate(&s->fs);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    WPSNRContext *s = ctx->priv;

    if (s->nb_frames > 0) {
        int j;
        char buf[256];

        buf[0] = 0;
        for (j = 0; j < s->nb_components; j++) {
            av_strlcatf(buf, sizeof(buf), " %c:%f", s->comps[j],
                        get_wpsnr(s->wmse_comp[j], s->nb_frames, s->max[j]));
        }
        av_log(ctx, AV_LOG_INFO, "WPSNR%c%s average:%f min:%f max:%f\n",
               s->mode ? 's' : 'b',
               buf,
               get_wpsnr(s->wmse, s->nb_frames, s->average_max),
               get_wpsnr(s->max_wmse, 1, s->average_max),
               get_wpsnr(s->min_wmse, 1, s->average_max));
    }

    ff_framesync_uninit(&s->fs);
    for (int t = 0; t < s->nb_threads && s->score; t++)
        av_freep(&s->score[t]);
    av_freep(&s->score);

    av_freep(&s->weights);
    av_freep(&s->sat);
    av_freep(&s->tmp);
}

static const AVFilterPad wpsnr_inputs[] = {
    {
        .name         = "main",
        .type         = AVMEDIA_TYPE_VIDEO,
    },{
        .name         = "reference",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input_ref,
    },
};

static const AVFilterPad wpsnr_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_output,
    },
};

const AVFilter ff_vf_wpsnr = {
    .name          = "wpsnr",
    .description   = NULL_IF_CONFIG_SMALL("Calculate the Perceptual Weighted PSNR between two video streams."),
    .preinit       = wpsnr_framesync_preinit,
    .init          = init,
    .uninit        = uninit,
    .activate      = activate,
    .priv_size     = sizeof(WPSNRContext),
    .priv_class    = &wpsnr_class,
    FILTER_INPUTS(wpsnr_inputs),
    FILTER_OUTPUTS(wpsnr_outputs),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_INTERNAL | AVFILTER_FLAG_SLICE_THREADS,
};
