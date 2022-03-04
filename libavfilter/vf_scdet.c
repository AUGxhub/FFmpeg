/*
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
 * video scene change detection filter
 */

#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/timestamp.h"

#include "avfilter.h"
#include "filters.h"
#include "scene_sad.h"

typedef struct SCDetScore {
    uint64_t sad[4];
    uint64_t count[4];
} SCDetScore;

typedef struct SCDetContext {
    const AVClass *class;

    int planewidth[4];
    int planeheight[4];
    int nb_threads;
    int nb_planes;
    int planes;
    int bitdepth;
    ff_scene_sad_fn sad;
    double prev_mafd;
    double scene_score;
    AVFrame *prev_picref;
    double threshold;
    int sc_pass;

    SCDetScore *scores;
} SCDetContext;

#define OFFSET(x) offsetof(SCDetContext, x)
#define V AV_OPT_FLAG_VIDEO_PARAM
#define F AV_OPT_FLAG_FILTERING_PARAM

static const AVOption scdet_options[] = {
    { "threshold", "set scene change detect threshold",        OFFSET(threshold), AV_OPT_TYPE_DOUBLE, {.dbl = 10.}, 0,  100., V|F },
    { "t",         "set scene change detect threshold",        OFFSET(threshold), AV_OPT_TYPE_DOUBLE, {.dbl = 10.}, 0,  100., V|F },
    { "sc_pass",   "set the flag to pass scene change frames", OFFSET(sc_pass),   AV_OPT_TYPE_BOOL,   {.dbl =  0},  0,    1,  V|F },
    { "s",         "set the flag to pass scene change frames", OFFSET(sc_pass),   AV_OPT_TYPE_BOOL,   {.dbl =  0},  0,    1,  V|F },
    { "planes",    "set what planes to filter",                OFFSET(planes),    AV_OPT_TYPE_FLAGS,  {.i64 = 1},   0,   15,  V|F },
    { "p",         "set what planes to filter",                OFFSET(planes),    AV_OPT_TYPE_FLAGS,  {.i64 = 1},   0,   15,  V|F },
    {NULL}
};

AVFILTER_DEFINE_CLASS(scdet);

static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_GRAY8,
        AV_PIX_FMT_GBRP, AV_PIX_FMT_GBRP9, AV_PIX_FMT_GBRP10, AV_PIX_FMT_GBRP12,
        AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUVJ420P,
        AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUVJ422P,
        AV_PIX_FMT_YUV440P, AV_PIX_FMT_YUVJ440P,
        AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUVJ444P,
        AV_PIX_FMT_YUV420P9, AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV420P12,
        AV_PIX_FMT_YUV422P9, AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV422P12,
        AV_PIX_FMT_YUV444P9, AV_PIX_FMT_YUV444P10, AV_PIX_FMT_YUV444P12,
        AV_PIX_FMT_NONE
};

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    SCDetContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);

    s->nb_threads = ff_filter_get_nb_threads(ctx);
    s->bitdepth = desc->comp[0].depth;
    s->nb_planes = av_pix_fmt_count_planes(inlink->format);

    s->planeheight[1] = s->planeheight[2] = AV_CEIL_RSHIFT(inlink->h, desc->log2_chroma_h);
    s->planeheight[0] = s->planeheight[3] = inlink->h;
    s->planewidth[1]  = s->planewidth[2]  = AV_CEIL_RSHIFT(inlink->w, desc->log2_chroma_w);
    s->planewidth[0]  = s->planewidth[3]  = inlink->w;

    s->sad = ff_scene_sad_get_fn(s->bitdepth == 8 ? 8 : 16);
    if (!s->sad)
        return AVERROR(EINVAL);

    s->scores = av_calloc(s->nb_threads, sizeof(*s->scores));
    if (!s->scores)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    SCDetContext *s = ctx->priv;

    av_frame_free(&s->prev_picref);
    av_freep(&s->scores);
}

static int compute_sad(AVFilterContext *ctx, void *arg,
                       int jobnr, int nb_jobs)
{
    SCDetContext *s = ctx->priv;
    AVFrame *frame = arg;
    AVFrame *prev_picref = s->prev_picref;

    for (int plane = 0; plane < s->nb_planes; plane++) {
        const int outw = s->planewidth[plane];
        const int outh = s->planeheight[plane];
        const int slice_start = (outh * jobnr) / nb_jobs;
        const int slice_end = (outh * (jobnr+1)) / nb_jobs;
        SCDetScore *score = &s->scores[jobnr];
        uint64_t plane_sad;

        if (!(s->planes & (1 << plane)))
            continue;

        s->sad(prev_picref->data[plane] + slice_start * prev_picref->linesize[plane],
               prev_picref->linesize[plane],
               frame->data[plane] + slice_start * frame->linesize[plane],
               frame->linesize[plane],
               outw, slice_end - slice_start, &plane_sad);

        score->sad[plane] = plane_sad;
        score->count[plane] = outw * (slice_end - slice_start);
    }

    return 0;
}

static double get_scene_score(AVFilterContext *ctx, AVFrame *frame)
{
    double ret = 0;
    SCDetContext *s = ctx->priv;
    AVFrame *prev_picref = s->prev_picref;

    if (prev_picref) {
        uint64_t sad = 0;
        double mafd, diff;
        uint64_t count = 0;

        ff_filter_execute(ctx, compute_sad, frame, NULL,
                          FFMIN(s->planeheight[1], s->nb_threads));

        for (int t = 0; t < s->nb_threads; t++) {
            for (int plane = 0; plane < s->nb_planes; plane++) {
                if (!(s->planes & (1 << plane)))
                    continue;

                sad += s->scores[t].sad[plane];
                count += s->scores[t].count[plane];
            }
        }

        emms_c();
        mafd = (double)sad * 100. / count / (1ULL << s->bitdepth);
        diff = fabs(mafd - s->prev_mafd);
        ret  = av_clipf(FFMIN(mafd, diff), 0, 100.);
        s->prev_mafd = mafd;
        av_frame_free(&prev_picref);
    }
    s->prev_picref = av_frame_clone(frame);
    return ret;
}

static int set_meta(SCDetContext *s, AVFrame *frame, const char *key, const char *value)
{
    return av_dict_set(&frame->metadata, key, value, 0);
}

static int activate(AVFilterContext *ctx)
{
    int ret;
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    SCDetContext *s = ctx->priv;
    AVFrame *frame;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    ret = ff_inlink_consume_frame(inlink, &frame);
    if (ret < 0)
        return ret;

    if (frame) {
        char buf[64];

        if (ctx->is_disabled)
            return ff_filter_frame(outlink, frame);

        s->scene_score = get_scene_score(ctx, frame);
        snprintf(buf, sizeof(buf), "%0.3f", s->prev_mafd);
        set_meta(s, frame, "lavfi.scd.mafd", buf);
        snprintf(buf, sizeof(buf), "%0.3f", s->scene_score);
        set_meta(s, frame, "lavfi.scd.score", buf);

        if (s->scene_score > s->threshold) {
            av_log(s, AV_LOG_INFO, "lavfi.scd.score: %.3f, lavfi.scd.time: %s\n",
                    s->scene_score, av_ts2timestr(frame->pts, &inlink->time_base));
            set_meta(s, frame, "lavfi.scd.time",
                    av_ts2timestr(frame->pts, &inlink->time_base));
        }
        if (s->sc_pass) {
            if (s->scene_score > s->threshold)
                return ff_filter_frame(outlink, frame);
            else {
                av_frame_free(&frame);
            }
        } else
            return ff_filter_frame(outlink, frame);
    }

    FF_FILTER_FORWARD_STATUS(inlink, outlink);
    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return FFERROR_NOT_READY;
}

static const AVFilterPad scdet_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
    },
};

static const AVFilterPad scdet_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
    },
};

const AVFilter ff_vf_scdet = {
    .name          = "scdet",
    .description   = NULL_IF_CONFIG_SMALL("Detect video scene change"),
    .priv_size     = sizeof(SCDetContext),
    .priv_class    = &scdet_class,
    .uninit        = uninit,
    FILTER_INPUTS(scdet_inputs),
    FILTER_OUTPUTS(scdet_outputs),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
    .activate      = activate,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_INTERNAL |
                     AVFILTER_FLAG_SLICE_THREADS             |
                     AVFILTER_FLAG_METADATA_ONLY,
};
