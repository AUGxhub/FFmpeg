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

typedef struct SCDetContext {
    const AVClass *class;

    int linesizes[4];
    int planewidth[4];
    int planeheight[4];
    int nb_planes;
    int planes;
    int bitdepth;
    ff_scene_sad_fn sad;
    double prev_mafd;
    double scene_score;
    AVFrame *prev_picref;
    double threshold;
    int sc_pass;
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
        AV_PIX_FMT_RGB24, AV_PIX_FMT_BGR24, AV_PIX_FMT_RGBA,
        AV_PIX_FMT_ABGR, AV_PIX_FMT_BGRA, AV_PIX_FMT_GRAY8,
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
    int ret;

    s->bitdepth = desc->comp[0].depth;
    s->nb_planes = av_pix_fmt_count_planes(inlink->format);
    if ((ret = av_image_fill_linesizes(s->linesizes, inlink->format, inlink->w)) < 0)
        return ret;

    s->planeheight[1] = s->planeheight[2] = AV_CEIL_RSHIFT(inlink->h, desc->log2_chroma_h);
    s->planeheight[0] = s->planeheight[3] = inlink->h;
    s->planewidth[1]  = s->planewidth[2]  = AV_CEIL_RSHIFT(inlink->w, desc->log2_chroma_w);
    s->planewidth[0]  = s->planewidth[3]  = inlink->w;

    s->sad = ff_scene_sad_get_fn(s->bitdepth == 8 ? 8 : 16);
    if (!s->sad)
        return AVERROR(EINVAL);

    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    SCDetContext *s = ctx->priv;

    av_frame_free(&s->prev_picref);
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

        for (int plane = 0; plane < s->nb_planes; plane++) {
            uint64_t plane_sad;

            if (!(s->planes & (1 << plane)))
                continue;

            s->sad(prev_picref->data[plane], prev_picref->linesize[plane],
                    frame->data[plane], frame->linesize[plane],
                    s->planewidth[plane], s->planeheight[plane], &plane_sad);
            sad += plane_sad;
            count += s->planewidth[plane] * s->planeheight[plane];
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
    .flags         = AVFILTER_FLAG_METADATA_ONLY,
    FILTER_INPUTS(scdet_inputs),
    FILTER_OUTPUTS(scdet_outputs),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
    .activate      = activate,
};
