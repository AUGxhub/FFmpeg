/*
 * Copyright (c) 2022 Paul B Mahol
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
#include "libavutil/channel_layout.h"
#include "libavutil/ffmath.h"
#include "libavutil/opt.h"
#include "avfilter.h"
#include "audio.h"
#include "filters.h"
#include "formats.h"

#define MAX_FRAME_HISTORY 61

#define FF_BUFQUEUE_SIZE MAX_FRAME_HISTORY
#include "libavfilter/bufferqueue.h"

enum var_name {
    VAR_M,
    VAR_S,
    VAR_I,
    VAR_LRA,
    VAR_TP,
    VAR_VARS_NB
};

typedef struct AudioMetaGainContext {
    const AVClass *class;
    double target_i;
    double target_tp;
    double gain;
    double tp;
    double prev_gain;
    int frame_history;

    int eof;
    int64_t pts;
    double var_values[MAX_FRAME_HISTORY][VAR_VARS_NB];
    double weights[MAX_FRAME_HISTORY];

    void (*filter)(void **dst, const void **src,
                   int nb_samples, int channels, struct AudioMetaGainContext *);

    struct FFBufQueue queue;
} AudioMetaGainContext;

#define OFFSET(x) offsetof(AudioMetaGainContext, x)
#define FLAGS AV_OPT_FLAG_AUDIO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

static const AVOption ametagain_options[] = {
    { "i",  "set the target I LUFS",            OFFSET(target_i),      AV_OPT_TYPE_DOUBLE, {.dbl = -25.0}, .min = -99., .max = -2., .flags = FLAGS },
    { "tp", "set the target TP dBFS",           OFFSET(target_tp),     AV_OPT_TYPE_DOUBLE, {.dbl = -2.0},  .min = -20., .max = -1., .flags = FLAGS },
    { "history", "set the frames history size", OFFSET(frame_history), AV_OPT_TYPE_INT, {.i64 = 31}, .min = 9, .max = MAX_FRAME_HISTORY, .flags = FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(ametagain);

static void filter_flt(void **ddst, const void **ssrc,
                       int nb_samples, int channels,
                       struct AudioMetaGainContext *s)
{
    float gain = s->gain;
    const float *src = ssrc[0];
    float *dst = ddst[0];

    for (int n = 0; n < nb_samples; n++) {
        for (int c = 0; c < channels; c++) {
            float d = src[c];

            dst[c] = d * gain;
        }

        dst += channels;
        src += channels;
    }
}

static void filter_dbl(void **ddst, const void **ssrc,
                       int nb_samples, int channels,
                       struct AudioMetaGainContext *s)
{
    const double new_gain = s->gain;
    const double *src = ssrc[0];
    double *dst = ddst[0];
    double pgain;

    for (int n = 0; n < nb_samples; n++) {
        pgain = s->prev_gain;

        for (int c = 0; c < channels; c++) {
            double d = src[c];
            double gain;

            gain = fmin(new_gain, 0.9 * pgain + 0.1 * new_gain);
            pgain = gain;
            dst[c] = d * gain;
        }

        dst += channels;
        src += channels;
    }

    s->prev_gain = pgain;
}

static void filter_fltp(void **ddst, const void **ssrc,
                        int nb_samples, int channels,
                        struct AudioMetaGainContext *s)
{
    float gain = s->gain;

    for (int c = 0; c < channels; c++) {
        const float *src = ssrc[c];
        float *dst = ddst[c];

        for (int n = 0; n < nb_samples; n++) {
            float d = src[n];

            dst[n] = d * gain;
        }
    }
}

static void filter_dblp(void **ddst, const void **ssrc,
                        int nb_samples, int channels,
                        struct AudioMetaGainContext *s)
{
    double gain = s->gain;

    for (int c = 0; c < channels; c++) {
        const double *src = ssrc[c];
        double *dst = ddst[c];

        for (int n = 0; n < nb_samples; n++) {
            double d = src[n];

            dst[n] = d * gain;
        }
    }
}

static void init_gaussian_filter(AudioMetaGainContext *s)
{
    double total_weight = 0.0;
    const double sigma = (((s->frame_history / 2.0) - 1.0) / 3.0) + (1.0 / 3.0);
    double adjust;

    // Pre-compute constants
    const int offset = s->frame_history / 2;
    const double c1 = 1.0 / (sigma * sqrt(2.0 * M_PI));
    const double c2 = 2.0 * sigma * sigma;

    // Compute weights
    for (int i = 0; i < s->frame_history; i++) {
        const int x = i - offset;

        s->weights[i] = c1 * exp(-x * x / c2);
        total_weight += s->weights[i];
    }

    // Adjust weights
    adjust = 1.0 / total_weight;
    for (int i = 0; i < s->frame_history; i++) {
        s->weights[i] *= adjust;
    }
}

static double max_item(AudioMetaGainContext *s, int var)
{
    double result = 0.0;

    for (int i = 0; i < s->frame_history; i++) {
        double item = s->var_values[i][var];

        result = fmax(result, item);
    }

    return result;
}

static double gaussian_filter(AudioMetaGainContext *s, int var)
{
    const double *weights = s->weights;
    double result = 0.0;

    for (int i = 0; i < s->frame_history; i++) {
        double item = s->var_values[i][var];

        result += weights[i] * item;
    }

    return result;
}

#define GET_VALUE_FUNC(name, key, def)               \
static double get_##name(void *priv, int h)          \
{                                                    \
    AudioMetaGainContext *s = priv;                  \
    AVDictionaryEntry *e;                            \
    const int index = FFMIN(s->queue.available-1,h); \
    AVFrame *in = ff_bufqueue_peek(&s->queue, index);\
    float m;                                         \
                                                     \
    e = av_dict_get(in->metadata, key, NULL, 0);     \
    if (e) {                                         \
        if (av_sscanf(e->value, "%f", &m) == 1)      \
            return m;                                \
    }                                                \
                                                     \
    return def;                                      \
}

GET_VALUE_FUNC(m,   "lavfi.r128.M",   0.)
GET_VALUE_FUNC(s,   "lavfi.r128.S",   0.)
GET_VALUE_FUNC(i,   "lavfi.r128.I",   0.)
GET_VALUE_FUNC(lra, "lavfi.r128.LRA", 0.)

static double get_tp(void *priv, int h)
{
    AudioMetaGainContext *s = priv;
    AVDictionaryEntry *e;
    const int index = FFMIN(s->queue.available-1,h);
    AVFrame *in = ff_bufqueue_peek(&s->queue, index);
    float m;

    e = av_dict_get(in->metadata, "lavfi.r128.true_peak", NULL, 0);
    if (!e)
        e = av_dict_get(in->metadata, "lavfi.r128.sample_peak", NULL, 0);
    if (e) {
        if (av_sscanf(e->value, "%f", &m) == 1)
            return m;
    }

    return 1.0;
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    AudioMetaGainContext *s = ctx->priv;

    s->prev_gain = 1.0;

    init_gaussian_filter(s);

    switch (inlink->format) {
    case AV_SAMPLE_FMT_FLT:  s->filter = filter_flt;  break;
    case AV_SAMPLE_FMT_DBL:  s->filter = filter_dbl;  break;
    case AV_SAMPLE_FMT_FLTP: s->filter = filter_fltp; break;
    case AV_SAMPLE_FMT_DBLP: s->filter = filter_dblp; break;
    default: return AVERROR_BUG;
    }

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    AudioMetaGainContext *s = ctx->priv;
    AVFrame *out;

    if (in) {
        ff_bufqueue_add(ctx, &s->queue, in);
        if (s->queue.available) {
            const int index = s->frame_history - 1;

            s->var_values[index][VAR_M]   = get_m(s, 0);
            s->var_values[index][VAR_S]   = get_s(s, 0);
            s->var_values[index][VAR_I]   = ff_exp10((s->target_i - get_i(s, 30)) / 20.);
            s->var_values[index][VAR_LRA] = get_lra(s, 0);
            s->var_values[index][VAR_TP]  = get_tp(s, 0);
        }
    }

    if (s->queue.available < s->frame_history && !s->eof) {
        if (ff_inlink_check_available_frame(inlink) > 0)
            ff_filter_set_ready(ctx, 100);
        return 1;
    }

    in = ff_bufqueue_get(&s->queue);
    if (!in)
        return AVERROR_BUG;

    memmove(s->var_values, s->var_values[1], sizeof(s->var_values[0]) * (s->frame_history - 1));

    if (av_frame_is_writable(in)) {
        out = in;
    } else {
        out = ff_get_audio_buffer(outlink, in->nb_samples);
        if (!out) {
            av_frame_free(&in);
            return AVERROR(ENOMEM);
        }
        av_frame_copy_props(out, in);
    }

    s->tp   = ctx->is_disabled ? 1.0 : ff_exp10(s->target_tp / 20.);
    s->gain = ctx->is_disabled ? 1.0 : FFMIN(s->tp, gaussian_filter(s, VAR_I));

    s->filter((void **)out->extended_data, (const void **)in->extended_data,
              in->nb_samples, in->channels, s);

    if (in != out)
        av_frame_free(&in);

    return ff_filter_frame(outlink, out);
}

static int activate(AVFilterContext *ctx)
{
    AVFilterLink *outlink = ctx->outputs[0];
    AVFilterLink *inlink = ctx->inputs[0];
    AudioMetaGainContext *s = ctx->priv;
    AVFrame *in = NULL;
    int ret = 0, status;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    if (s->eof && s->queue.available) {
        ret = filter_frame(inlink, NULL);
        if (ret < 0)
            return ret;
    } else if (!s->eof) {
        ret = ff_inlink_consume_frame(inlink, &in);
        if (ret < 0)
            return ret;
        if (ret > 0) {
            ret = filter_frame(inlink, in);
            if (ret <= 0)
                return ret;
        }
    }

    if (!s->eof && ff_inlink_acknowledge_status(inlink, &status, &s->pts)) {
        if (status == AVERROR_EOF)
            s->eof = 1;
    }

    if (!s->eof)
        FF_FILTER_FORWARD_WANTED(outlink, inlink);

    if (s->eof) {
        if (!s->queue.available)
            ff_outlink_set_status(outlink, AVERROR_EOF, s->pts);
        else
            ff_filter_set_ready(ctx, 100);
        return 0;
    }

    return FFERROR_NOT_READY;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    AudioMetaGainContext *s = ctx->priv;

    ff_bufqueue_discard_all(&s->queue);
}

static const AVFilterPad inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_AUDIO,
        .config_props = config_input,
    },
};

static const AVFilterPad outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_AUDIO,
    },
};

const AVFilter ff_af_ametagain = {
    .name           = "ametagain",
    .description    = NULL_IF_CONFIG_SMALL("Normalize audio using audio frames metadata."),
    .priv_size      = sizeof(AudioMetaGainContext),
    .priv_class     = &ametagain_class,
    .activate       = activate,
    .uninit         = uninit,
    FILTER_INPUTS(inputs),
    FILTER_OUTPUTS(outputs),
    FILTER_SAMPLEFMTS(AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_FLT,
                      AV_SAMPLE_FMT_DBLP, AV_SAMPLE_FMT_DBL),
    .flags          = AVFILTER_FLAG_SUPPORT_TIMELINE_INTERNAL,
};
