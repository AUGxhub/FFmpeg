/*
 * Copyright (c) 2022 Paul B Mahol
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with FFmpeg; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <float.h>

#include "libavutil/avassert.h"
#include "libavutil/avstring.h"
#include "libavutil/channel_layout.h"
#include "libavutil/eval.h"
#include "libavutil/opt.h"
#include "audio.h"
#include "avfilter.h"
#include "filters.h"
#include "internal.h"

#define MAX_TONES 64

typedef struct AudioToneContext {
    const AVClass *class;

    char *tone_str;

    double frequency[MAX_TONES];
    double amplitude[MAX_TONES];
    double phase[MAX_TONES];
    double fmf[MAX_TONES];
    double fma[MAX_TONES];
    double fmp[MAX_TONES];
    double amf[MAX_TONES];
    double ama[MAX_TONES];
    double amp[MAX_TONES];

    int nb_tones;
    int samples_per_frame;
    int sample_rate;
    int64_t duration;

    int64_t pts;
} AudioToneContext;

#define CONTEXT AudioToneContext
#define FLAGS AV_OPT_FLAG_AUDIO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

#define OPT_GENERIC(name, field, def, min, max, descr, type, deffield, ...) \
    { name, descr, offsetof(CONTEXT, field), AV_OPT_TYPE_ ## type,          \
      { .deffield = def }, min, max, FLAGS, __VA_ARGS__ }

#define OPT_INT(name, field, def, min, max, descr, ...) \
    OPT_GENERIC(name, field, def, min, max, descr, INT, i64, __VA_ARGS__)

#define OPT_DBL(name, field, def, min, max, descr, ...) \
    OPT_GENERIC(name, field, def, min, max, descr, DOUBLE, dbl, __VA_ARGS__)

#define OPT_DUR(name, field, def, min, max, descr, ...) \
    OPT_GENERIC(name, field, def, min, max, descr, DURATION, str, __VA_ARGS__)

#define OPT_STR(name, field, def, min, max, descr, ...) \
    OPT_GENERIC(name, field, def, min, max, descr, STRING, str, __VA_ARGS__)

static const AVOption atone_options[] = {
    OPT_STR("tones",             tone_str,           "440", 0, 0,   "set the tones",),
    OPT_INT("sample_rate",       sample_rate,        44100, 1, INT_MAX,   "set the sample rate",),
    OPT_INT("r",                 sample_rate,        44100, 1, INT_MAX,   "set the sample rate",),
    OPT_DUR("duration",          duration,               0, 0, INT64_MAX, "set the audio duration",),
    OPT_DUR("d",                 duration,               0, 0, INT64_MAX, "set the audio duration",),
    OPT_INT("samples_per_frame", samples_per_frame,   1024,64, 65536,     "set the number of samples per frame",),
    {NULL}
};

AVFILTER_DEFINE_CLASS(atone);

static const char *const var_names[] = {
    "n",
    "pts",
    "t",
    "TB",
    NULL
};

enum {
    VAR_N,
    VAR_PTS,
    VAR_T,
    VAR_TB,
    VAR_VARS_NB
};

static av_cold int query_formats(AVFilterContext *ctx)
{
    AudioToneContext *s = ctx->priv;
    static const AVChannelLayout chlayouts[] = { AV_CHANNEL_LAYOUT_MONO, { 0 } };
    int sample_rates[] = { s->sample_rate, -1 };
    static const enum AVSampleFormat sample_fmts[] = { AV_SAMPLE_FMT_DBLP,
                                                       AV_SAMPLE_FMT_NONE };
    int ret = ff_set_common_formats_from_list(ctx, sample_fmts);
    if (ret < 0)
        return ret;

    ret = ff_set_common_channel_layouts_from_list(ctx, chlayouts);
    if (ret < 0)
        return ret;

    return ff_set_common_samplerates_from_list(ctx, sample_rates);
}

static int parse_tone(AVFilterContext *ctx)
{
    AudioToneContext *s = ctx->priv;
    char *p, *arg, *saveptr = NULL;
    int n = 0;
    int ret;

    p = s->tone_str;

    while (n < MAX_TONES) {
        if (!(arg = av_strtok(p, "|", &saveptr)))
            break;

        p = NULL;
        ret = av_sscanf(arg, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
                        &s->frequency[n],
                        &s->amplitude[n],
                        &s->phase[n],
                        &s->fmf[n],
                        &s->fma[n],
                        &s->fmp[n],
                        &s->amf[n],
                        &s->ama[n],
                        &s->amp[n]);
        if (ret != 9)
            break;

        n++;
    }

    s->nb_tones = n;

    return 0;
}

static av_cold int config_props(AVFilterLink *outlink)
{
    AudioToneContext *s = outlink->src->priv;

    s->duration = av_rescale(s->duration, s->sample_rate, AV_TIME_BASE);

    parse_tone(outlink->src);

    return 0;
}

static int activate(AVFilterContext *ctx)
{
    AVFilterLink *outlink = ctx->outputs[0];
    AudioToneContext *s = ctx->priv;
    int nb_samples = s->samples_per_frame;
    const double tfactor = av_q2d(outlink->time_base);
    double *samples;
    AVFrame *frame;

    if (!ff_outlink_frame_wanted(outlink))
        return FFERROR_NOT_READY;

    if (s->duration) {
        nb_samples = FFMIN(nb_samples, s->duration - s->pts);
        av_assert1(nb_samples >= 0);
        if (!nb_samples) {
            ff_outlink_set_status(outlink, AVERROR_EOF, s->pts);
            return 0;
        }
    }

    if (!(frame = ff_get_audio_buffer(outlink, nb_samples)))
        return AVERROR(ENOMEM);
    samples = (double *)frame->data[0];

    for (int i = 0; i < nb_samples; i++) {
        samples[i] = 0.;

        for (int n = 0; n < s->nb_tones; n++) {
            double amplitude = s->amplitude[n];
            double frequency = s->frequency[n];
            double phase = s->phase[n];
            double fma = s->fma[n];
            double fmf = s->fmf[n];
            double fmp = s->fmp[n];
            double ama = s->ama[n];
            double amf = s->amf[n];
            double amp = s->amp[n];
            double t = (s->pts + i) * tfactor;
            double a = amplitude * (1. + ama * sin(2. * M_PI * amf * t + amp));
            double f = frequency * t + fma * sin(2. * M_PI * fmf * t + fmp);

            samples[i] += a * sin(2. * M_PI * f + phase);
        }
    }

    frame->pts = s->pts;
    s->pts += nb_samples;

    return ff_filter_frame(outlink, frame);
}

static const AVFilterPad atone_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_AUDIO,
        .config_props  = config_props,
    },
};

const AVFilter ff_asrc_atone = {
    .name          = "atone",
    .description   = NULL_IF_CONFIG_SMALL("Generate tone audio signal."),
    .activate      = activate,
    .priv_size     = sizeof(AudioToneContext),
    .inputs        = NULL,
    FILTER_OUTPUTS(atone_outputs),
    FILTER_QUERY_FUNC(query_formats),
    .priv_class    = &atone_class,
};
