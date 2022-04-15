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

#include "libavutil/channel_layout.h"
#include "libavutil/common.h"
#include "libavutil/opt.h"

#include "audio.h"
#include "avfilter.h"
#include "formats.h"
#include "filters.h"
#include "internal.h"

typedef struct ChannelHistory {
    double q, r, a;
} ChannelHistory;

typedef struct AudioBalanceContext {
    int channels;
    int64_t pts;

    ChannelHistory *ch;
    double c1, c2;

    AVFrame *cache[2];
} AudioBalanceContext;

static void balance(AVFilterContext *ctx, const AVFrame *u, const AVFrame *v, AVFrame *o)
{
    AudioBalanceContext *s = ctx->priv;
    const double c1 = s->c1;
    const double c2 = s->c2;

    for (int ch = 0; ch < u->ch_layout.nb_channels; ch++) {
        const double *const us = (const double *)u->extended_data[ch];
        const double *const vs = (const double *)v->extended_data[ch];
        double *os = (double *)o->extended_data[ch];
        ChannelHistory *h = &s->ch[ch];
        double inc, diff, m;
        double q = h->q;
        double r = h->r;
        double a;

        for (int n = 0; n < u->nb_samples; n++) {
            const double as = us[n];
            const double cs = vs[n];

            q = c1 * as * as + c2 * q;
            r = c1 * cs * cs + c2 * r;
        }

        h->q = q;
        h->r = r;

        a = r > 0. ? sqrt(r / q) : sqrt(r);
        diff = a - h->a;
        m = h->a;
        inc = diff / u->nb_samples;

        for (int n = 0; n < u->nb_samples; n++) {
            os[n] = us[n] * m;
            m += inc;
        }

        h->a = a;
    }
}

static int activate(AVFilterContext *ctx)
{
    AudioBalanceContext *s = ctx->priv;
    int ret, status;
    int available;
    int64_t pts;

    FF_FILTER_FORWARD_STATUS_BACK_ALL(ctx->outputs[0], ctx);

    available = FFMIN(ff_inlink_queued_samples(ctx->inputs[0]), ff_inlink_queued_samples(ctx->inputs[1]));
    if (available > 0) {
        AVFrame *out;

        for (int i = 0; i < 2; i++) {
            ret = ff_inlink_consume_samples(ctx->inputs[i], available, available, &s->cache[i]);
            if (ret > 0) {
                if (s->pts == AV_NOPTS_VALUE)
                    s->pts = s->cache[i]->pts;
            }
        }

        out = ff_get_audio_buffer(ctx->outputs[0], available);
        if (out)
            balance(ctx, s->cache[0], s->cache[1], out);

        av_frame_free(&s->cache[0]);
        av_frame_free(&s->cache[1]);

        if (!out)
            return AVERROR(ENOMEM);

        out->pts = s->pts;
        s->pts += available;

        return ff_filter_frame(ctx->outputs[0], out);
    }

    for (int i = 0; i < 2; i++) {
        if (ff_inlink_acknowledge_status(ctx->inputs[i], &status, &pts)) {
            ff_outlink_set_status(ctx->outputs[0], status, s->pts);
            return 0;
        }
    }

    if (ff_outlink_frame_wanted(ctx->outputs[0])) {
        for (int i = 0; i < 2; i++) {
            if (ff_inlink_queued_samples(ctx->inputs[i]) > 0)
                continue;
            ff_inlink_request_frame(ctx->inputs[i]);
        }
        return 0;
    }

    return FFERROR_NOT_READY;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];
    AudioBalanceContext *s = ctx->priv;
    const double w0 = 2. * M_PI * 10. / inlink->sample_rate;
    const double b = 2. - cos(w0);

    s->pts = AV_NOPTS_VALUE;
    s->c2 = b - sqrt(b * b - 1.0);
    s->c1 = 1. - s->c2;

    s->channels = inlink->ch_layout.nb_channels;

    s->ch = av_calloc(s->channels, sizeof(*s->ch));
    if (!s->ch)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    AudioBalanceContext *s = ctx->priv;

    av_frame_free(&s->cache[0]);
    av_frame_free(&s->cache[1]);

    av_freep(&s->ch);
}

static const AVFilterPad inputs[] = {
    {
        .name = "input0",
        .type = AVMEDIA_TYPE_AUDIO,
    },
    {
        .name = "input1",
        .type = AVMEDIA_TYPE_AUDIO,
    },
};

static const AVFilterPad outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_AUDIO,
        .config_props = config_output,
    },
};

const AVFilter ff_af_abalance = {
    .name           = "abalance",
    .description    = NULL_IF_CONFIG_SMALL("Adjust first audio stream amplitude according to the second stream."),
    .priv_size      = sizeof(AudioBalanceContext),
    .activate       = activate,
    .uninit         = uninit,
    FILTER_INPUTS(inputs),
    FILTER_OUTPUTS(outputs),
    FILTER_SINGLE_SAMPLEFMT(AV_SAMPLE_FMT_DBLP),
};
