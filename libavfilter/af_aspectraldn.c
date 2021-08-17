/*
 * Copyright (c) 2022 The FFmpeg Project
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

#include "libavutil/avstring.h"
#include "libavutil/channel_layout.h"
#include "libavutil/opt.h"
#include "libavutil/tx.h"
#include "avfilter.h"
#include "audio.h"
#include "formats.h"
#include "filters.h"
#include "window_func.h"

enum OutModes {
    IN_MODE,
    OUT_MODE,
    NOISE_MODE,
    NB_MODES
};

typedef struct AudioSpectralDenoiseContext
{
    const AVClass *class;

    float std_thresh;
    float reduction;
    float overlap;
    float smooth_freq;
    int stationary;
    int fft_factor;
    int fft_size;
    int win_size;
    int win_func;
    int output_mode;
    int sample_advance;

    AVFrame *fft_in;
    AVFrame *fft_out;
    AVFrame *sig_db;
    AVFrame *sig_mask;
    AVFrame *temp;
    AVFrame *winframe;

    AVTXContext **fft, **ifft;
    av_tx_fn tx_fn, itx_fn;

    float *window;
    float smooth_freq_a[3];
    float smooth_freq_b[3];

    int channels;
} AudioSpectralDenoiseContext;

#define OFFSET(x) offsetof(AudioSpectralDenoiseContext, x)
#define AF  AV_OPT_FLAG_AUDIO_PARAM|AV_OPT_FLAG_FILTERING_PARAM
#define AFR AV_OPT_FLAG_AUDIO_PARAM|AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_RUNTIME_PARAM

static const AVOption aspectraldn_options[] = {
    { "stdthr", "set the standard threshold",   OFFSET(std_thresh),  AV_OPT_TYPE_FLOAT,  {.dbl = 0.0},      -1, 30, AFR },
    { "reduction", "set the noise reduction",   OFFSET(reduction),   AV_OPT_TYPE_FLOAT,  {.dbl = 0.9},       0,  1, AFR },
    { "smoothf", "set the freq smooth for mask",OFFSET(smooth_freq), AV_OPT_TYPE_FLOAT,  {.dbl = 0.01},      0,  1, AF  },
    { "stationary", "use stationary reduction", OFFSET(stationary),  AV_OPT_TYPE_BOOL,   {.i64 = 1},         0,  1, AFR },
    { "fft_factor", "set the fft factor",       OFFSET(fft_factor),  AV_OPT_TYPE_INT,    {.i64 = 1},         1,  8, AF },
    { "win_size", "set the window size",        OFFSET(win_size),    AV_OPT_TYPE_INT,    {.i64 = 2048},    128,8192,AF },
    WIN_FUNC_OPTION("win_func",                 OFFSET(win_func), AF, WFUNC_HANNING),
    { "overlap", "set the window overlap",      OFFSET(overlap),     AV_OPT_TYPE_FLOAT,  {.dbl = .75},       0,  1, AF  },
    { "output", "set the output mode",          OFFSET(output_mode), AV_OPT_TYPE_INT,    {.i64 = OUT_MODE},  0,  NB_MODES-1, AFR, "mode" },
    {  "input", "input",                        0,                   AV_OPT_TYPE_CONST,  {.i64 = IN_MODE},   0,  0, AFR, "mode" },
    {  "i", "input",                            0,                   AV_OPT_TYPE_CONST,  {.i64 = IN_MODE},   0,  0, AFR, "mode" },
    {  "output", "output",                      0,                   AV_OPT_TYPE_CONST,  {.i64 = OUT_MODE},  0,  0, AFR, "mode" },
    {  "o", "output",                           0,                   AV_OPT_TYPE_CONST,  {.i64 = OUT_MODE},  0,  0, AFR, "mode" },
    {  "noise", "noise",                        0,                   AV_OPT_TYPE_CONST,  {.i64 = NOISE_MODE},0,  0, AFR, "mode" },
    {  "n", "noise",                            0,                   AV_OPT_TYPE_CONST,  {.i64 = NOISE_MODE},0,  0, AFR, "mode" },
    { NULL }
};

AVFILTER_DEFINE_CLASS(aspectraldn);

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    AudioSpectralDenoiseContext *s = ctx->priv;
    float overlap, w0, alpha;
    int ret;

    s->sample_advance = s->win_size * (1.f - s->overlap);
    s->fft_size = s->win_size * s->fft_factor;
    w0 = M_PI * s->smooth_freq;
    alpha = sinf(w0) * 0.5f;

    s->smooth_freq_a[0] =  1.f + alpha;
    s->smooth_freq_a[1] = -2.f * cosf(w0);
    s->smooth_freq_a[2] =  1.f - alpha;
    s->smooth_freq_b[0] = (1.f - cosf(w0)) / 2.f;
    s->smooth_freq_b[1] =  1.f - cosf(w0);
    s->smooth_freq_b[2] = (1.f - cosf(w0)) / 2.f;
    s->smooth_freq_a[1] /= s->smooth_freq_a[0];
    s->smooth_freq_a[2] /= s->smooth_freq_a[0];
    s->smooth_freq_b[0] /= s->smooth_freq_a[0];
    s->smooth_freq_b[1] /= s->smooth_freq_a[0];
    s->smooth_freq_b[2] /= s->smooth_freq_a[0];

    s->winframe = ff_get_audio_buffer(inlink, s->fft_size * 2);
    s->fft_in = ff_get_audio_buffer(inlink, s->fft_size * 2);
    s->fft_out = ff_get_audio_buffer(inlink, s->fft_size * 2);
    s->temp = ff_get_audio_buffer(inlink, s->fft_size * 2);
    s->sig_db = ff_get_audio_buffer(inlink, s->fft_size);
    s->sig_mask = ff_get_audio_buffer(inlink, s->fft_size);
    s->window = av_calloc(s->win_size, sizeof(*s->window));
    if (!s->fft_in || !s->winframe || !s->fft_out || !s->temp || !s->sig_db ||
        !s->sig_mask || !s->window)
        return AVERROR(ENOMEM);

    generate_window_func(s->window, s->win_size, s->win_func, &overlap);

    s->channels = inlink->channels;
    s->fft  = av_calloc(s->channels, sizeof(*s->fft));
    s->ifft = av_calloc(s->channels, sizeof(*s->ifft));
    if (!s->fft || !s->ifft)
        return AVERROR(ENOMEM);

    for (int ch = 0; ch < s->channels; ch++) {
        float scale = 1.f, iscale = 1.f;

        ret = av_tx_init(&s->fft[ch], &s->tx_fn, AV_TX_FLOAT_FFT, 0, s->fft_size, &scale, 0);
        if (ret < 0)
            return ret;

        ret = av_tx_init(&s->ifft[ch], &s->itx_fn, AV_TX_FLOAT_FFT, 1, s->fft_size, &iscale, 0);
        if (ret < 0)
            return ret;
    }

    return 0;
}

static void get_mask(float *out, const float *in, int size, float db_thresh, float reduction)
{
    const float original = 1.f - reduction;

    for (int n = 0; n < size; n++) {
        const float mask = in[n] > db_thresh;

        out[n] = mask * reduction + original;
    }
}

static void power_to_db(float *S, int size, float ref, float amin, float top_db, float max_mag)
{
    float log_max = FLT_MIN;

    for (int n = 0; n < size; n++) {
        S[n]  = 10.f * log10f(fmaxf(amin, S[n]));
        S[n] -= 10.f * log10f(fmaxf(amin, ref));
        log_max = fmaxf(log_max, S[n]);
    }

    for (int n = 0; n < size; n++)
        S[n] = fmaxf(log_max - top_db, S[n]);
}

static void amplitude_to_db(float *S, AVComplexFloat *in, int size, float ref, float amin, float top_db)
{
    float max_mag = 0.f;

    for (int n = 0; n < size; n++) {
        const float re = in[n].re;
        const float im = in[n].im;

        S[n] = re * re + im * im;
        max_mag = fmaxf(S[n], max_mag);
    }

    power_to_db(S, size, ref, amin, top_db, max_mag);
}

static float get_mean(const float *S, int size)
{
    double sum = 0.0;

    for (int n = 0; n < size; n++)
        sum += S[n];

    return sum / size;
}

static float get_stddev(const float *S, int size, float mean)
{
    double stddev = 0.0;

    for (int n = 0; n < size; n++) {
        const float p = S[n] - mean;

        stddev += p * p;
    }

    stddev = sqrt(stddev / size);

    return stddev;
}

static void smooth_mask(float *m, int size, float *a, float *b)
{
    const float b0 = b[0], b1 = b[1], b2 = b[2];
    const float a1 = -a[1], a2 = -a[2];
    float w1 = 0.f, w2 = 0.f;

    for (int n = 0; n < size; n++) {
        float in = m[n], out;

        out = b0 * in + w1;
        w1 = b1 * in + w2 + a1 * out;
        w2 = b2 * in + a2 * out;
        m[n] = out;
    }

    w1 = 0.f, w2 = 0.f;
    for (int n = size - 1; n >= 0; n--) {
        float in = m[n], out;

        out = b0 * in + w1;
        w1 = b1 * in + w2 + a1 * out;
        w2 = b2 * in + a2 * out;
        m[n] = out;
    }
}

static int filter_channel(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    AudioSpectralDenoiseContext *s = ctx->priv;
    AVFrame *in = arg;
    const int fft_size = s->fft_size;
    const int win_size = s->win_size;
    const float wscale = 1.f / fft_size;
    const float *window = s->window;
    const int start = (in->channels * jobnr) / nb_jobs;
    const int end = (in->channels * (jobnr+1)) / nb_jobs;

    for (int ch = start; ch < end; ch++) {
        float *src = (float *)s->winframe->extended_data[ch];
        float *dst = (float *)s->temp->extended_data[ch];
        AVComplexFloat *fft_out = (AVComplexFloat *)s->fft_out->extended_data[ch];
        AVComplexFloat *fft_in = (AVComplexFloat *)s->fft_in->extended_data[ch];
        float *sig_mask = (float *)s->sig_mask->extended_data[ch];
        float *sig_db = (float *)s->sig_db->extended_data[ch];
        const int offset = win_size - s->sample_advance;
        float mean, stddev, noise_thresh;

        memmove(src, &src[s->sample_advance], offset * sizeof(float));
        memcpy(&src[offset], in->extended_data[ch], s->sample_advance * sizeof(float));

        for (int n = 0; n < win_size; n++) {
            fft_in[n].re = window[n] * src[n];
            fft_in[n].im = 0;
        }

        for (int n = win_size; n < fft_size; n++) {
            fft_in[n].re = 0;
            fft_in[n].im = 0;
        }

        s->tx_fn(s->fft[ch], fft_out, fft_in, sizeof(float));
        for (int n = 0; n < fft_size; n++) {
            fft_out[n].re *= wscale;
            fft_out[n].im *= wscale;
        }

        amplitude_to_db(sig_db, fft_out, fft_size, 1.f, 1e-20f, 80.f);
        mean = get_mean(sig_db, fft_size);
        stddev = get_stddev(sig_db, fft_size, mean);

        noise_thresh = mean + stddev * s->std_thresh;

        get_mask(sig_mask, sig_db, fft_size, noise_thresh, s->reduction);
        smooth_mask(sig_mask, fft_size, s->smooth_freq_a, s->smooth_freq_b);

        for (int n = 0; n < fft_size; n++) {
            fft_out[n].re *= sig_mask[n];
            fft_out[n].im *= sig_mask[n];
        }

        s->itx_fn(s->ifft[ch], fft_in, fft_out, sizeof(float));

        for (int n = 0; n < win_size; n++)
            dst[n] += window[n] * fft_in[n].re;
    }

    return 0;
}

static int output_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    AudioSpectralDenoiseContext *s = ctx->priv;
    const int output_mode = ctx->is_disabled ? IN_MODE : s->output_mode;
    AVFrame *out;

    ff_filter_execute(ctx, filter_channel, in, NULL,
                      FFMIN(outlink->channels, ff_filter_get_nb_threads(ctx)));

    if (av_frame_is_writable(in)) {
        out = in;
    } else {
        out = ff_get_audio_buffer(outlink, in->nb_samples);
        if (!out) {
            av_frame_free(&in);
            return AVERROR(ENOMEM);
        }

        out->pts = in->pts;
    }

    for (int ch = 0; ch < inlink->channels; ch++) {
        float *src = (float *)s->temp->extended_data[ch];
        float *orig = (float *)s->winframe->extended_data[ch];
        float *dst = (float *)out->extended_data[ch];
        const float scale = sqrtf(1.f - s->overlap);

        switch (output_mode) {
        case IN_MODE:
            for (int n = 0; n < out->nb_samples; n++)
                dst[n] = orig[n];
            break;
        case OUT_MODE:
            for (int n = 0; n < out->nb_samples; n++)
                dst[n] = src[n] * scale;
            break;
        case NOISE_MODE:
            for (int n = 0; n < out->nb_samples; n++)
                dst[n] = orig[n] - src[n] * scale;
            break;
        default:
            if (out != in)
                av_frame_free(&out);
            av_frame_free(&in);
            return AVERROR_BUG;
        }

        memmove(src, src + s->sample_advance, (s->fft_size * 2 - s->sample_advance) * sizeof(*src));
    }

    if (out != in)
        av_frame_free(&in);
    return ff_filter_frame(outlink, out);
}

static int activate(AVFilterContext *ctx)
{
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    AudioSpectralDenoiseContext *s = ctx->priv;
    AVFrame *in = NULL;
    int ret;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    ret = ff_inlink_consume_samples(inlink, s->sample_advance, s->sample_advance, &in);
    if (ret < 0)
        return ret;
    if (ret > 0)
        return output_frame(inlink, in);

    FF_FILTER_FORWARD_STATUS(inlink, outlink);
    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return FFERROR_NOT_READY;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    AudioSpectralDenoiseContext *s = ctx->priv;

    av_freep(&s->window);

    av_frame_free(&s->fft_in);
    av_frame_free(&s->fft_out);
    av_frame_free(&s->temp);
    av_frame_free(&s->sig_db);
    av_frame_free(&s->sig_mask);
    av_frame_free(&s->winframe);

    for (int n = 0; n < s->channels; n++) {
        if (s->fft)
            av_tx_uninit(&s->fft[n]);
        if (s->ifft)
            av_tx_uninit(&s->ifft[n]);
    }

    av_freep(&s->fft);
    av_freep(&s->ifft);
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

const AVFilter ff_af_aspectraldn = {
    .name            = "aspectraldn",
    .description     = NULL_IF_CONFIG_SMALL("Denoise audio samples using Spectral Gating."),
    .priv_size       = sizeof(AudioSpectralDenoiseContext),
    .priv_class      = &aspectraldn_class,
    .activate        = activate,
    .uninit          = uninit,
    FILTER_INPUTS(inputs),
    FILTER_OUTPUTS(outputs),
    FILTER_SINGLE_SAMPLEFMT(AV_SAMPLE_FMT_FLTP),
    .flags           = AVFILTER_FLAG_SUPPORT_TIMELINE_INTERNAL |
                       AVFILTER_FLAG_SLICE_THREADS,
};
