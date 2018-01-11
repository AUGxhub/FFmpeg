/*
 * Copyright (c) 2022 Paul B Mahol
 * Copyright (c) 2017 Sanchit Sinha
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
#include <math.h>
#include <stdio.h>

#include "libavutil/avstring.h"
#include "libavutil/channel_layout.h"
#include "libavutil/float_dsp.h"
#include "libavutil/opt.h"
#include "libavutil/avassert.h"
#include "audio.h"
#include "avfilter.h"
#include "formats.h"
#include "internal.h"

#define EVEN 0
#define ODD 1
#define MAX_ORDER 3
#define SQR(x) ((x) * (x))
#define MAX_CHANNELS SQR(MAX_ORDER + 1)

enum A_NAME {
    A_W, A_Y, A_Z, A_X, A_V, A_T, A_R, A_S, A_U, A_Q, A_O, A_M, A_K, A_L, A_N, A_P,
};

enum NearFieldType {
    NF_AUTO = -1,
    NF_NONE,
    NF_IN,
    NF_OUT,
    NB_NFTYPES,
};

enum PrecisionType {
    P_AUTO = -1,
    P_SINGLE,
    P_DOUBLE,
    NB_PTYPES,
};

enum PTypes {
    PT_AMP,
    PT_RMS,
    PT_ENERGY,
    PT_NBTYPES,
};

enum NormType {
    N3D,
    SN3D,
    FUMA,
    NB_NTYPES,
};

enum DirectionType {
    D_X,
    D_Y,
    D_Z,
    D_C,
    NB_DTYPES,
};

enum SequenceType {
    M_ACN,
    M_FUMA,
    M_SID,
    NB_MTYPES,
};

enum Layouts {
    MONO,
    STEREO,
    STEREO_DOWNMIX,
    SURROUND,
    L2_1,
    TRIANGLE,
    QUAD,
    SQUARE,
    L4_0,
    L5_0,
    L5_0_SIDE,
    L6_0,
    L7_0,
    CUBE,
    NB_LAYOUTS,
};

typedef struct NearField {
    double g;
    double d[MAX_ORDER];
    double z[MAX_ORDER];
} NearField;

typedef struct Xover {
    double b[3];
    double a[3];
    double w[2];
} Xover;

static const double gains_2d[][4] =
{
    { 1 },
    { 1, 0.707107 },
    { 1, 0.866025, 0.5 },
    { 1, 0.92388, 0.707107, 0.382683 },
};

static const double gains_3d[][4] =
{
    { 1 },
    { 1, 0.57735027 },
    { 1, 0.774597, 0.4 },
    { 1, 0.861136, 0.612334, 0.304747 },
};

static const double same_distance[] =
{
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
};

static const double cube_azimuth[] =
{
    315, 45, 135, 225, 315, 45, 135, 225,
};

static const double cube_elevation[] =
{
    45, 45, 45, 45, -45, -45, -45, -45,
};

static const struct {
    const int              order;
    const int              inputs;
    const int              speakers;
    const int              near_field;
    const int              type;
    const double           xover;
    const AVChannelLayout  outlayout;
    const double          *speakers_azimuth;
    const double          *speakers_elevation;
    const double          *speakers_distance;
} ambisonic_tab[] = {
    [MONO] = {
        .order = 0,
        .inputs = 1,
        .speakers = 1,
        .near_field = NF_NONE,
        .xover = 0.,
        .outlayout = (AVChannelLayout)AV_CHANNEL_LAYOUT_MONO,
        .speakers_azimuth = (const double[1]){ 0. },
        .speakers_distance = (const double[1]){ 1. },
    },
    [STEREO] = {
        .order = 1,
        .inputs = 4,
        .speakers = 2,
        .near_field = NF_NONE,
        .xover = 0.,
        .outlayout = (AVChannelLayout)AV_CHANNEL_LAYOUT_STEREO,
        .speakers_azimuth = (const double[2]){ -30, 30},
        .speakers_distance = same_distance,
    },
    [STEREO_DOWNMIX] = {
        .order = 1,
        .inputs = 4,
        .speakers = 2,
        .near_field = NF_NONE,
        .xover = 0.,
        .outlayout = (AVChannelLayout)AV_CHANNEL_LAYOUT_STEREO_DOWNMIX,
        .speakers_azimuth = (const double[2]){ -90, 90 },
        .speakers_distance = same_distance,
    },
    [SURROUND] = {
        .order = 1,
        .inputs = 4,
        .speakers = 3,
        .near_field = NF_NONE,
        .xover = 0.,
        .outlayout = (AVChannelLayout)AV_CHANNEL_LAYOUT_SURROUND,
        .speakers_azimuth = (const double[3]){ -45, 45, 0 },
        .speakers_distance = same_distance,
    },
    [L2_1] = {
        .order = 1,
        .inputs = 4,
        .speakers = 3,
        .near_field = NF_NONE,
        .xover = 0.,
        .outlayout = (AVChannelLayout)AV_CHANNEL_LAYOUT_2_1,
        .speakers_azimuth = (const double[3]){ -45, 45, 180 },
        .speakers_distance = same_distance,
    },
    [TRIANGLE] = {
        .order = 1,
        .inputs = 4,
        .speakers = 3,
        .type = 1,
        .near_field = NF_NONE,
        .xover = 0.,
        .outlayout = (AVChannelLayout)AV_CHANNEL_LAYOUT_SURROUND,
        .speakers_azimuth = (const double[3]){ -120, 120, 0 },
        .speakers_distance = same_distance,
    },
    [QUAD] = {
        .order = 1,
        .inputs = 4,
        .speakers = 4,
        .near_field = NF_NONE,
        .xover = 0.,
        .outlayout = (AVChannelLayout)AV_CHANNEL_LAYOUT_QUAD,
        .speakers_azimuth = (const double[4]){ -45, 45, -135, 135 },
        .speakers_distance = same_distance,
    },
    [SQUARE] = {
        .order = 1,
        .inputs = 4,
        .speakers = 4,
        .type = 1,
        .near_field = NF_NONE,
        .xover = 0.,
        .outlayout = (AVChannelLayout)AV_CHANNEL_LAYOUT_QUAD,
        .speakers_azimuth = (const double[4]){ -90, 90, 0, 180 },
        .speakers_distance = same_distance,
    },
    [L4_0] = {
        .order = 1,
        .inputs = 4,
        .speakers = 4,
        .near_field = NF_NONE,
        .xover = 0.,
        .outlayout = (AVChannelLayout)AV_CHANNEL_LAYOUT_4POINT0,
        .speakers_azimuth = (const double[4]){ -30, 30, 0, 180 },
        .speakers_distance = same_distance,
    },
    [L5_0] = {
        .order = 1,
        .inputs = 4,
        .speakers = 5,
        .near_field = NF_NONE,
        .xover = 0.,
        .outlayout = (AVChannelLayout)AV_CHANNEL_LAYOUT_5POINT0_BACK,
        .speakers_azimuth = (const double[5]){ -30, 30, 0, -145, 145 },
        .speakers_distance = same_distance,
    },
    [L5_0_SIDE] = {
        .order = 1,
        .inputs = 4,
        .speakers = 5,
        .near_field = NF_NONE,
        .xover = 0.,
        .outlayout = (AVChannelLayout)AV_CHANNEL_LAYOUT_5POINT0,
        .speakers_azimuth = (const double[5]){ -30, 30, 0, -110, 110 },
        .speakers_distance = same_distance,
    },
    [L6_0] = {
        .order = 1,
        .inputs = 4,
        .speakers = 6,
        .near_field = NF_NONE,
        .xover = 0.,
        .outlayout = (AVChannelLayout)AV_CHANNEL_LAYOUT_6POINT0,
        .speakers_azimuth = (const double[6]){ -30, 30, 0, 180, -110, 110 },
        .speakers_distance = same_distance,
    },
    [L7_0] = {
        .order = 1,
        .inputs = 4,
        .speakers = 7,
        .near_field = NF_NONE,
        .xover = 0.,
        .outlayout = (AVChannelLayout)AV_CHANNEL_LAYOUT_7POINT0,
        .speakers_azimuth = (const double[7]){ -30, 30, 0, -145, 145, -110, 110 },
        .speakers_distance = same_distance,
    },
    [CUBE] = {
        .order = 1,
        .inputs = 4,
        .speakers = 8,
        .type = 2,
        .near_field = NF_NONE,
        .xover = 0.,
        .outlayout = (AVChannelLayout)AV_CHANNEL_LAYOUT_7POINT1,
        .speakers_azimuth = cube_azimuth,
        .speakers_elevation = cube_elevation,
        .speakers_distance = same_distance,
    },
};

typedef struct AmbisonicContext {
    const AVClass *class;
    int order;                    /* Order of ambisonic */
    int level;                    /* Output Level compensation */
    enum Layouts layout;          /* Output speaker layout */
    enum NormType norm;           /* Normalization Type */
    enum PrecisionType precision; /* Processing Precision Type */
    enum SequenceType seq;        /* Input Channel sequence type */
    enum NearFieldType near_field; /* Near Field compensation type */

    int invert[NB_DTYPES];        /* Axis Odd/Even Invert */
    double gain[2][NB_DTYPES];    /* Axis Odd/Even Gains */
    double pgains[2][MAX_ORDER+1];/* LF/HF perceptual gains */

    double yaw;                   /* Angle for yaw(x) rotation */
    double pitch;                 /* Angle for pitch(y) rotation */
    double roll;                  /* Angle for roll(z) rotation */

    int pgtype;
    int max_channels;             /* Max Channels */
    double matching;

    double temp;
    double xover_freq;
    double xover_ratio;

    Xover xover[2][MAX_CHANNELS];
    NearField nf[2][MAX_CHANNELS];

    int    seq_tab[NB_MTYPES][MAX_CHANNELS];
    double norm_tab[NB_NTYPES][MAX_CHANNELS];
    double rotate_mat[MAX_CHANNELS][MAX_CHANNELS];
    double decode_mat[MAX_CHANNELS][MAX_CHANNELS];
    double u[MAX_CHANNELS][MAX_CHANNELS];
    double v[MAX_CHANNELS][MAX_CHANNELS];
    double w[MAX_CHANNELS];
    double mirror_mat[MAX_CHANNELS];
    double level_tab[MAX_CHANNELS];
    double gains_tab[2][MAX_CHANNELS];
    double dominance[2];

    AVFrame *sframe;
    AVFrame *rframe;
    AVFrame *frame2;

    void (*nf_init[MAX_ORDER])(NearField *nf, double radius,
                               double speed, double rate,
                               double gain);
    void (*nf_process[MAX_ORDER])(NearField *nf,
                                  AVFrame *frame,
                                  int ch, int add,
                                  double gain);
    void (*process)(AVFilterContext *ctx, AVFrame *in, AVFrame *out);

    AVFloatDSPContext *fdsp;
} AmbisonicContext;

#define OFFSET(x) offsetof(AmbisonicContext,x)
#define AF AV_OPT_FLAG_AUDIO_PARAM|AV_OPT_FLAG_FILTERING_PARAM
#define AFT AV_OPT_FLAG_AUDIO_PARAM|AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_RUNTIME_PARAM

static const AVOption ambisonic_options[] = {
    { "layout", "layout of output", OFFSET(layout), AV_OPT_TYPE_INT, {.i64=STEREO}, 0, NB_LAYOUTS-1, AF , "lyt"},
    {   "mono",   "mono layout",   0, AV_OPT_TYPE_CONST, {.i64=MONO},   0, 0, AF , "lyt"},
    {   "stereo", "stereo layout", 0, AV_OPT_TYPE_CONST, {.i64=STEREO}, 0, 0, AF , "lyt"},
    {   "downmix","stereo downmix", 0, AV_OPT_TYPE_CONST, {.i64=STEREO_DOWNMIX}, 0, 0, AF , "lyt"},
    {   "3.0",    "3.0 layout",    0, AV_OPT_TYPE_CONST, {.i64=SURROUND}, 0, 0, AF , "lyt"},
    {   "3.0(back)","3.0(back) layout", 0, AV_OPT_TYPE_CONST, {.i64=L2_1}, 0, 0, AF , "lyt"},
    {   "triangle","triangle layout", 0, AV_OPT_TYPE_CONST, {.i64=TRIANGLE}, 0, 0, AF , "lyt"},
    {   "quad",   "quad layout",   0, AV_OPT_TYPE_CONST, {.i64=QUAD},   0, 0, AF , "lyt"},
    {   "square", "square layout", 0, AV_OPT_TYPE_CONST, {.i64=SQUARE}, 0, 0, AF , "lyt"},
    {   "4.0",    "4.0 layout",    0, AV_OPT_TYPE_CONST, {.i64=L4_0},   0, 0, AF , "lyt"},
    {   "5.0",    "5.0 layout",    0, AV_OPT_TYPE_CONST, {.i64=L5_0},   0, 0, AF , "lyt"},
    {   "5.0(side)", "5.0(side) layout", 0, AV_OPT_TYPE_CONST, {.i64=L5_0_SIDE}, 0, 0, AF , "lyt"},
    {   "6.0",    "6.0 layout",    0, AV_OPT_TYPE_CONST, {.i64=L6_0},   0, 0, AF , "lyt"},
    {   "7.0",    "7.0 layout",    0, AV_OPT_TYPE_CONST, {.i64=L7_0},   0, 0, AF , "lyt"},
    {   "cube",   "cube layout",   0, AV_OPT_TYPE_CONST, {.i64=CUBE},   0, 0, AF , "lyt"},
    { "sequence", "input channel sequence", OFFSET(seq), AV_OPT_TYPE_INT, {.i64=M_ACN},  0, NB_MTYPES-1, AF, "seq"},
    {   "acn",  "ACN",  0, AV_OPT_TYPE_CONST, {.i64=M_ACN},  0, 0, AF, "seq"},
    {   "fuma", "FuMa", 0, AV_OPT_TYPE_CONST, {.i64=M_FUMA}, 0, 0, AF, "seq"},
    {   "sid",  "SID",  0, AV_OPT_TYPE_CONST, {.i64=M_SID},  0, 0, AF, "seq"},
    { "scaling", "input scaling format", OFFSET(norm), AV_OPT_TYPE_INT,   {.i64=SN3D}, 0, NB_NTYPES-1, AF, "scl"},
    {   "n3d",  "N3D scaling (normalised)",       0, AV_OPT_TYPE_CONST, {.i64=N3D},  0, 0, AF, "scl"},
    {   "sn3d", "SN3D scaling (semi-normalised)", 0, AV_OPT_TYPE_CONST, {.i64=SN3D}, 0, 0, AF, "scl"},
    {   "fuma", "furse malham scaling",           0, AV_OPT_TYPE_CONST, {.i64=FUMA}, 0, 0, AF, "scl"},
    { "nearfield", "near-field compenstation", OFFSET(near_field), AV_OPT_TYPE_INT, {.i64=NF_AUTO}, NF_AUTO, NB_NFTYPES-1, AF, "nf"},
    {   "auto", "auto", 0, AV_OPT_TYPE_CONST, {.i64=NF_AUTO}, 0, 0, AF, "nf"},
    {   "none", "none", 0, AV_OPT_TYPE_CONST, {.i64=NF_NONE}, 0, 0, AF, "nf"},
    {   "in",   "in",   0, AV_OPT_TYPE_CONST, {.i64=NF_IN},   0, 0, AF, "nf"},
    {   "out",  "out",  0, AV_OPT_TYPE_CONST, {.i64=NF_OUT},  0, 0, AF, "nf"},
    { "matching", "set matching for decode matrix", OFFSET(matching), AV_OPT_TYPE_DOUBLE, {.dbl=0}, 0., 1., AF, "matching" },
    {   "mode",   "set exact mode matching",  0, AV_OPT_TYPE_CONST, {.dbl=0}, 0, 0, AF, "matching" },
    {   "energy", "set even energy matching", 0, AV_OPT_TYPE_CONST, {.dbl=1}, 0, 0, AF, "matching" },
    { "xoverfreq", "cross-over frequency", OFFSET(xover_freq), AV_OPT_TYPE_DOUBLE, {.dbl=-1.}, -1., 800., AF },
    { "xoverratio", "cross-over HF/LF ratio", OFFSET(xover_ratio), AV_OPT_TYPE_DOUBLE, {.dbl=0.}, -30., 30., AF },
    { "pgtype", "set perceptual LF/HF gains type", OFFSET(pgtype), AV_OPT_TYPE_INT, {.i64=PT_RMS}, 0, PT_NBTYPES-1, AF, "pgt" },
    {   "amplitude", NULL, 0, AV_OPT_TYPE_CONST, {.i64=PT_AMP},    0, 0, AF, "pgt" },
    {   "rms",       NULL, 0, AV_OPT_TYPE_CONST, {.i64=PT_RMS},    0, 0, AF, "pgt" },
    {   "energy",    NULL, 0, AV_OPT_TYPE_CONST, {.i64=PT_ENERGY}, 0, 0, AF, "pgt" },
    { "temp", "set temperature °C", OFFSET(temp), AV_OPT_TYPE_DOUBLE, {.dbl=20.}, -50., 50., AF },
    { "yaw",    "angle for yaw (x-axis)",   OFFSET(yaw),   AV_OPT_TYPE_DOUBLE, {.dbl=0.}, -180., 180., AFT },
    { "pitch",  "angle for pitch (y-axis)", OFFSET(pitch), AV_OPT_TYPE_DOUBLE, {.dbl=0.}, -180., 180., AFT },
    { "roll",   "angle for roll (z-axis)",  OFFSET(roll),  AV_OPT_TYPE_DOUBLE, {.dbl=0.}, -180., 180., AFT },
    { "level",  "output level compensation", OFFSET(level), AV_OPT_TYPE_BOOL, {.i64=1}, 0, 1, AF },
    { "precision", "processing precision", OFFSET(precision), AV_OPT_TYPE_INT, {.i64=P_AUTO}, P_AUTO, NB_PTYPES-1, AF, "pre"},
    {   "auto",   "auto",                             0, AV_OPT_TYPE_CONST, {.i64=P_AUTO}, 0, 0, AF, "pre"},
    {   "single", "single floating-point precision",  0, AV_OPT_TYPE_CONST, {.i64=P_SINGLE}, 0, 0, AF, "pre"},
    {   "double", "double floating-point precision" , 0, AV_OPT_TYPE_CONST, {.i64=P_DOUBLE}, 0, 0, AF, "pre"},
    { "invert_x", "invert X", OFFSET(invert[D_X]), AV_OPT_TYPE_FLAGS, {.i64=0}, 0, 3, AF, "ix"},
    {   "odd",  "invert odd harmonics",  0, AV_OPT_TYPE_CONST, {.i64=1}, 0, 0, AF, "ix"},
    {   "even", "invert even harmonics", 0, AV_OPT_TYPE_CONST, {.i64=2}, 0, 0, AF, "ix"},
    { "invert_y", "invert Y", OFFSET(invert[D_Y]), AV_OPT_TYPE_FLAGS, {.i64=0}, 0, 3, AF, "iy"},
    {   "odd",  "invert odd harmonics",  0, AV_OPT_TYPE_CONST, {.i64=1}, 0, 0, AF, "iy"},
    {   "even", "invert even harmonics", 0, AV_OPT_TYPE_CONST, {.i64=2}, 0, 0, AF, "iy"},
    { "invert_z", "invert Z", OFFSET(invert[D_Z]), AV_OPT_TYPE_FLAGS, {.i64=0}, 0, 3, AF, "iz"},
    {   "odd",  "invert odd harmonics",  0, AV_OPT_TYPE_CONST, {.i64=1}, 0, 0, AF, "iz"},
    {   "even", "invert even harmonics", 0, AV_OPT_TYPE_CONST, {.i64=2}, 0, 0, AF, "iz"},
    { "invert_c", "circular invert", OFFSET(invert[D_C]), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, AF},
    { "x_odd",  "X odd harmonics gain",  OFFSET(gain[ODD][D_X]),  AV_OPT_TYPE_DOUBLE, {.dbl=1.}, 0, 2., AF },
    { "x_even", "X even harmonics gain", OFFSET(gain[EVEN][D_X]), AV_OPT_TYPE_DOUBLE, {.dbl=1.}, 0, 2., AF },
    { "y_odd",  "Y odd harmonics gain",  OFFSET(gain[ODD][D_Y]),  AV_OPT_TYPE_DOUBLE, {.dbl=1.}, 0, 2., AF },
    { "y_even", "Y even harmonics gain", OFFSET(gain[EVEN][D_Y]), AV_OPT_TYPE_DOUBLE, {.dbl=1.}, 0, 2., AF },
    { "z_odd",  "Z odd harmonics gain",  OFFSET(gain[ODD][D_Z]),  AV_OPT_TYPE_DOUBLE, {.dbl=1.}, 0, 2., AF },
    { "z_even", "Z even harmonics gain", OFFSET(gain[EVEN][D_Z]), AV_OPT_TYPE_DOUBLE, {.dbl=1.}, 0, 2., AF },
    { "c_gain", "set circular gain",     OFFSET(gain[0][D_C]),    AV_OPT_TYPE_DOUBLE, {.dbl=1.}, 0, 2., AF },
    { "f_dom",  "set forward dominance", OFFSET(dominance[0]),    AV_OPT_TYPE_DOUBLE, {.dbl=0.},-12,12.,AF },
    { "v_dom",  "set vertical dominance",OFFSET(dominance[1]),    AV_OPT_TYPE_DOUBLE, {.dbl=0.},-12,12.,AF },
    {NULL}
};

static double pythag(double a, double b)
{
    double absa = fabs(a);
    double absb = fabs(b);

    if (absa > absb) {
        return absa * sqrt(1.0+SQR(absb/absa));
    } else {
        if (absb == 0.0)
            return 0.0;
        else
            return absb * sqrt(1.0+SQR(absa/absb));
    }
}

static void mstep(int m, int n, double h, int l, int i,
                  double u[MAX_CHANNELS][MAX_CHANNELS])
{
    for (int j = l; j < n; j++) {
        double s = 0.0, f;

        for (int k = i; k < m; k++)
            s += u[k][i] * u[k][j];
        f = s / h;
        for (int k = i; k < m; k++)
            u[k][j] += f * u[k][i];
    }
}

static void svdcmp(AVFilterContext *ctx,
                   double u[MAX_CHANNELS][MAX_CHANNELS],
                   int m, int n,
                   double *q, double v[MAX_CHANNELS][MAX_CHANNELS])
{
    double e[MAX_CHANNELS] = { 0. };
    double g, x, s, f, h, z, c, y;
    double eps = 1e-15;
    const double tol = 1e-64 / eps;
    const int itmax = 50;
    int l, l1;

    av_assert0(1.0 + eps > 1.0);
    av_assert0(tol > 0.0);

    g = 0.0;
    x = 0.0;

    for (int i = 0; i < n; i++) {
        s = 0.0;

        e[i] = g;
        l = i + 1;
        for (int j = i; j < m; j++)
            s += SQR(u[j][i]);
        if (s <= tol) {
            g = 0.0;
        } else {
            f = u[i][i];
            g = f < 0.0 ? sqrt(s) : -sqrt(s);
            h = f * g - s;
            u[i][i] = f - g;
            mstep(m, n, h, l, i, u);
        }

        q[i] = g;
        s = 0.0;
        for (int j = l; j < n; j++)
            s += u[i][j]*u[i][j];
        if (s <= tol) {
            g = 0.0;
        } else {
            f = u[i][i+1];
            g = f < 0.0 ? sqrt(s) : -sqrt(s);
            h = f*g - s;
            u[i][i+1] = f-g;
            for (int j = l; j < n; j++)
                e[j] = u[i][j] / h;
            for (int j = l; j < m; j++) {
                s = 0.0;
                for (int k = l; k < n; k++)
                     s += u[j][k]*u[i][k];
                for (int k = l; k < n; k++)
                     u[j][k] += s * e[k];
            }
        }

        y = fabs(q[i])+fabs(e[i]);
        if (y > x)
            x = y;
    }

    for (int i = n - 1; i > -1; i--) {
        if (g != 0.0) {
            h = g*u[i][i+1];

            for (int j = l; j < n; j++)
                 v[j][i] = u[i][j] / h;

            for (int j = l; j < n; j++) {
                s = 0.0;
                for (int k = l; k < n; k++)
                    s += u[i][k] * v[k][j];
                for (int k = l; k < n; k++)
                    v[k][j] += s * v[k][i];
            }
        }

        for (int j = l; j < n; j++) {
            v[i][j] = 0.0;
            v[j][i] = 0.0;
        }

        v[i][i] = 1.0;
        g = e[i];
        l = i;
    }

    for (int i = n - 1; i > -1; i--) {
        l = i+1;
        g = q[i];
        for (int j = l; j < n; j++)
            u[i][j] = 0.0;
        if (g != 0.0) {
            h = u[i][i] * g;
            mstep(m, n, h, l, i, u);
            for (int j = i; j < m; j++)
                u[j][i] = u[j][i] / g;
        } else {
            for (int j = i; j < m; j++)
                u[j][i] = 0.0;
        }
        u[i][i] += 1.0;
    }

    eps = eps * x;
    for (int k = n-1; k >= 0; k--) {
        for (int iteration = 0; iteration < itmax; iteration++) {
            int goto_test_f_convergence = 0;

            for (l = k; l >= 0; l--) {
                goto_test_f_convergence = 0;
                if (fabs(e[l]) <= eps) {
                    goto_test_f_convergence = 1;
                    break;
                }

                av_assert0(l > 0);
                if (fabs(q[l-1]) <= eps)
                    break;
            }
            if (!goto_test_f_convergence) {
                c = 0.0;
                s = 1.0;
                l1 = l-1;
                av_assert0(l1 >= 0);
                for (int i = l; i <= k; i++) {
                    f = s*e[i];
                    e[i] = c*e[i];

                    if (fabs(f) <= eps)
                        break;

                    g = q[i];
                    h = pythag(f,g);
                    q[i] = h;
                    c = g/h;
                    s = -f/h;
                    for (int j = 0; j < m; j++) {
                        y = u[j][l1];
                        z = u[j][i];
                        u[j][l1] = y*c+z*s;
                        u[j][i] = -y*s+z*c;
                    }
                }
            }
            z = q[k];
            if (l == k) {
                if (z <= 0.0) {
                    q[k] = -z;
                    for (int j = 0; j < n; j++)
                        v[j][k] = -v[j][k];
                }
                break;
            }

            if (iteration >= itmax-1)
                break;

            x = q[l];
            av_assert0(k > 0);
            y = q[k-1];
            g = e[k-1];
            h = e[k];
            f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
            g = pythag(f,1.0);
            if (f < 0.)
                f = ((x-z)*(x+z)+h*(y/(f-g)-h))/x;
            else
                f = ((x-z)*(x+z)+h*(y/(f+g)-h))/x;
            c = 1.0;
            s = 1.0;
            for (int i = l+1; i < k+1; i++) {
                g = e[i];
                y = q[i];
                h = s*g;
                g = c*g;
                z = pythag(f,h);
                e[i-1] = z;
                c = f/z;
                s = h/z;
                f = x*c+g*s;
                g = -x*s+g*c;
                h = y*s;
                y = y*c;
                for (int j = 0; j < n; j++) {
                    x = v[j][i-1];
                    z = v[j][i];
                    v[j][i-1] = x*c+z*s;
                    v[j][i] = -x*s+z*c;
                }
                z = pythag(f,h);
                q[i-1] = z;
                c = f/z;
                s = h/z;
                f = c*g+s*y;
                x = -s*g+c*y;
                for (int j = 0; j < m; j++) {
                    y = u[j][i-1];
                    z = u[j][i];
                    u[j][i-1] = y*c+z*s;
                    u[j][i] = -y*s+z*c;
                }
            }

            e[l] = 0.0;
            e[k] = f;
            q[k] = x;
        }
    }

    av_log(ctx, AV_LOG_DEBUG, "um:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++)
            av_log(ctx, AV_LOG_DEBUG, "\t%g,", u[i][j]);
        av_log(ctx, AV_LOG_DEBUG, "\n");
    }

    av_log(ctx, AV_LOG_DEBUG, "wv:\n");
    for (int i = 0; i < n; i++)
        av_log(ctx, AV_LOG_DEBUG, "\t%g,", q[i]);
    av_log(ctx, AV_LOG_DEBUG, "\n");

    av_log(ctx, AV_LOG_DEBUG, "vm:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            av_log(ctx, AV_LOG_DEBUG, "\t%g", v[i][j]);
        av_log(ctx, AV_LOG_DEBUG, "\n");
    }
}

static void levelf(AmbisonicContext *s,
                   AVFrame *out, double level_tab[MAX_CHANNELS],
                   int nb_samples, int nb_channels)
{
    for (int ch = 0; ch < nb_channels; ch++) {
        float *dst = (float *)out->extended_data[ch];
        float mul = level_tab[ch];

        s->fdsp->vector_fmul_scalar(dst, dst, mul, FFALIGN(nb_samples, 16));
    }
}

static void leveld(AmbisonicContext *s,
                   AVFrame *out, double level_tab[MAX_CHANNELS],
                   int nb_samples, int nb_channels)
{
    for (int ch = 0; ch < nb_channels; ch++) {
        double *dst = (double *)out->extended_data[ch];
        double mul = level_tab[ch];

        s->fdsp->vector_dmul_scalar(dst, dst, mul, FFALIGN(nb_samples, 16));
    }
}

static void mirrorf(AmbisonicContext *s,
                    AVFrame *out, double mirror_mat[MAX_CHANNELS],
                    int nb_samples, int nb_channels, int *seq_tab)
{
    for (int ch = 0; ch < nb_channels; ch++) {
        float *dst = (float *)out->extended_data[ch];
        float mul = mirror_mat[seq_tab[ch]];

        s->fdsp->vector_fmul_scalar(dst, dst, mul, FFALIGN(nb_samples, 16));
    }
}

static void mirrord(AmbisonicContext *s,
                    AVFrame *out, double mirror_mat[MAX_CHANNELS],
                    int nb_samples, int nb_channels, int *seq_tab)
{
    for (int ch = 0; ch < nb_channels; ch++) {
        double *dst = (double *)out->extended_data[ch];
        double mul = mirror_mat[seq_tab[ch]];

        s->fdsp->vector_dmul_scalar(dst, dst, mul, FFALIGN(nb_samples, 16));
    }
}

static void rotatef(AmbisonicContext *s,
                    AVFrame *in, AVFrame *out,
                    double rotate_mat[MAX_CHANNELS][MAX_CHANNELS],
                    int nb_samples, int nb_channels, int *seq_tab)
{
    for (int ch = 0; ch < nb_channels; ch++) {
        const float *src = (const float *)in->extended_data[0];
        float *dst = (float *)out->extended_data[ch];
        float mul = rotate_mat[seq_tab[ch]][seq_tab[0]];

        s->fdsp->vector_fmul_scalar(dst, src, mul, FFALIGN(nb_samples, 16));

        for (int ch2 = 1; ch2 < nb_channels; ch2++) {
            const float *src = (const float *)in->extended_data[ch2];
            float mul = rotate_mat[seq_tab[ch]][seq_tab[ch2]];

            s->fdsp->vector_fmac_scalar(dst, src, mul, FFALIGN(nb_samples, 16));
        }
    }
}

static void rotated(AmbisonicContext *s,
                    AVFrame *in, AVFrame *out,
                    double rotate_mat[MAX_CHANNELS][MAX_CHANNELS],
                    int nb_samples, int nb_channels, int *seq_tab)
{
    for (int ch = 0; ch < nb_channels; ch++) {
        const double *src = (const double *)in->extended_data[0];
        double *dst = (double *)out->extended_data[ch];
        double mul = rotate_mat[seq_tab[ch]][seq_tab[0]];

        s->fdsp->vector_dmul_scalar(dst, src, mul, FFALIGN(nb_samples, 16));

        for (int ch2 = 1; ch2 < nb_channels; ch2++) {
            const double *src = (const double *)in->extended_data[ch2];
            double mul = rotate_mat[seq_tab[ch]][seq_tab[ch2]];

            s->fdsp->vector_dmac_scalar(dst, src, mul, FFALIGN(nb_samples, 16));
        }
    }
}

static void multiplyf(AmbisonicContext *s,
                      const double decode_matrix[MAX_CHANNELS][MAX_CHANNELS],
                      int inputs, int outputs,
                      int *seq_tab, const double *gains_tab,
                      int nb_channels, int max_channels,
                      AVFrame *in, AVFrame *out)
{
    for (int ch = 0; ch < outputs; ch++) {
        float *dst = (float *)out->extended_data[ch];

        for (int ch2 = 0; ch2 < FFMIN3(nb_channels, max_channels, inputs); ch2++) {
            const int index = FFMIN(seq_tab[ch2], nb_channels - 1);
            const float *src = (const float *)in->extended_data[index];
            const float gain = gains_tab ? gains_tab[ch2] : 1.f;
            const float mul = decode_matrix[ch][ch2] * gain;

            s->fdsp->vector_fmac_scalar(dst, src, mul, FFALIGN(in->nb_samples, 16));
        }
    }
}

static void multiplyd(AmbisonicContext *s,
                      const double decode_matrix[MAX_CHANNELS][MAX_CHANNELS],
                      int inputs, int outputs,
                      int *seq_tab, const double *gains_tab,
                      int nb_channels, int max_channels,
                      AVFrame *in, AVFrame *out)
{
    for (int ch = 0; ch < outputs; ch++) {
        double *dst = (double *)out->extended_data[ch];

        for (int ch2 = 0; ch2 < FFMIN3(nb_channels, max_channels, inputs); ch2++) {
            const int index = FFMIN(seq_tab[ch2], nb_channels - 1);
            const double *src = (const double *)in->extended_data[index];
            const double gain = gains_tab ? gains_tab[ch2] : 1.f;
            const double mul = decode_matrix[ch][ch2] * gain;

            s->fdsp->vector_dmac_scalar(dst, src, mul, FFALIGN(in->nb_samples, 16));
        }
    }
}

static void scalef(AmbisonicContext *s,
                   AVFrame *in, AVFrame *out,
                   double scale[MAX_CHANNELS],
                   int nb_samples, int nb_channels, int *seq_tab)
{
    for (int ch = 0; ch < nb_channels; ch++) {
        const float *src = (const float *)in->extended_data[ch];
        float *dst = (float *)out->extended_data[ch];
        float mul = scale[seq_tab[ch]];

        s->fdsp->vector_fmul_scalar(dst, src, mul, FFALIGN(nb_samples, 16));
    }
}

static void scaled(AmbisonicContext *s,
                   AVFrame *in, AVFrame *out,
                   double scale[MAX_CHANNELS],
                   int nb_samples, int nb_channels, int *seq_tab)
{
    for (int ch = 0; ch < nb_channels; ch++) {
        const double *src = (const double *)in->extended_data[ch];
        double *dst = (double *)out->extended_data[ch];
        double mul = scale[seq_tab[ch]];

        s->fdsp->vector_dmul_scalar(dst, src, mul, FFALIGN(nb_samples, 16));
    }
}

static int query_formats(AVFilterContext *ctx)
{
    AmbisonicContext *s = ctx->priv;
    AVFilterFormats *formats = NULL;
    AVFilterChannelLayouts *outlayouts = NULL;
    AVFilterChannelLayouts *inlayouts = NULL;
    AVChannelLayout *outlayout = (AVChannelLayout *)&ambisonic_tab[s->layout].outlayout;
    AVChannelLayout *inlayout = &(AVChannelLayout)AV_CHANNEL_LAYOUT_AMBISONIC_FIRST_ORDER;
    int ret = 0;

    if (s->precision == P_AUTO) {
        ret = ff_add_format(&formats, AV_SAMPLE_FMT_FLTP);
        if (ret)
            return ret;
        ret = ff_add_format(&formats, AV_SAMPLE_FMT_DBLP);
    } else if (s->precision == P_SINGLE) {
        ret = ff_add_format(&formats, AV_SAMPLE_FMT_FLTP);
    } else if (s->precision == P_DOUBLE) {
        ret = ff_add_format(&formats, AV_SAMPLE_FMT_DBLP);
    }
    if (ret)
        return ret;
    ret = ff_set_common_formats(ctx, formats);
    if (ret)
        return ret;

    ret = ff_add_channel_layout(&outlayouts, outlayout);
    if (ret)
        return ret;

    ret = ff_channel_layouts_ref(outlayouts, &ctx->outputs[0]->incfg.channel_layouts);
    if (ret)
        return ret;

    ret = ff_add_channel_layout(&inlayouts, inlayout);
    if (ret)
        return ret;

    ret = ff_channel_layouts_ref(inlayouts, &ctx->inputs[0]->outcfg.channel_layouts);
    if (ret)
        return ret;

    return ff_set_common_all_samplerates(ctx);
}

static void acn_to_level_order(int acn, int *level, int *order)
{
    *level = floor(sqrt(acn));
    *order = acn - *level * *level - *level;
}

static void calc_acn_sequence(AmbisonicContext *s)
{
    int *dst = s->seq_tab[M_ACN];

    for (int n = 0, i = 0; n <= s->order; n++) {
        for (int m = -n; m <= n; m++, i++)
            dst[i] = n * n + n + m;
    }
}

static void calc_fuma_sequence(AmbisonicContext *s)
{
    int *dst = s->seq_tab[M_FUMA];

    for (int n = 0, i = 0; n <= s->order; n++) {
        if (n < 2) {
            for (int m = -n; m <= n; m++)
                dst[i++] = n * n + 2 * (n - FFABS(m)) + (m < 0);
        } else {
            for (int m = -n; m <= n; m++)
                dst[i++] = SQR(n) + FFABS(m) * 2 - (m > 0);
        }
    }
}

static void calc_sid_sequence(AmbisonicContext *s)
{
    int *dst = s->seq_tab[M_SID];

    for (int n = 0, i = 0; n <= s->order; n++) {
        for (int m = -n; m <= n; m++, i++)
            dst[i] = n * n + 2 * (n - FFABS(m)) + (m < 0);
    }
}

static double factorial(int x)
{
    double prod = 1.;

    for (int i = 1; i <= x; i++)
        prod *= i;

    return prod;
}

static double n3d_norm(int i)
{
    int n, m;

    acn_to_level_order(i, &n, &m);

    return sqrt((2 * n + 1) * (2 - (m == 0)) * factorial(n - FFABS(m)) / factorial(n + FFABS(m)));
}

static double sn3d_norm(int i)
{
    int n, m;

    acn_to_level_order(i, &n, &m);

    return sqrt((2 - (m == 0)) * factorial(n - FFABS(m)) / factorial(n + FFABS(m)));
}

static void calc_sn3d_scaling(AmbisonicContext *s)
{
    double *dst = s->norm_tab[SN3D];

    for (int i = 0; i < s->max_channels; i++)
        dst[i] = 1.;
}

static void calc_n3d_scaling(AmbisonicContext *s)
{
    double *dst = s->norm_tab[N3D];

    for (int i = 0; i < s->max_channels; i++)
        dst[i] = n3d_norm(i) / sn3d_norm(i);
}

static void calc_fuma_scaling(AmbisonicContext *s)
{
    double *dst = s->norm_tab[FUMA];

    for (int i = 0; i < s->max_channels; i++) {
        dst[i] = sn3d_norm(i);

        switch (i) {
        case 0:
            dst[i] *= 1. / M_SQRT2;
        case 1:
        case 2:
        case 3:
        case 12:
        default:
            break;
        case 4:
            dst[i] *= 2. / sqrt(3.);
            break;
        case 5:
            dst[i] *= 2. / sqrt(3.);
            break;
        case 6:
            break;
        case 7:
            dst[i] *= 2. / sqrt(3.);
            break;
        case 8:
            dst[i] *= 2. / sqrt(3.);
            break;
        case 9:
            dst[i] *= sqrt(8. / 5.);
            break;
        case 10:
            dst[i] *= 3. / sqrt(5.);
            break;
        case 11:
            dst[i] *= sqrt(45. / 32.);
            break;
        case 13:
            dst[i] *= sqrt(45. / 32.);
            break;
        case 14:
            dst[i] *= 3. / sqrt(5.);
            break;
        case 15:
            dst[i] *= sqrt(8./5.);
            break;
        }
    }
}

static void multiply_mat(double out[3][3],
                         const double a[3][3],
                         const double b[3][3])
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double sum = 0.;

            for (int k = 0; k < 3; k++)
                sum += a[i][k] * b[k][j];

            out[i][j] = sum;
        }
    }
}

static double P(int i, int l, int mu, int m_, double R_1[3][3],
                double R_lm1[2 * MAX_ORDER + 1][2 * MAX_ORDER + 1])
{
    double ret = 0.;
    double ri1  = R_1[i + 1][2];
    double rim1 = R_1[i + 1][0];
    double ri0  = R_1[i + 1][1];

    if (m_ == -l) {
        ret = ri1 * R_lm1[mu + l - 1][0] + rim1 * R_lm1[mu + l- 1][2 * l - 2];
    } else {
        if (m_ == l)
            ret = ri1 * R_lm1[mu + l - 1][2 * l - 2] - rim1 * R_lm1[mu + l - 1][0];
        else
            ret = ri0 * R_lm1[mu + l - 1][m_ + l - 1];
    }
    return ret;
}

static double U(int l, int m, int n, double R_1[3][3],
                double R_lm1[2 * MAX_ORDER + 1][2 * MAX_ORDER + 1])
{
    return P(0, l, m, n, R_1, R_lm1);
}

static double V(int l, int m, int n, double R_1[3][3],
                double R_lm1[2 * MAX_ORDER + 1][2 * MAX_ORDER + 1])
{
    double ret = 0.;

    if (m == 0) {
        double p0 = P( 1, l,  1, n, R_1, R_lm1);
        double p1 = P(-1, l, -1, n, R_1, R_lm1);
        ret = p0+p1;
    } else {
        if (m > 0) {
            int d = (m == 1) ? 1 : 0;
            double p0 = P( 1, l,  m - 1, n, R_1, R_lm1);
            double p1 = P(-1, l, -m + 1, n, R_1, R_lm1);

            ret = p0 * sqrt(1 + d) - p1 * (1 - d);
        } else {
            int d = (m == -1) ? 1 : 0;
            double p0 = P( 1, l,  m + 1, n, R_1, R_lm1);
            double p1 = P(-1, l, -m - 1, n, R_1, R_lm1);

            ret = p0 * (1 - d) + p1 * sqrt(1 + d);
        }
    }
    return ret;
}

static double W(int l, int m, int n, double R_1[3][3],
                double R_lm1[2 * MAX_ORDER + 1][2 * MAX_ORDER + 1])
{
    double ret = 0.;

    if (m != 0) {
        if (m > 0) {
            double p0 = P( 1, l, m + 1, n, R_1, R_lm1);
            double p1 = P(-1, l,-m - 1, n, R_1, R_lm1);

            ret = p0 + p1;
        } else {
            double p0 = P( 1, l,  m - 1, n, R_1, R_lm1);
            double p1 = P(-1, l, -m + 1, n, R_1, R_lm1);

            ret = p0 - p1;
        }
    }

    return ret;
}

static void calc_rotation_mat(AVFilterContext *ctx,
                              AmbisonicContext *s,
                              double yaw, double pitch, double roll)
{
    double X[3][3] = {{0.}}, Y[3][3] = {{0.}}, Z[3][3] = {{0.}}, R[3][3], t[3][3];
    double R_lm1[2 * MAX_ORDER + 1][2 * MAX_ORDER + 1] = {{0.}};
    double R_1[3][3];

    yaw   = (M_PI / 180.) * yaw;
    pitch = (M_PI / 180.) * pitch;
    roll  = (M_PI / 180.) * roll;

    X[0][0] = 1.;
    X[1][1] = X[2][2] = cos(roll);
    X[1][2] = sin(roll);
    X[2][1] = -X[1][2];

    Y[0][0] = Y[2][2] = cos(pitch);
    Y[0][2] = sin(pitch);
    Y[2][0] = -Y[0][2];
    Y[1][1] = 1.;

    Z[0][0] = Z[1][1] = cos(yaw);
    Z[0][1] = sin(yaw);
    Z[1][0] = -Z[0][1];
    Z[2][2] = 1.;

    multiply_mat(t, X, Y);
    multiply_mat(R, t, Z);

    R_1[0][0] = R[1][1];
    R_1[0][1] = R[1][2];
    R_1[0][2] = R[1][0];
    R_1[1][0] = R[2][1];
    R_1[1][1] = R[2][2];
    R_1[1][2] = R[2][0];
    R_1[2][0] = R[0][1];
    R_1[2][1] = R[0][2];
    R_1[2][2] = R[0][0];

    memset(s->rotate_mat, 0, sizeof(s->rotate_mat));

    s->rotate_mat[0][0] = 1.;
    s->rotate_mat[1][1] = R_1[0][0];
    s->rotate_mat[1][2] = R_1[0][1];
    s->rotate_mat[1][3] = R_1[0][2];
    s->rotate_mat[2][1] = R_1[1][0];
    s->rotate_mat[2][2] = R_1[1][1];
    s->rotate_mat[2][3] = R_1[1][2];
    s->rotate_mat[3][1] = R_1[2][0];
    s->rotate_mat[3][2] = R_1[2][1];
    s->rotate_mat[3][3] = R_1[2][2];

    R_lm1[0][0] = R_1[0][0];
    R_lm1[0][1] = R_1[0][1];
    R_lm1[0][2] = R_1[0][2];
    R_lm1[1][0] = R_1[1][0];
    R_lm1[1][1] = R_1[1][1];
    R_lm1[1][2] = R_1[1][2];
    R_lm1[2][0] = R_1[2][0];
    R_lm1[2][1] = R_1[2][1];
    R_lm1[2][2] = R_1[2][2];

    for (int l = 2; l <= s->order; l++) {
        double R_l[2 * MAX_ORDER + 1][2 * MAX_ORDER + 1] = {{0.}};

        for (int m = -l; m <= l; m++) {
            for (int n = -l; n <= l; n++) {
                int d = (m == 0) ? 1 : 0;
                double denom = FFABS(n) == l ? (2 * l) * (2 * l - 1) : l * l - n * n;
                double u = sqrt((l * l - m * m) / denom);
                double v = sqrt((1. + d) * (l + FFABS(m) - 1.) * (l + FFABS(m)) / denom) * (1. - 2. * d) * 0.5;
                double w = sqrt((l - FFABS(m) - 1.)*(l - FFABS(m)) / denom) * (1. - d) * -0.5;

                if (u)
                    u *= U(l, m, n, R_1, R_lm1);
                if (v)
                    v *= V(l, m, n, R_1, R_lm1);
                if (w)
                    w *= W(l, m, n, R_1, R_lm1);

                R_l[m + l][n + l] = u + v + w;
            }
        }

        for (int i = 0; i < 2 * l + 1; i++) {
            for (int j = 0; j < 2 * l + 1; j++)
                s->rotate_mat[l * l + i][l * l + j] = R_l[i][j];
        }

        memcpy(R_lm1, R_l, sizeof(R_l));
    }

    av_log(ctx, AV_LOG_DEBUG, "rotation matrix:\n");
    for (int i = 0; i < SQR(s->order + 1); i++) {
        for (int j = 0; j < SQR(s->order + 1); j++) {
            if (fabs(s->rotate_mat[i][j]) < 1e-6f)
                s->rotate_mat[i][j] = 0.;
            av_log(ctx, AV_LOG_DEBUG, "\t%g", s->rotate_mat[i][j]);
        }
        av_log(ctx, AV_LOG_DEBUG, "\n");
    }
}

static void calc_mirror_mat(AmbisonicContext *s)
{
    for (int i = 0; i < s->max_channels; i++) {
        double gain = 1.;
        int level, order;

        acn_to_level_order(i, &level, &order);

        if (i == 0 || (!((level + order) & 1))) {
            gain *= s->gain[EVEN][D_Z];

            if (s->invert[D_Z] & 2)
                gain *= -1.;
        }

        if ((level + order) & 1) {
            gain *= s->gain[ODD][D_Z];

            if (s->invert[D_Z] & 1)
                gain *= -1.;
        }

        if (order >= 0) {
            gain *= s->gain[EVEN][D_Y];

            if (s->invert[D_Y] & 2)
                gain *= -1.;
        }

        if (order < 0) {
            gain *= s->gain[ODD][D_Y];

            if (s->invert[D_Y] & 1)
                gain *= -1.;
        }


        if (((order < 0) && (order & 1)) || ((order >= 0) && !(order & 1)) ) {
            gain *= s->gain[EVEN][D_X];

            if (s->invert[D_X] & 2)
                gain *= -1.;
        }

        if (((order < 0) && !(order & 1)) || ((order >= 0) && (order & 1))) {
            gain *= s->gain[ODD][D_X];

            if (s->invert[D_X] & 1)
                gain *= -1.;
        }

        if (level == order || level == -order) {
            gain *= s->gain[0][D_C];

            if (s->invert[D_C])
                gain *= -1.;
        }

        s->mirror_mat[i] = gain;
    }
}

static void near_field(AmbisonicContext *s, AVFrame *frame, int out, int add)
{
    for (int ch = 1; ch < frame->ch_layout.nb_channels; ch++) {
        int n, m;

        acn_to_level_order(ch, &n, &m);

        if (!s->nf_process[n - 1])
            break;

        s->nf_process[n - 1](&s->nf[out][ch], frame, ch, add, 1.);
    }
}

static void xover_processf(Xover *xover, const float *src, float *dst, int nb_samples)
{
    float b0 = xover->b[0];
    float b1 = xover->b[1];
    float b2 = xover->b[2];
    float a1 = xover->a[1];
    float a2 = xover->a[2];
    float w0 = xover->w[0];
    float w1 = xover->w[1];

    for (int i = 0; i < nb_samples; i++) {
        float in = src[i];
        float out = b0 * in + w0;

        w0 = b1 * in + w1 + a1 * out;
        w1 = b2 * in + a2 * out;

        dst[i] = out;
    }

    xover->w[0] = w0;
    xover->w[1] = w1;
}

static void xover_processd(Xover *xover, const double *src, double *dst, int nb_samples)
{
    double b0 = xover->b[0];
    double b1 = xover->b[1];
    double b2 = xover->b[2];
    double a1 = xover->a[1];
    double a2 = xover->a[2];
    double w0 = xover->w[0];
    double w1 = xover->w[1];

    for (int i = 0; i < nb_samples; i++) {
        double in = src[i];
        double out = b0 * in + w0;

        w0 = b1 * in + w1 + a1 * out;
        w1 = b2 * in + a2 * out;

        dst[i] = out;
    }

    xover->w[0] = w0;
    xover->w[1] = w1;
}

static void xoverf(AmbisonicContext *s,
                   AVFrame *in, AVFrame *lf, AVFrame *hf)
{
    for (int ch = 0; ch < in->ch_layout.nb_channels; ch++) {
        xover_processf(&s->xover[0][ch],
                       (const float *)in->extended_data[ch],
                       (float *)lf->extended_data[ch], in->nb_samples);

        xover_processf(&s->xover[1][ch],
                       (const float *)in->extended_data[ch],
                       (float *)hf->extended_data[ch], in->nb_samples);
    }
}

static void xoverd(AmbisonicContext *s,
                   AVFrame *in, AVFrame *lf, AVFrame *hf)
{
    for (int ch = 0; ch < in->ch_layout.nb_channels; ch++) {
        xover_processd(&s->xover[0][ch],
                       (const double *)in->extended_data[ch],
                       (double *)lf->extended_data[ch], in->nb_samples);

        xover_processd(&s->xover[1][ch],
                       (const double *)in->extended_data[ch],
                       (double *)hf->extended_data[ch], in->nb_samples);
    }
}

static void dominancef(AmbisonicContext *s,
                       AVFrame *in, AVFrame *out)
{
    const float Lf = sqrtf(powf(10.f, s->dominance[0] / 20.f));
    const float Lv = sqrtf(powf(10.f, s->dominance[1] / 20.f));
    const float af = Lf + 1.f / Lf;
    const float bf = Lf - 1.f / Lf;
    const float av = Lv + 1.f / Lv;
    const float bv = Lv - 1.f / Lv;
    const float *Wsrc = (const float *)in->extended_data[s->seq_tab[s->seq][A_W]];
    const float *Xsrc = (const float *)in->extended_data[s->seq_tab[s->seq][A_X]];
    const float *Zsrc = (const float *)in->extended_data[s->seq_tab[s->seq][A_Z]];
    float *Wdst = (float *)out->extended_data[s->seq_tab[s->seq][A_W]];
    float *Xdst = (float *)out->extended_data[s->seq_tab[s->seq][A_X]];
    float *Zdst = (float *)out->extended_data[s->seq_tab[s->seq][A_Z]];
    float W, X, Z;

    for (int n = 0; n < in->nb_samples; n++) {
        W = Wsrc[n];
        X = Xsrc[n];
        Z = Zsrc[n];

        Wdst[n] = 0.5f * af * W + sqrtf(0.125f) * bf * X;
        Xdst[n] = 0.5f * af * X + sqrtf(0.5f  ) * bf * W;

        W = Wdst[n];
        Wdst[n] = 0.5f * av * W + sqrtf(0.125f) * bv * Z;
        Zdst[n] = 0.5f * av * Z + sqrtf(0.5f  ) * bv * W;
    }
}

static void dominanced(AmbisonicContext *s,
                       AVFrame *in, AVFrame *out)
{
    const double Lf = sqrt(pow(10., s->dominance[0] / 20.));
    const double Lv = sqrt(pow(10., s->dominance[1] / 20.));
    const double af = Lf + 1. / Lf;
    const double bf = Lf - 1. / Lf;
    const double av = Lv + 1. / Lv;
    const double bv = Lv - 1. / Lv;
    const double *Wsrc = (const double *)in->extended_data[s->seq_tab[s->seq][A_W]];
    const double *Xsrc = (const double *)in->extended_data[s->seq_tab[s->seq][A_X]];
    const double *Zsrc = (const double *)in->extended_data[s->seq_tab[s->seq][A_Z]];
    double *Wdst = (double *)out->extended_data[s->seq_tab[s->seq][A_W]];
    double *Xdst = (double *)out->extended_data[s->seq_tab[s->seq][A_X]];
    double *Zdst = (double *)out->extended_data[s->seq_tab[s->seq][A_Z]];
    double W, X, Z;

    for (int n = 0; n < in->nb_samples; n++) {
        W = Wsrc[n];
        X = Xsrc[n];
        Z = Zsrc[n];

        Wdst[n] = 0.5 * af * W + sqrt(0.125) * bf * X;
        Xdst[n] = 0.5 * af * X + sqrt(0.5  ) * bf * W;

        W = Wdst[n];
        Wdst[n] = 0.5 * av * W + sqrt(0.125) * bv * Z;
        Zdst[n] = 0.5 * av * Z + sqrt(0.5  ) * bv * W;
    }
}

static void process_float(AVFilterContext *ctx,
                          AVFrame *in, AVFrame *out)
{
    AmbisonicContext *s = ctx->priv;

    scalef(s, in, s->sframe, s->norm_tab[s->norm],
           in->nb_samples, FFMIN(in->ch_layout.nb_channels, s->max_channels),
           s->seq_tab[s->seq]);

    dominancef(s, s->sframe, s->sframe);

    rotatef(s, s->sframe, s->rframe, s->rotate_mat,
            in->nb_samples, FFMIN(in->ch_layout.nb_channels, s->max_channels),
            s->seq_tab[s->seq]);

    mirrorf(s, s->rframe, s->mirror_mat,
            in->nb_samples, FFMIN(in->ch_layout.nb_channels, s->max_channels),
            s->seq_tab[s->seq]);

    if (s->near_field == NF_IN)
        near_field(s, s->rframe, 0, 0);

    if (s->xover_freq > 0.) {
        xoverf(s, s->rframe, s->frame2, s->rframe);

        multiplyf(s, s->decode_mat,
                  ambisonic_tab[s->layout].inputs,
                  ambisonic_tab[s->layout].speakers,
                  s->seq_tab[s->seq],
                  s->gains_tab[0],
                  in->ch_layout.nb_channels,
                  s->max_channels,
                  s->frame2, out);
    }

    multiplyf(s, s->decode_mat,
              ambisonic_tab[s->layout].inputs,
              ambisonic_tab[s->layout].speakers,
              s->seq_tab[s->seq],
              s->xover_freq > 0. ? s->gains_tab[1] : NULL,
              in->ch_layout.nb_channels,
              s->max_channels,
              s->rframe, out);

    if (s->near_field == NF_OUT)
        near_field(s, out, 1, 1);

    levelf(s, out, s->level_tab,
           out->nb_samples, out->ch_layout.nb_channels);
}

static void process_double(AVFilterContext *ctx,
                           AVFrame *in, AVFrame *out)
{
    AmbisonicContext *s = ctx->priv;

    scaled(s, in, s->sframe, s->norm_tab[s->norm],
           in->nb_samples, FFMIN(in->ch_layout.nb_channels, s->max_channels),
           s->seq_tab[s->seq]);

    dominanced(s, s->sframe, s->sframe);

    rotated(s, s->sframe, s->rframe, s->rotate_mat,
            in->nb_samples, FFMIN(in->ch_layout.nb_channels, s->max_channels),
            s->seq_tab[s->seq]);

    mirrord(s, s->rframe, s->mirror_mat,
            in->nb_samples, FFMIN(in->ch_layout.nb_channels, s->max_channels),
            s->seq_tab[s->seq]);

    if (s->near_field == NF_IN)
        near_field(s, s->rframe, 0, 0);

    if (s->xover_freq > 0.) {
        xoverd(s, s->rframe, s->frame2, s->rframe);

        multiplyd(s, s->decode_mat,
                  ambisonic_tab[s->layout].inputs,
                  ambisonic_tab[s->layout].speakers,
                  s->seq_tab[s->seq],
                  s->gains_tab[0],
                  in->ch_layout.nb_channels,
                  s->max_channels,
                  s->frame2, out);
    }

    multiplyd(s, s->decode_mat,
              ambisonic_tab[s->layout].inputs,
              ambisonic_tab[s->layout].speakers,
              s->seq_tab[s->seq],
              s->xover_freq > 0. ? s->gains_tab[1] : NULL,
              in->ch_layout.nb_channels,
              s->max_channels,
              s->rframe, out);

    if (s->near_field == NF_OUT)
        near_field(s, out, 1, 1);

    leveld(s, out, s->level_tab,
           out->nb_samples, out->ch_layout.nb_channels);
}

static double speed_of_sound(double temp)
{
    return 1.85325 * (643.95 * sqrt(((temp + 273.15) / 273.15))) * 1000.0 / (60. * 60.);
}

static void nfield1_init(NearField *nf, double radius,
                         double speed, double rate,
                         double gain)
{
    double omega = speed / (radius * rate);
    double b1 = omega * 0.5;
    double g1 = 1.0 + b1;

    nf->d[0] = (2.0 * b1) / g1;
    nf->g = gain / g1;
}

static void nfield1_processf(NearField *nf, AVFrame *frame, int ch, int add,
                             double gain)
{
    float *dst = (float *)frame->extended_data[ch];
    float g, z0, d0;

    g = nf->g * gain;
    z0 = nf->z[0];
    d0 = nf->d[0];

    for (int n = 0; n < frame->nb_samples; n++) {
        float x = g * dst[n] - d0 * z0;
        z0 += x;
        dst[n] = x + (add ? dst[n] : 0.f);
    }

    nf->z[0] = z0;
}

static void nfield1_processd(NearField *nf, AVFrame *frame, int ch, int add,
                             double gain)
{
    double *dst = (double *)frame->extended_data[ch];
    double g, z0, d0;

    g = nf->g * gain;
    z0 = nf->z[0];
    d0 = nf->d[0];

    for (int n = 0; n < frame->nb_samples; n++) {
        double x = g * dst[n] - d0 * z0;
        z0 += x;
        dst[n] = x + (add ? dst[n] : 0.);
    }

    nf->z[0] = z0;
}

static void near_field_init(AmbisonicContext *s, int out,
                            double speed, double rate, double gain)
{
    for (int ch = 1; ch < s->max_channels; ch++) {
        int n, m;

        acn_to_level_order(ch, &n, &m);

        if (!s->nf_init[n - 1])
            break;

        s->nf_init[n - 1](&s->nf[out][ch], 1., speed, rate, gain);
    }
}

static void calc_level_tab(AmbisonicContext *s, int layout)
{
    double max_distance = 0.;

    for (int spkr = 0; spkr < ambisonic_tab[s->layout].speakers; spkr++) {
        double spkr_distance = ambisonic_tab[s->layout].speakers_distance[spkr];

        if (spkr_distance > max_distance)
            max_distance = spkr_distance;
    }

    for (int spkr = 0; spkr < ambisonic_tab[s->layout].speakers; spkr++) {
        const double scale = s->level ? ambisonic_tab[s->layout].speakers_distance[spkr] / max_distance : 1.;

        s->level_tab[spkr] = scale;
    }
}

static void calc_pgains_tab(AmbisonicContext *s, int type)
{
    const int order = s->order;

    if (!type) {
        for (int level = 0; level < order + 1; level++)
            s->pgains[0][level] = s->pgains[1][level] = 1.0;
    } else if (type == 1 || type == 2) {
        const int components = type == 1 ? 2 * order + 1 : SQR(order + 1);
        const int speakers = ambisonic_tab[s->layout].speakers;
        double E_gain = 0;
        double g, g2 = 0.;

        for (int level = 0; level < order + 1; level++) {
            const double f = type == 1 ? 1 + (level > 0):
                                         2 * SQR(level) + 1;
            const double e = type == 1 ? gains_2d[order][level]:
                                         gains_3d[order][level];
            E_gain += SQR(e) * f;
        }

        if (s->pgtype == PT_ENERGY) {
            g2 = speakers / E_gain;
        } else if (s->pgtype == PT_RMS) {
            g2 = components / E_gain;
        } else if (s->pgtype == PT_AMP) {
            g2 = 1.;
        }

        g = sqrt(g2);

        for (int level = 0; level < order + 1; level++) {
            const double e = type == 1 ? gains_2d[order][level]:
                                         gains_3d[order][level];
            s->pgains[0][level] = 1.0;
            s->pgains[1][level] = g * e;
        }
    }
}

static void calc_gains_tab(AVFilterContext *ctx,
                           AmbisonicContext *s, double xover_ratio)
{
    const int inputs = ambisonic_tab[s->layout].inputs;
    const double xover_gain = pow(10., xover_ratio / 20.);

    for (int level = 0, ch = 0; level < s->order + 1; level++) {
        for (int i = 0; i < 1 + level * 2; i++, ch++) {
            const double lf_gain = s->pgains[0][level];
            const double hf_gain = s->pgains[1][level];

            s->gains_tab[0][ch] = lf_gain / xover_gain;
            s->gains_tab[1][ch] = hf_gain * xover_gain;
        }
    }

    av_log(ctx, AV_LOG_DEBUG, "gains tab:\n");
    for (int ch = 0; ch < inputs; ch++)
        av_log(ctx, AV_LOG_DEBUG, "\t%g", s->gains_tab[0][ch]);
    av_log(ctx, AV_LOG_DEBUG, "\n");
    for (int ch = 0; ch < inputs; ch++)
        av_log(ctx, AV_LOG_DEBUG, "\t%g", s->gains_tab[1][ch]);
    av_log(ctx, AV_LOG_DEBUG, "\n");
}

static void xover_init_input(Xover *xover, double freq, double rate, int hf)
{
    double k = tan(M_PI * freq / rate);
    double k2 = k * k;
    double d = k2 + 2. * k + 1.;

    if (hf) {
        xover->b[0] =  1. / d;
        xover->b[1] = -2. / d;
        xover->b[2] =  1. / d;
    } else {
        xover->b[0] = k2 / d;
        xover->b[1] = 2. * k2 / d;
        xover->b[2] = k2 / d;
    }

    xover->a[0] = 1.;
    xover->a[1] = -2 * (k2 - 1.) / d;
    xover->a[2] = -(k2 - 2 * k + 1.) / d;
}

static void xover_init(AmbisonicContext *s, double freq, double rate, int channels)
{
    for (int ch = 0; ch < channels; ch++) {
        xover_init_input(&s->xover[0][ch], freq, rate, 0);
        xover_init_input(&s->xover[1][ch], freq, rate, 1);
    }
}

static void calc_factor(double *factors,
                        int inputs,
                        double a, double e,
                        int m)
{
    const double cos_a = cos(a);
    const double sin_a = sin(a);
    const double cos_e = cos(e);
    const double sin_e = sin(e);
    const double sqrt3 = sqrt(3.);

    factors[A_W] = 1.;

    if (inputs <= 1)
        return;

    factors[A_Y] = sin_a * cos_e;
    factors[A_Z] = sin_e * m;
    factors[A_X] = cos_a * cos_e;

    if (inputs <= 4)
        return;

    factors[A_V] = sqrt3 * sin_a * SQR(cos_e) * cos_a;
    factors[A_T] = 0.25 * sqrt3 * (cos(2. * e - a) - cos(2. * e + a));
    factors[A_R] = (3./2.) * SQR(sin_e) - 0.5;
    factors[A_S] = 0.25 * sqrt3 * (sin(2. * e - a) + sin(2. * e + a));
    factors[A_U] = 0.5 * sqrt3 * SQR(cos_e)*cos(2. * a);

    if (inputs <= 9)
        return;

    factors[A_Q] = 0.25 * sqrt(10.) * (-4. * SQR(sin_a) + 3.) * sin_a * SQR(cos_e) * cos_e;
    factors[A_O] = sqrt(15.) * sin_e * sin_a * SQR(cos_e) * cos_a;
    factors[A_M] = 0.25 * sqrt(6.) * (5. * SQR(sin_e) - 1.) * sin_a * cos_e;
    factors[A_K] = 0.5 * (5*SQR(sin_e) - 3)*sin_e;
    factors[A_L] = 0.25 * sqrt(6.) * (5. * SQR(sin_e) - 1.) * cos_e * cos_a;
    factors[A_N] = 0.5 * sqrt(15.) * sin_e * SQR(cos_e) * cos(2. * a);
    factors[A_P] = 0.25 * sqrt(10.)*(-4. * SQR(sin_a) + 1.) * SQR(cos_e) * cos_e * cos_a;
}

static void calc_factors(AVFilterContext *ctx,
                         AmbisonicContext *s)
{
    const int speakers = ambisonic_tab[s->layout].speakers;
    const double *elevation = ambisonic_tab[s->layout].speakers_elevation;
    const double *azimuth = ambisonic_tab[s->layout].speakers_azimuth;
    const int inputs = ambisonic_tab[s->layout].inputs;

    if (elevation) {
        for (int ch = 0; ch < speakers; ch++)
            calc_factor(s->decode_mat[ch], inputs,
                        (M_PI / 180.) * azimuth[ch] * -1,
                        (M_PI / 180.) * elevation[ch], 1);
    } else {
        for (int ch = 0; ch < speakers; ch++)
            calc_factor(s->decode_mat[ch], inputs,
                        (M_PI / 180.) * azimuth[ch] * -1,
                        0., 0);
    }

    av_log(ctx, AV_LOG_DEBUG, "factors matrix:\n");
    for (int i = 0; i < speakers; i++) {
        for (int j = 0; j < inputs; j++)
            av_log(ctx, AV_LOG_DEBUG, "\t%g", s->decode_mat[i][j]);
        av_log(ctx, AV_LOG_DEBUG, "\n");
    }
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AmbisonicContext *s = ctx->priv;
    const int type = ambisonic_tab[s->layout].type;
    const int inputs = ambisonic_tab[s->layout].inputs;
    const int speakers = ambisonic_tab[s->layout].speakers;
    const double matching = s->matching;
    double w_mean = 0.;

    s->order = ambisonic_tab[s->layout].order;
    s->max_channels = SQR(s->order + 1);

    if (s->near_field == NF_AUTO)
        s->near_field = ambisonic_tab[s->layout].near_field;
    if (s->xover_freq < 0)
        s->xover_freq = ambisonic_tab[s->layout].xover;

    calc_factors(ctx, s);

    memset(s->v, 0, sizeof(s->v));
    memset(s->w, 0, sizeof(s->w));

    memcpy(s->u, s->decode_mat, sizeof(s->u));
    svdcmp(ctx, s->u, speakers, inputs, s->w, s->v);

    for (int x = 0; x < inputs; x++) {
        s->w[x] = s->w[x] > 1e-9 ? 1. / s->w[x] : 0.f;
        w_mean += s->w[x];
    }

    w_mean /= inputs;
    for (int x = 0; x < inputs; x++)
        s->w[x] = s->w[x] * (1. - matching) + matching * w_mean;

    for (int y = 0; y < inputs; y++) {
        for (int x = 0; x < inputs; x++)
            s->v[y][x] *= s->w[x];
    }

    for (int y = 0; y < speakers; y++) {
        for (int x = 0; x < inputs; x++) {
            double sum = 0.;

            for (int z = 0; z < inputs; z++)
                sum += s->v[x][z] * s->u[y][z];
            s->decode_mat[y][x] = sum;
        }
    }

    av_log(ctx, AV_LOG_DEBUG, "decode matrix:\n");
    for (int y = 0; y < speakers; y++) {
        for (int x = 0; x < inputs; x++)
            av_log(ctx, AV_LOG_DEBUG, "\t%g", s->decode_mat[y][x]);
        av_log(ctx, AV_LOG_DEBUG, "\n");
    }

    calc_sn3d_scaling(s);
    calc_n3d_scaling(s);
    calc_fuma_scaling(s);

    calc_acn_sequence(s);
    calc_fuma_sequence(s);
    calc_sid_sequence(s);

    near_field_init(s, 0, speed_of_sound(s->temp), outlink->sample_rate, 1.);
    near_field_init(s, 1, speed_of_sound(s->temp), outlink->sample_rate, 1.);

    calc_rotation_mat(ctx, s, s->yaw, s->pitch, s->roll);
    calc_mirror_mat(s);
    calc_level_tab(s, s->layout);
    calc_pgains_tab(s, type);
    calc_gains_tab(ctx, s, s->xover_ratio);
    xover_init(s, s->xover_freq, outlink->sample_rate, s->max_channels);

    switch (s->precision) {
    case P_AUTO:
        s->nf_process[0] = outlink->format == AV_SAMPLE_FMT_FLTP ? nfield1_processf : nfield1_processd;
        s->process = outlink->format == AV_SAMPLE_FMT_FLTP ? process_float : process_double;
        break;
    case P_SINGLE:
        s->nf_process[0] = nfield1_processf;
        s->process = process_float;
        break;
    case P_DOUBLE:
        s->nf_process[0] = nfield1_processd;
        s->process = process_double;
        break;
    default: av_assert0(0);
    }

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AmbisonicContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *out;

    if (!s->rframe || s->rframe->nb_samples < in->nb_samples) {
        av_frame_free(&s->sframe);
        av_frame_free(&s->rframe);
        av_frame_free(&s->frame2);
        s->sframe = ff_get_audio_buffer(inlink, in->nb_samples);
        s->rframe = ff_get_audio_buffer(inlink, in->nb_samples);
        s->frame2 = ff_get_audio_buffer(inlink, in->nb_samples);
        if (!s->sframe || !s->rframe || !s->frame2) {
            av_frame_free(&s->sframe);
            av_frame_free(&s->rframe);
            av_frame_free(&s->frame2);
            av_frame_free(&in);
            return AVERROR(ENOMEM);
        }
    }

    out = ff_get_audio_buffer(outlink, in->nb_samples);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(out, in);

    s->process(ctx, in, out);

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);

}

static av_cold int init(AVFilterContext *ctx)
{
    AmbisonicContext *s = ctx->priv;

    s->nf_init[0] = nfield1_init;

    s->fdsp = avpriv_float_dsp_alloc(0);
    if (!s->fdsp)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    AmbisonicContext *s = ctx->priv;

    av_freep(&s->fdsp);
    av_frame_free(&s->sframe);
    av_frame_free(&s->rframe);
    av_frame_free(&s->frame2);
}

static const AVFilterPad inputs[] = {
    {
        .name           = "default",
        .type           = AVMEDIA_TYPE_AUDIO,
        .filter_frame   = filter_frame,
    },
};
static const AVFilterPad outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_AUDIO,
        .config_props = config_output,
    },
};

static int process_command(AVFilterContext *ctx, const char *cmd, const char *args,
                           char *res, int res_len, int flags)
{
    AmbisonicContext *s = ctx->priv;
    const double yaw = s->yaw, pitch = s->pitch, roll = s->roll;
    int ret;

    ret = ff_filter_process_command(ctx, cmd, args, res, res_len, flags);
    if (ret < 0)
        return ret;
    if (yaw != s->yaw || pitch != s->pitch || roll != s->roll)
        calc_rotation_mat(ctx, s, s->yaw, s->pitch, s->roll);

    return 0;
}

AVFILTER_DEFINE_CLASS(ambisonic);

const AVFilter ff_af_ambisonic = {
    .name            = "ambisonic",
    .description     = NULL_IF_CONFIG_SMALL("Ambisonic decoder"),
    .priv_size       = sizeof(AmbisonicContext),
    .priv_class      = &ambisonic_class,
    .init            = init,
    .uninit          = uninit,
    FILTER_QUERY_FUNC(query_formats),
    FILTER_INPUTS(inputs),
    FILTER_OUTPUTS(outputs),
    .process_command = process_command,
};
