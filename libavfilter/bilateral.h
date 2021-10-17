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

#ifndef AVFILTER_BILATERAL_H
#define AVFILTER_BILATERAL_H

#include "avfilter.h"

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

    void (*sum_and_div_float)(float *dst,
                              const float *num0, const float *num1,
                              const float *den0, const float *den1, int width);

    int (*input_fun)(AVFilterContext *ctx, void *arg,
                     int jobnr, int nb_jobs);

    int (*output_fun)(AVFilterContext *ctx, void *arg,
                      int jobnr, int nb_jobs);

    int (*hderivate_pass)(AVFilterContext *ctx, void *arg,
                          int jobnr, int nb_jobs);

    int (*vderivate_pass)(AVFilterContext *ctx, void *arg,
                          int jobnr, int nb_jobs);

    int stride[4];

    uint16_t *hderivative[4];
    uint16_t *vderivative[4];

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

void ff_bilateral_init_x86(BilateralContext *s);

#endif /* AVFILTER_BILATERAL_H */
